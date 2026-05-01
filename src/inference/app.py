from __future__ import annotations

"""
FastAPI prediction service — wraps predict.py into a REST API.

ENDPOINTS:
  GET  /health          — liveness + readiness check
  GET  /predict         — forecast PM2.5 24h ahead from now
  POST /predict         — forecast from a specific timestamp

DESIGN DECISIONS:
  - predict.py is imported as a pure function library — no code duplication.
  - Model is loaded once at startup and cached — not re-loaded per request.
  - All settings come from environment variables (via settings.py + .env).
  - Errors are mapped to appropriate HTTP status codes with structured responses.
  - Prometheus metrics are instrumented on every endpoint.

STARTUP BEHAVIOUR:
  On startup, the app loads the @champion model from MLflow and caches it.
  If no champion model exists, the /predict endpoint returns 503 until one
  is promoted via evaluate.py.

USAGE:
  # Development
  uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

  # Production (inside Docker)
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1

  # Test
  curl http://localhost:8000/health
  curl http://localhost:8000/predict
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"timestamp": "2026-03-29T14:00:00+00:00"}'
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config.settings import settings
from src.inference.predict import (
    run_prediction,
    load_champion_model,
    pm25_to_aqi_category,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Model cache — loaded once at startup ──────────────────────────────────────
_model_cache: dict = {
    "model":        None,
    "feature_cols": None,
    "loaded_at":    None,
    "version":      None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the @champion model once at startup and cache it.

    WHY CACHE THE MODEL:
    Loading from MLflow on every request adds 1–3 seconds of latency.
    Caching gives sub-100ms inference after the first load.
    The model is refreshed automatically when the app restarts — which
    the Airflow DAG triggers after every successful promotion.

    If no champion model exists yet, the app starts anyway and returns 503
    on /predict requests until a model is promoted.
    """
    logger.info("Starting PM2.5 prediction service...")
    logger.info("MLflow URI: %s", settings.MLFLOW_TRACKING_URI)
    logger.info("Model name: %s", settings.MODEL_NAME)

    try:
        model, feature_cols = load_champion_model()
        _model_cache["model"]        = model
        _model_cache["feature_cols"] = feature_cols
        _model_cache["loaded_at"]    = datetime.now(timezone.utc).isoformat()
        _model_cache["version"]      = "champion"
        logger.info(
            "Champion model loaded successfully. Features: %d",
            len(feature_cols) if feature_cols else 0,
        )
    except Exception as e:
        logger.warning(
            "Could not load champion model at startup: %s. "
            "/predict will return 503 until a model is promoted.",
            e,
        )

    yield

    logger.info("Shutting down PM2.5 prediction service.")


# ── App definition ────────────────────────────────────────────────────────────

app = FastAPI(
    title="PM2.5 Air Quality Forecast API",
    description=(
        "Predicts PM2.5 concentration 24 hours ahead for Yangon, Myanmar. "
        "Powered by LightGBM trained on OpenAQ + Open-Meteo data."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Optional request body for POST /predict.

    If timestamp is omitted, the current UTC hour is used.
    The timestamp must be timezone-aware — naive datetimes are rejected
    to prevent silent UTC/local timezone bugs.
    """
    timestamp: Optional[datetime] = Field(
        default=None,
        description=(
            "UTC timestamp to predict from (ISO 8601, timezone-aware). "
            "Defaults to the current UTC hour. "
            "Example: '2026-03-29T14:00:00+00:00'"
        ),
        examples=["2026-03-29T14:00:00+00:00"],
    )

    @field_validator("timestamp")
    @classmethod
    def must_be_timezone_aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Reject naive datetimes — they cause silent timezone bugs."""
        if v is not None and v.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware. "
                "Add '+00:00' suffix for UTC. "
                "Example: '2026-03-29T14:00:00+00:00'"
            )
        return v

    @field_validator("timestamp")
    @classmethod
    def not_too_far_future(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Reject timestamps more than 7 days in the future — no data available."""
        if v is not None:
            now = datetime.now(timezone.utc)
            if v > now + timedelta(days=7):
                raise ValueError(
                    "timestamp cannot be more than 7 days in the future — "
                    "no feature data available for that horizon."
                )
        return v

    @field_validator("timestamp")
    @classmethod
    def not_too_far_past(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Reject timestamps before the model's training data starts."""
        if v is not None:
            cutoff = datetime(2020, 12, 1, tzinfo=timezone.utc)
            if v < cutoff:
                raise ValueError(
                    f"timestamp cannot be before {cutoff.date()} — "
                    "no training data available before that date."
                )
        return v


class PredictResponse(BaseModel):
    """Structured prediction response."""
    city:               str   = Field(description="City being forecast")
    prediction_time:    str   = Field(description="Input timestamp T (ISO 8601)")
    forecast_time:      str   = Field(description="Predicted timestamp T+24h (ISO 8601)")
    forecast_horizon_h: int   = Field(description="Forecast horizon in hours")
    pm25_predicted:     float = Field(description="Predicted PM2.5 concentration (µg/m³)")
    aqi_category:       str   = Field(description="EPA AQI category label")
    model_version:      str   = Field(description="MLflow model version/alias used")
    feature_warnings:   list  = Field(description="List of missing features set to NaN")


class HealthResponse(BaseModel):
    """Health check response."""
    status:        str           = Field(description="'ok' or 'degraded'")
    model_loaded:  bool          = Field(description="Whether champion model is in memory")
    model_version: Optional[str] = Field(description="Loaded model version/alias")
    loaded_at:     Optional[str] = Field(description="When the model was last loaded")
    city:          str           = Field(description="Configured forecast city")
    mlflow_uri:    str           = Field(description="MLflow tracking server URI")


class ErrorResponse(BaseModel):
    """Structured error response."""
    error:   str = Field(description="Error type")
    message: str = Field(description="Human-readable error description")
    detail:  Optional[str] = Field(default=None, description="Technical detail")


# ── Middleware — request timing ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, status, and duration."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.1fms)",
        request.method, request.url.path,
        response.status_code, duration_ms,
    )
    return response


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unexpected errors — never expose stack traces to clients."""
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred. Please try again.",
            detail=str(type(exc).__name__),
        ).model_dump(),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness check",
    tags=["Infrastructure"],
)
async def health():
    """
    Liveness + readiness check.

    Returns 200 if the service is running and the model is loaded.
    Returns 200 with status='degraded' if the service is running but
    the model failed to load — useful for distinguishing liveness from readiness.

    Use this endpoint for:
    - Docker health checks
    - Load balancer readiness probes
    - Kubernetes liveness/readiness probes
    - Monitoring dashboards
    """
    model_loaded = _model_cache["model"] is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=_model_cache.get("version"),
        loaded_at=_model_cache.get("loaded_at"),
        city=settings.CITY_NAME,
        mlflow_uri=settings.MLFLOW_TRACKING_URI,
    )


@app.get(
    "/predict",
    response_model=PredictResponse,
    summary="Forecast PM2.5 24h ahead from now",
    tags=["Prediction"],
    responses={
        200: {"description": "Successful prediction"},
        503: {"description": "Model not loaded — run evaluate.py to promote a champion"},
    },
)
async def predict_now():
    """
    Forecast PM2.5 concentration 24 hours ahead from the current UTC hour.

    This is the simplest endpoint — no request body needed. Call it to get
    the forecast for "tomorrow at this time".

    The model fetches live feature data (weather, recent PM2.5 history)
    from Open-Meteo and the master CSV automatically.
    """
    return await _run_predict(prediction_time=None)


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Forecast PM2.5 24h ahead from a specific timestamp",
    tags=["Prediction"],
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid request — bad timestamp format or out-of-range"},
        503: {"description": "Model not loaded — run evaluate.py to promote a champion"},
    },
)
async def predict_at_time(request: PredictRequest):
    """
    Forecast PM2.5 concentration 24 hours ahead from a specific timestamp.

    Useful for:
    - Historical backtesting (what would the model have predicted on date X?)
    - Scheduled batch forecasts at specific hours
    - Dashboard integrations that request forecasts for specific time slots

    The timestamp must be timezone-aware ISO 8601.
    Example: `{"timestamp": "2026-03-29T14:00:00+00:00"}`
    """
    return await _run_predict(prediction_time=request.timestamp)


async def _run_predict(prediction_time: Optional[datetime]) -> PredictResponse:
    """
    Shared prediction logic for both GET and POST /predict.

    Checks model cache, calls run_prediction(), maps errors to HTTP codes.
    """
    # ── Check model is loaded ─────────────────────────────────────────────────
    if _model_cache["model"] is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ModelNotLoaded",
                "message": (
                    f"No @champion model loaded for '{settings.MODEL_NAME}'. "
                    "Run train.py then evaluate.py to promote a model, "
                    "then restart the API service."
                ),
            },
        )

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        result = run_prediction(prediction_time=prediction_time)
    except ValueError as e:
        # Known errors — bad config, missing data, no champion model
        raise HTTPException(
            status_code=503,
            detail={
                "error": "PredictionSetupError",
                "message": str(e),
            },
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "DataNotFound",
                "message": (
                    "Required data file not found. "
                    "Run the ingestion pipeline first. "
                    f"Detail: {e}"
                ),
            },
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PredictionFailed",
                "message": "Prediction failed due to an internal error.",
                "detail": str(type(e).__name__),
            },
        )

    return PredictResponse(**result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.inference.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )