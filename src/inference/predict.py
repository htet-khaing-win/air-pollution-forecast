from __future__ import annotations

"""
Inference pipeline — fetches live features and returns a PM2.5 forecast.

WHAT THIS MODULE DOES:
1. Loads the @champion model from the MLflow Model Registry
2. Fetches recent PM2.5 history from the master CSV (for lag features)
3. Fetches current weather from Open-Meteo (for weather features)
4. Fetches current OWM air pollution estimate (for owm_* features)
5. Builds the feature vector for the current hour
6. Returns a prediction: PM2.5 µg/m³ at T+FORECAST_HORIZON_HOURS (24h ahead)
   along with the corresponding AQI category

FORECAST CONTRACT:
  Input:  current timestamp T (defaults to now)
  Output: predicted PM2.5 at T + 24 hours

WHY WE LOAD FROM MLFLOW REGISTRY (@champion alias):
The model artifact in MLflow is the exact object that was registered after
training — same weights, same feature order. Loading by alias means predict.py
automatically uses the promoted model without any code changes when a new
champion is promoted by evaluate.py.

WHY WE NEED THE MASTER CSV FOR INFERENCE:
The model was trained on lag features (pm25_lag_24h, pm25_lag_48h, pm25_lag_72h)
and rolling window stats. At inference time we need the last 72+ hours of real
PM2.5 readings to compute these features. The master CSV is the source of truth
for historical readings — the same data the model was trained on.

FEATURE VECTOR CONSTRUCTION:
We reproduce the exact same feature engineering steps as features.py but for
a single row (the current hour). The feature names and order must match exactly
what the model was trained on — we load the feature list from the MLflow run
to guarantee this.

USAGE:
  python -m src.inference.predict                    # predict from now
  python -m src.inference.predict --timestamp "2026-03-22 14:00:00+00:00"
"""

import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from config.settings import settings
from src.ingestion.openmeteo import fetch_openmeteo_data
from src.ingestion.openweather import fetch_owm_air_pollution_history
from src.preprocessing.features import (
    AQI_BREAKPOINTS,
    add_time_features,
    add_interaction_features,
)

logger = logging.getLogger(__name__)

# ── Columns dropped before training — must match train.py exactly ────────────
ALWAYS_DROP_COLS: list[str] = ["pm10", "co", "no2", "o3"]
NON_FEATURE_COLS: list[str] = [
    "timestamp", "pm25", "aqi_category", "aqi_numeric", "pm25_owm_delta",
]

# ── Lag hours used at training time — must match features.py LAG_HOURS ───────
LAG_HOURS: list[int] = [1, 3, 6, 12, 24, 48, 72]

# ── Rolling windows — must match features.py ROLLING_WINDOWS ─────────────────
ROLLING_WINDOWS: list[int] = [6, 12, 24]

# ── Minimum history needed to compute all features ────────────────────────────
# 72h lag + 24h rolling window = need at least 96 hours of history
# Short lags (1h, 3h etc) are safe — they reference past observations
MIN_HISTORY_HOURS: int = max(LAG_HOURS) + max(ROLLING_WINDOWS)


def load_champion_model():
    """
    Load the @champion model from the MLflow Model Registry.

    Returns:
        Tuple of (model, feature_cols) where feature_cols is the ordered
        list of feature names the model was trained on.

    Raises:
        ValueError if no @champion model exists — run evaluate.py first.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        mv = client.get_model_version_by_alias(settings.MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        raise ValueError(
            f"No @champion model found for '{settings.MODEL_NAME}'. "
            "Run train.py then evaluate.py first."
        )

    model_uri = f"models:/{settings.MODEL_NAME}@champion"
    model = mlflow.lightgbm.load_model(model_uri)

    # Retrieve the feature column list logged during training
    run = client.get_run(mv.run_id)
    feature_cols_str = run.data.params.get("feature_cols", "")

    # feature_cols was logged as a Python list repr — parse it safely
    try:
        import ast
        feature_cols = ast.literal_eval(feature_cols_str)
    except (ValueError, SyntaxError):
        logger.warning(
            "Could not parse feature_cols from MLflow run. "
            "Will infer from model at prediction time."
        )
        feature_cols = None

    logger.info(
        "Loaded @champion model: version=%s  run_id=%s  features=%s",
        mv.version, mv.run_id,
        f"{len(feature_cols)} columns" if feature_cols else "unknown",
    )
    return model, feature_cols


def load_recent_pm25(n_hours: int = MIN_HISTORY_HOURS) -> pd.Series:
    """
    Load the most recent PM2.5 readings from the master CSV.

    We need at least MIN_HISTORY_HOURS (96h) of history to compute
    lag and rolling features for the current prediction.

    Args:
        n_hours: Number of recent hours to load.

    Returns:
        Series of PM2.5 values indexed by UTC timestamp, sorted ascending.
        Returns empty Series if master CSV not found.
    """
    csv_path = settings.RAW_DATA_DIR / settings.RAW_FILENAME

    if not csv_path.exists():
        logger.error(
            "Master CSV not found at %s. Run ingest.py first.", csv_path
        )
        return pd.Series(dtype=float)

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").tail(n_hours * 2)  # load extra for safety

    if "pm25" not in df.columns or df["pm25"].dropna().empty:
        logger.error("No pm25 values found in master CSV.")
        return pd.Series(dtype=float)

    series = df.set_index("timestamp")["pm25"].dropna()
    series.index = pd.to_datetime(series.index, utc=True)
    series = series.sort_index().tail(n_hours)

    logger.info(
        "Loaded %d hours of PM2.5 history (%s → %s)",
        len(series), series.index.min(), series.index.max(),
    )
    return series


def build_feature_row(
    prediction_time: datetime,
    pm25_history: pd.Series,
    df_weather: pd.DataFrame,
    df_owm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for the prediction timestamp.

    Reproduces the exact feature engineering from features.py but for
    one row only. The feature names must match what the model was trained on.

    Args:
        prediction_time: The timestamp T at which we're making a prediction
                         (we predict pm25 at T + FORECAST_HORIZON_HOURS).
        pm25_history:    Recent PM2.5 readings indexed by UTC timestamp.
        df_weather:      Weather DataFrame from Open-Meteo (recent hours).
        df_owm:          OWM air pollution DataFrame (recent hours).

    Returns:
        Single-row DataFrame with all feature columns.
    """
    row: dict = {"timestamp": prediction_time}

    # ── Lag features ──────────────────────────────────────────────────────────
    for lag in LAG_HOURS:
        lag_time = prediction_time - timedelta(hours=lag)
        # Find closest reading within ±30 minutes of the lag timestamp
        if not pm25_history.empty:
            diffs = abs(pm25_history.index - lag_time)
            min_diff = diffs.min()
            if min_diff <= timedelta(minutes=30):
                row[f"pm25_lag_{lag}h"] = pm25_history[diffs == min_diff].iloc[0]
            else:
                row[f"pm25_lag_{lag}h"] = np.nan
        else:
            row[f"pm25_lag_{lag}h"] = np.nan

    # ── Rolling window features ───────────────────────────────────────────────
    for window in ROLLING_WINDOWS:
        window_start = prediction_time - timedelta(hours=window)
        # Get readings in the window (excluding current hour — closed='left')
        mask = (pm25_history.index >= window_start) & (pm25_history.index < prediction_time)
        window_vals = pm25_history[mask]

        if len(window_vals) > 0:
            row[f"pm25_roll_{window}h_mean"] = window_vals.mean()
            row[f"pm25_roll_{window}h_std"]  = window_vals.std() if len(window_vals) > 1 else 0.0
            row[f"pm25_roll_{window}h_max"]  = window_vals.max()
        else:
            row[f"pm25_roll_{window}h_mean"] = np.nan
            row[f"pm25_roll_{window}h_std"]  = np.nan
            row[f"pm25_roll_{window}h_max"]  = np.nan

    # ── Weather features ──────────────────────────────────────────────────────
    weather_cols = ["temperature", "humidity", "wind_speed", "pressure"]
    if not df_weather.empty:
        df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"], utc=True)
        df_weather = df_weather.sort_values("timestamp")
        # Get the most recent weather reading at or before prediction_time
        past_weather = df_weather[df_weather["timestamp"] <= prediction_time]
        if not past_weather.empty:
            latest_weather = past_weather.iloc[-1]
            for col in weather_cols:
                row[col] = latest_weather.get(col, np.nan)
        else:
            for col in weather_cols:
                row[col] = np.nan
    else:
        for col in weather_cols:
            row[col] = np.nan

    # ── OWM features ─────────────────────────────────────────────────────────
    owm_cols = ["owm_pm25", "owm_pm10", "owm_co", "owm_no2", "owm_o3"]
    if not df_owm.empty:
        df_owm["timestamp"] = pd.to_datetime(df_owm["timestamp"], utc=True)
        past_owm = df_owm[df_owm["timestamp"] <= prediction_time]
        if not past_owm.empty:
            latest_owm = past_owm.iloc[-1]
            for col in owm_cols:
                row[col] = latest_owm.get(col, np.nan)
        else:
            for col in owm_cols:
                row[col] = np.nan
    else:
        for col in owm_cols:
            row[col] = np.nan

    # ── Build single-row DataFrame and add time + interaction features ────────
    df_row = pd.DataFrame([row])
    df_row = add_time_features(df_row)
    df_row = add_interaction_features(df_row)

    return df_row


def pm25_to_aqi_category(pm25: float) -> str:
    """Convert a PM2.5 µg/m³ value to its EPA AQI category label."""
    if pd.isna(pm25) or pm25 < 0:
        return "Unknown"
    for lo, hi, label in AQI_BREAKPOINTS:
        if lo <= pm25 <= hi:
            return label
    return "Hazardous"


def run_prediction(
    prediction_time: datetime | None = None,
) -> dict:
    """
    Full inference pipeline: load model → fetch features → predict.

    Args:
        prediction_time: UTC timestamp to predict from. Defaults to now.
                         The prediction is for prediction_time + FORECAST_HORIZON_HOURS.

    Returns:
        Dict with keys:
            prediction_time     (str)   — the input timestamp T
            forecast_time       (str)   — the predicted timestamp T+24h
            forecast_horizon_h  (int)   — forecast horizon in hours
            pm25_predicted      (float) — predicted PM2.5 in µg/m³
            aqi_category        (str)   — EPA AQI category label
            model_version       (str)   — MLflow model version used
            feature_warnings    (list)  — list of any missing feature warnings
    """
    if prediction_time is None:
        prediction_time = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )

    forecast_time = prediction_time + timedelta(hours=settings.FORECAST_HORIZON_HOURS)

    logger.info("=" * 60)
    logger.info("PM2.5 FORECAST")
    logger.info("=" * 60)
    logger.info("City:             %s", settings.CITY_NAME)
    logger.info("Prediction time:  %s (features known at this time)", prediction_time)
    logger.info("Forecast time:    %s (what we're predicting)", forecast_time)
    logger.info("Horizon:          %dh ahead", settings.FORECAST_HORIZON_HOURS)

    # ── Load model ────────────────────────────────────────────────────────────
    model, feature_cols = load_champion_model()

    # ── Fetch data ────────────────────────────────────────────────────────────
    logger.info("\nFetching live data sources...")

    pm25_history = load_recent_pm25(n_hours=MIN_HISTORY_HOURS)

    # Fetch recent weather (last 7 days covers all lag/rolling windows)
    df_weather = fetch_openmeteo_data(lookback_days=7)
    df_owm     = fetch_owm_air_pollution_history(lookback_days=7)

    # ── Build feature row ─────────────────────────────────────────────────────
    logger.info("\nBuilding feature vector for T=%s...", prediction_time)
    df_row = build_feature_row(prediction_time, pm25_history, df_weather, df_owm)

    # ── Align features with training feature order ────────────────────────────
    feature_warnings = []

    if feature_cols:
        # Add any missing columns as NaN
        for col in feature_cols:
            if col not in df_row.columns:
                df_row[col] = np.nan
                feature_warnings.append(f"Missing feature '{col}' — set to NaN")
                logger.warning("Missing feature '%s' at inference time — set to NaN", col)

        # Drop columns not in training features (e.g. timestamp)
        extra_cols = [c for c in df_row.columns if c not in feature_cols]
        if extra_cols:
            df_row = df_row.drop(columns=extra_cols)

        # Reorder to match training feature order exactly
        df_row = df_row[feature_cols]
    else:
        # Fall back: drop non-feature cols
        drop = [c for c in NON_FEATURE_COLS + ALWAYS_DROP_COLS if c in df_row.columns]
        df_row = df_row.drop(columns=drop, errors="ignore")

    # ── Predict ───────────────────────────────────────────────────────────────
    prediction_array = model.predict(df_row)
    pm25_predicted   = round(float(prediction_array[0]), 2)
    aqi_category     = pm25_to_aqi_category(pm25_predicted)

    # ── Log and return ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FORECAST RESULT")
    logger.info("=" * 60)
    logger.info("Predicted PM2.5:  %.2f µg/m³", pm25_predicted)
    logger.info("AQI Category:     %s", aqi_category)
    logger.info("Forecast time:    %s", forecast_time.strftime("%Y-%m-%d %H:%M UTC"))
    if feature_warnings:
        logger.warning("Feature warnings: %s", feature_warnings)

    result = {
        "prediction_time":    prediction_time.isoformat(),
        "forecast_time":      forecast_time.isoformat(),
        "forecast_horizon_h": settings.FORECAST_HORIZON_HOURS,
        "pm25_predicted":     pm25_predicted,
        "aqi_category":       aqi_category,
        "city":               settings.CITY_NAME,
        "model_version":      "champion",
        "feature_warnings":   feature_warnings,
    }

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Predict PM2.5 concentration 24h ahead for Yangon"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help=(
            "UTC timestamp to predict from (ISO format). "
            "Defaults to current hour. "
            "Example: '2026-03-22 14:00:00+00:00'"
        ),
    )
    args = parser.parse_args()

    ts = None
    if args.timestamp:
        ts = datetime.fromisoformat(args.timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

    result = run_prediction(prediction_time=ts)

    print("\n" + "=" * 50)
    print(f"  City:        {result['city']}")
    print(f"  Predict at:  {result['prediction_time']}")
    print(f"  Forecast:    {result['forecast_time']}")
    print(f"  PM2.5:       {result['pm25_predicted']} µg/m³")
    print(f"  AQI:         {result['aqi_category']}")
    if result["feature_warnings"]:
        print(f"  Warnings:    {len(result['feature_warnings'])} missing features")
    print("=" * 50)