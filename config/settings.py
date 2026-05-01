"""
Centralized configuration for the Air Pollution Forecast project.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable project-wide settings."""

    #  Project paths 
    PROJECT_ROOT: Path = _PROJECT_ROOT
    RAW_DATA_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "raw")
    PROCESSED_DATA_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "processed")

    #  Target city 
    CITY_NAME: str = "Yangon"
    CITY_LAT: float = 16.8661     # Yangon latitude
    CITY_LON: float = 96.1951     # Yangon longitude

    #  OpenAQ 
    OPENAQ_BASE_URL: str = "https://api.openaq.org/v3"
    OPENAQ_POLLUTANTS: tuple = ("pm25", "pm10", "co", "no2", "o3")
    OPENAQ_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENAQ_API_KEY", "")
    )

    #  OpenWeatherMap 
    OWM_API_KEY: str = field(
        default_factory=lambda: os.getenv("OWM_API_KEY", "")
    )
    OWM_BASE_URL: str = "https://api.openweathermap.org/data/2.5"

    # ── Open-Meteo (free, no key required) ────────────────────────
    OPENMETEO_BASE_URL: str = "https://archive-api.open-meteo.com/v1/archive"

    #  Data ingestion 
    LOOKBACK_DAYS: int = 7          # How many days of history to fetch (daily Airflow run)
    BACKFILL_DAYS: int = 365        # One-time historical pull for initial model training
    RAW_FILENAME: str = "yangon_pollution.csv"

    #  MLflow 
    MLFLOW_TRACKING_URI: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")  
    )
    MLFLOW_EXPERIMENT_NAME: str = "air_pollution_forecast"

    #  Model 
    MODEL_NAME: str = "pm25_lightgbm"
    TEST_SIZE: float = 0.2  # 20% holdout for final time-based split

    #  Forecast horizon 
    # How many hours ahead the model predicts.
    # 24 = predict PM2.5 at T+24h using features known at T.
    # Lags shorter than this horizon are removed in features.py to
    # prevent leakage — you can't know pm25_lag_1h at T+24h prediction time.
    FORECAST_HORIZON_HOURS: int = 24

    #  Promotion threshold 
    PROMOTION_THRESHOLD: float = 0.04

    #  Cross-validation 
    CV_N_FOLDS: int = 5           # number of walk-forward folds
    CV_MIN_TRAIN_SIZE: float = 0.5  # first fold uses at least 50% of data for training

    def __post_init__(self):
        """Create data directories if they don't exist."""
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)



settings = Settings()