from __future__ import annotations

"""
Open-Meteo API client — fetches historical weather data for Yangon.

WHY OPEN-METEO INSTEAD OF OWM FOR WEATHER:
- Completely free, no API key required
- Historical archive goes back to 1940 — essential for backfilling a year
  of training data
- Hourly resolution natively (no 3-hour interpolation needed)
- OWM's free weather forecast only goes 5 days ahead — useless for history

API USED:
  https://archive-api.open-meteo.com/v1/archive

IMPORTANT — ARCHIVE DELAY:
Open-Meteo's archive typically lags by 5 days. For the daily Airflow run
(LOOKBACK_DAYS=7) we fetch t-12 days to t-5 days to ensure full coverage.
The ingest deduplication step means re-fetching overlapping days is harmless.

VARIABLES FETCHED (hourly):
  temperature_2m          → temperature   (°C, 2m above ground)
  relative_humidity_2m    → humidity      (%, 2m above ground)
  wind_speed_10m          → wind_speed    (km/h, 10m above ground)
  surface_pressure        → pressure      (hPa)

NOTE ON WIND UNITS:
Open-Meteo returns wind speed in km/h. We convert to m/s on ingest
(÷ 3.6) to match the unit convention used in OWM and the rest of the
pipeline. This prevents silent unit mismatch bugs in the model.
"""

import logging
from datetime import date, timedelta

import httpx
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# Open-Meteo archive lags by ~5 days — offset end date to avoid empty tail
_ARCHIVE_LAG_DAYS = 5

# Variable name mapping: Open-Meteo name → our pipeline name
_VARIABLE_MAP = {
    "temperature_2m":       "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m":       "wind_speed",
    "surface_pressure":     "pressure",
}


def fetch_openmeteo_data(
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """
    Fetch hourly historical weather from Open-Meteo for Yangon.

    No API key required. Data is fetched from the Open-Meteo archive
    which covers the full historical record at hourly resolution.

    DATE RANGE LOGIC:
        end_date   = today - ARCHIVE_LAG_DAYS  (last reliably available day)
        start_date = end_date - lookback_days

    For the daily Airflow run (lookback_days=7) this gives an 8-day window
    that always sits safely within the archive's coverage. The append +
    deduplicate logic in ingest.py handles any day overlap with existing data.

    Args:
        lookback_days: Days of history to fetch. Defaults to settings.LOOKBACK_DAYS.

    Returns:
        DataFrame with columns: timestamp, temperature, humidity,
        wind_speed (m/s), pressure
        Empty DataFrame on failure.
    """
    if lookback_days is None:
        lookback_days = settings.LOOKBACK_DAYS

    end_date   = date.today() - timedelta(days=_ARCHIVE_LAG_DAYS)
    start_date = end_date - timedelta(days=lookback_days)

    params = {
        "latitude":   settings.CITY_LAT,
        "longitude":  settings.CITY_LON,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "hourly":     ",".join(_VARIABLE_MAP.keys()),
        "timezone":   "UTC",
        "wind_speed_unit": "ms",   # request m/s directly — avoids manual conversion
    }

    logger.info(
        "Fetching Open-Meteo weather for %s (%s to %s)...",
        settings.CITY_NAME, start_date, end_date,
    )

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(settings.OPENMETEO_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            "Open-Meteo API error %d: %s",
            e.response.status_code, e.response.text,
        )
        return pd.DataFrame()
    except httpx.RequestError as e:
        logger.error("Network error fetching Open-Meteo data: %s", e)
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        logger.warning("Open-Meteo returned no hourly data.")
        return pd.DataFrame()

    df = pd.DataFrame({"timestamp": hourly["time"]})

    for api_name, our_name in _VARIABLE_MAP.items():
        if api_name in hourly:
            df[our_name] = hourly[api_name]
        else:
            logger.warning("Open-Meteo: expected variable '%s' not in response.", api_name)
            df[our_name] = float("nan")

    # Parse timestamps — Open-Meteo returns ISO strings e.g. "2026-03-01T00:00"
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "Open-Meteo weather: %d hourly records (%s to %s)",
        len(df), df["timestamp"].min(), df["timestamp"].max(),
    )
    return df


def fetch_openmeteo_backfill(
    backfill_days: int | None = None,
) -> pd.DataFrame:
    """
    Fetch a large historical weather window for initial model training.

    Identical to fetch_openmeteo_data() but uses settings.BACKFILL_DAYS
    (default 365) as the lookback window. Open-Meteo handles large date
    ranges in a single request — no pagination needed.

    Call this once via backfill.py, not from the daily Airflow DAG.

    Args:
        backfill_days: Days of history to fetch. Defaults to settings.BACKFILL_DAYS.

    Returns:
        DataFrame covering ~1 year of hourly weather records.
    """
    if backfill_days is None:
        backfill_days = settings.BACKFILL_DAYS

    logger.info(
        "Starting Open-Meteo backfill: %d days of history...", backfill_days
    )
    return fetch_openmeteo_data(lookback_days=backfill_days)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_openmeteo_data(lookback_days=7)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(10))