from __future__ import annotations

"""
OpenWeatherMap API client — air pollution history only.

WHAT THIS MODULE DOES:
Fetches OWM's model-driven air pollution estimates (PM2.5, PM10, CO, NO2, O3)
for Yangon using the /air_pollution/history endpoint.

WHY WE KEEP THIS ALONGSIDE OPENAQ:
1. Resilience  — OWM is model-driven and always returns a value even when
                 physical government sensors go offline for maintenance.
2. Signal      — The *difference* between OWM's model estimate and the OpenAQ
                 ground-truth reading is itself a useful XGBoost feature. A large
                 gap suggests unusual local conditions the global model missed.

WHY WEATHER (TEMP/HUMIDITY/WIND) WAS REMOVED FROM THIS MODULE:
OWM's /forecast endpoint only looks ahead 5 days. For historical weather
data we now use Open-Meteo (openmeteo.py), which provides free hourly
records going back to 1940 — far better for training data.

REQUIRES:
- OWM paid plan for /air_pollution/history.
- OWM_API_KEY set in .env
"""

import logging
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


def _check_api_key() -> bool:
    """Verify the OWM API key is set."""
    if not settings.OWM_API_KEY:
        logger.error(
            "OpenWeatherMap API key not set! "
            "Please add OWM_API_KEY to your .env file."
        )
        return False
    return True


def fetch_owm_air_pollution_history(
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """
    Fetch historical air pollution data from OWM's Air Pollution API.

    Returns hourly model-driven estimates for PM2.5, PM10, CO, NO2, O3.
    All columns are prefixed with 'owm_' to distinguish them from OpenAQ
    ground-truth measurements in the merged DataFrame.

    COLUMN NAMING CONVENTION:
        owm_pm25, owm_pm10, owm_co, owm_no2, owm_o3

    This lets features.py compute a delta feature later:
        pm25_delta = pm25 (OpenAQ sensor) - owm_pm25 (OWM model)

    REQUIRES OWM PAID PLAN:
    Returns an empty DataFrame (with a warning) on the free tier.
    The pipeline is designed to continue gracefully without this data.

    Args:
        lookback_days: Days of history to fetch. Defaults to settings.LOOKBACK_DAYS.

    Returns:
        DataFrame with columns: timestamp, owm_pm25, owm_pm10, owm_co, owm_no2, owm_o3
        Empty DataFrame if API call fails or key is missing.
    """
    if not _check_api_key():
        return pd.DataFrame()

    if lookback_days is None:
        lookback_days = settings.LOOKBACK_DAYS

    end   = int(datetime.now(timezone.utc).timestamp())
    start = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())

    url = f"{settings.OWM_BASE_URL}/air_pollution/history"
    params = {
        "lat":   settings.CITY_LAT,
        "lon":   settings.CITY_LON,
        "start": start,
        "end":   end,
        "appid": settings.OWM_API_KEY,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.warning(
                "OWM air pollution history requires a paid plan. "
                "Skipping OWM supplement — pipeline will continue with OpenAQ only."
            )
        else:
            logger.error(
                "OWM air pollution history error: %d",
                e.response.status_code,
            )
        return pd.DataFrame()
    except httpx.RequestError as e:
        logger.error("Network error fetching OWM air pollution history: %s", e)
        return pd.DataFrame()

    rows = []
    for entry in data.get("list", []):
        components = entry.get("components", {})
        rows.append({
            "timestamp": datetime.fromtimestamp(entry["dt"], tz=timezone.utc),
            "owm_pm25":  components.get("pm2_5"),
            "owm_pm10":  components.get("pm10"),
            "owm_co":    components.get("co"),
            "owm_no2":   components.get("no2"),
            "owm_o3":    components.get("o3"),
        })

    if not rows:
        logger.warning("OWM air pollution history returned no records.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")

    # Average readings that fall in the same hour
    df = (
        df.groupby("timestamp")
          .mean(numeric_only=True)
          .reset_index()
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    logger.info(
        "OWM air pollution history: %d hourly records (%s to %s)",
        len(df), df["timestamp"].min(), df["timestamp"].max(),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_owm_air_pollution_history(lookback_days=7)
    print(f"\nShape: {df.shape}")
    print(df.head(10))