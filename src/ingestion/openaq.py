from __future__ import annotations

"""
OpenAQ API client — fetches air pollution measurements for Bangkok.

HOW OPENAQ v3 WORKS:
1. First we search for sensor locations near Bangkok using /locations
2. Then we fetch recent measurements from those locations using /locations/{id}/measurements
3. We normalize the data into a flat DataFrame with one row per timestamp

WHY we use OpenAQ:
- Free tier available (requires API key since 2024)
- Real-time PM2.5, PM10, CO, NO2, O3 readings from government sensors
- Data available for 100+ countries

COMMON PITFALLS:
- OpenAQ v3 changed the response format significantly from v2
- Some Bangkok sensors only report PM2.5, not all pollutants
- Timestamps come in UTC — we keep them in UTC for consistency
- Rate limit is generous (no official cap) but we add retries anyway
"""


import logging
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


def _get_headers() -> dict:
    """Build request headers with API key."""
    headers = {}
    key = settings.OPENAQ_API_KEY
    if key:
        headers["X-API-Key"] = key
    else:
        logger.warning(
            "OpenAQ API key not set! Requests will likely fail (401). "
            "Sign up at https://explore.openaq.org/register and add "
            "OPENAQ_API_KEY to your .env file."
        )
    return headers


def fetch_bangkok_locations(
    radius_meters: int = 25000,
    limit: int = 50,
) -> list[dict]:
    """
    Find OpenAQ sensor locations near Bangkok.

    Args:
        radius_meters: Search radius around Bangkok center (default 25km covers
                       the metro area while avoiding sensors in other provinces).
        limit: Max number of locations to return.

    Returns:
        List of location dicts with keys: id, name, coordinates, etc.
    """
    url = f"{settings.OPENAQ_BASE_URL}/locations"
    params = {
        "coordinates": f"{settings.CITY_LAT},{settings.CITY_LON}",
        "radius": radius_meters,
        "limit": limit,
    }

    try:
        with httpx.Client(timeout=30.0, headers=_get_headers()) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAQ locations API returned {e.response.status_code}: {e.response.text}")
        return []
    except httpx.RequestError as e:
        logger.error(f"Network error fetching OpenAQ locations: {e}")
        return []

    results = data.get("results", [])
    logger.info(f"Found {len(results)} OpenAQ locations near Bangkok")
    return results


def fetch_measurements_for_sensor(
    sensor_id: int,
    date_from: datetime,
    date_to: datetime,
    limit: int = 1000,
) -> list[dict]:
    """
    Fetch measurements for a specific SENSOR within a date range.

    OpenAQ v3 uses sensor-based endpoints, not location-based.
    Each location has multiple sensors (one per pollutant).

    Args:
        sensor_id: OpenAQ sensor ID.
        date_from: Start of time range (UTC).
        date_to: End of time range (UTC).
        limit: Max measurements per request.

    Returns:
        Raw list of measurement dicts from the API.
    """
    url = f"{settings.OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements"
    params = {
        "datetime_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datetime_to": date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": limit,
    }

    all_results = []
    page = 1

    with httpx.Client(timeout=30.0, headers=_get_headers()) as client:
        while True:
            params["page"] = page
            try:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"OpenAQ sensor {sensor_id} measurements error: "
                    f"{e.response.status_code}"
                )
                break
            except httpx.RequestError as e:
                logger.warning(f"Network error for sensor {sensor_id}: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)
            logger.debug(
                f"Sensor {sensor_id}: fetched page {page} "
                f"({len(results)} records)"
            )

            if len(results) < limit:
                break
            page += 1

    return all_results


def _parse_measurements(raw_measurements: list[dict]) -> pd.DataFrame:
    """
    Parse raw OpenAQ v3 measurement records into a flat DataFrame.

    The v3 sensor API returns records like:
        {
            "period": {
                "datetimeFrom": {"utc": "2026-03-10T00:00:00+00:00", "local": "..."},
                "datetimeTo": {"utc": "...", "local": "..."}
            },
            "value": 42.3,
            "_param_name": "pm25"  (injected by our fetcher)
        }

    We pivot this so each timestamp has columns: pm25, pm10, co, no2, o3.
    """
    if not raw_measurements:
        return pd.DataFrame()

    rows = []
    for m in raw_measurements:
        # Extract timestamp from nested v3 structure
        period = m.get("period", {})
        dt_from = period.get("datetimeFrom", {})
        timestamp_str = dt_from.get("utc", "") if isinstance(dt_from, dict) else ""

        # Use our injected param name, or fall back to nested parameter.name
        param_name = m.get("_param_name", "")
        if not param_name:
            param_info = m.get("parameter", {})
            param_name = param_info.get("name", "unknown") if isinstance(param_info, dict) else "unknown"

        value = m.get("value")

        if timestamp_str and value is not None:
            rows.append({
                "timestamp": timestamp_str,
                "parameter": param_name.lower(),
                "value": float(value),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Round to nearest hour for consistent merging with weather data
    df["timestamp"] = df["timestamp"].dt.floor("h")

    # Pivot: one row per timestamp, one column per pollutant
    # Use mean to handle duplicate readings within the same hour
    df_pivot = df.pivot_table(
        index="timestamp",
        columns="parameter",
        values="value",
        aggfunc="mean",
    ).reset_index()

    # Flatten column names
    df_pivot.columns.name = None

    return df_pivot


def fetch_openaq_data(lookback_days: int | None = None) -> pd.DataFrame:
    """
    Main entry point: fetch pollution data for Bangkok.

    Steps:
        1. Find sensor locations near Bangkok
        2. Fetch recent measurements from each location
        3. Parse and merge into a single DataFrame

    Args:
        lookback_days: Days of history to fetch. Defaults to settings.LOOKBACK_DAYS.

    Returns:
        DataFrame with columns: timestamp, pm25, pm10, co, no2, o3
        (some columns may be NaN if not measured by any sensor).
    """
    if lookback_days is None:
        lookback_days = settings.LOOKBACK_DAYS

    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=lookback_days)

    logger.info(
        f"Fetching OpenAQ data for Bangkok from "
        f"{date_from.date()} to {date_to.date()}"
    )

    # Step 1: Find locations
    locations = fetch_bangkok_locations()
    if not locations:
        logger.warning("No OpenAQ locations found near Bangkok")
        return pd.DataFrame()

    # Step 2: Extract sensor IDs from locations and fetch measurements
    # Each location has sensors — one per pollutant. We collect all of them.
    all_measurements = []
    locations_processed = 0
    max_locations = 10  # Limit to avoid excessive API calls

    for loc in locations:
        if locations_processed >= max_locations:
            break

        loc_name = loc.get("name", "Unknown")
        sensors = loc.get("sensors", [])

        if not sensors:
            continue

        locations_processed += 1
        logger.info(
            f"Fetching from: {loc_name} "
            f"({len(sensors)} sensors)"
        )

        for sensor in sensors:
            sensor_id = sensor.get("id")
            param_info = sensor.get("parameter", {})
            param_name = param_info.get("name", "unknown").lower()

            # Only fetch pollutants we care about
            if param_name not in settings.OPENAQ_POLLUTANTS:
                continue

            if sensor_id is None:
                continue

            measurements = fetch_measurements_for_sensor(
                sensor_id=sensor_id,
                date_from=date_from,
                date_to=date_to,
            )
            # Tag each measurement with the parameter name
            for m in measurements:
                m["_param_name"] = param_name
            all_measurements.extend(measurements)

    logger.info(f"Total raw measurements collected: {len(all_measurements)}")

    # Step 3: Parse and pivot
    df = _parse_measurements(all_measurements)

    if df.empty:
        logger.warning("No measurements parsed from OpenAQ")
        return df

    # Ensure expected columns exist (fill missing pollutants with NaN)
    for pollutant in settings.OPENAQ_POLLUTANTS:
        if pollutant not in df.columns:
            df[pollutant] = float("nan")

    # Select and order columns
    columns = ["timestamp"] + list(settings.OPENAQ_POLLUTANTS)
    df = df[[c for c in columns if c in df.columns]]

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"OpenAQ data ready: {len(df)} rows, "
        f"date range {df['timestamp'].min()} → {df['timestamp'].max()}"
    )

    return df


if __name__ == "__main__":
    # Quick test: run this file directly to see the data
    logging.basicConfig(level=logging.INFO)
    df = fetch_openaq_data(lookback_days=3)
    print(f"\nShape: {df.shape}")
    print(df.head(10))
    print(f"\nMissing values:\n{df.isnull().sum()}")