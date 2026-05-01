from __future__ import annotations

"""
WAQI historical PM2.5 ingestion — converts daily AQI CSV to hourly µg/m³.

WHAT THIS MODULE DOES:
1. Reads the WAQI historical CSV (date, pm25 AQI columns)
2. Converts PM2.5 AQI index → µg/m³ using EPA piecewise formula
3. Disaggregates each daily value into 24 hourly rows using Yangon's
   typical diurnal PM2.5 pattern
4. Returns a DataFrame with timestamp and pm25 columns matching the
   format of openaq.py — ready to merge in ingest.py

WHY DAILY → HOURLY DISAGGREGATION:
The model was built on hourly data and all lag/rolling features assume
hourly resolution. Feeding it daily data directly would break the feature
engineering. Disaggregation using a diurnal pattern is standard practice
in air quality modelling when high-frequency data is unavailable.

WHY THIS SPECIFIC DIURNAL PATTERN (YANGON):
Yangon PM2.5 follows Southeast Asian urban combustion patterns:
  - Pre-dawn peak (5–6am): open burning, charcoal cooking fires
  - Morning rush peak (7–9am): motorcycle and bus traffic
  - Midday trough (11am–2pm): thermal mixing disperses pollutants
  - Evening rush peak (7–9pm): return traffic + cooking fires
  - Late night decline (10pm+): activity winds down

These weights are derived from published diurnal patterns for Southeast
Asian cities with similar combustion profiles (Bangkok, Ho Chi Minh City).
They are approximate — ground truth hourly data will always be preferred
where available (hence OpenAQ readings overwrite WAQI for overlapping hours).

WHY AQI → µg/m³ CONVERSION IS NECESSARY:
WAQI exports values as US EPA AQI index numbers (0–500 scale), not raw
concentrations. OpenAQ data is in µg/m³. Merging them without conversion
would cause a ~3–5× scale discontinuity at the boundary, severely biasing
the model. After conversion, both sources should read 25–50 µg/m³ for
Yangon's typical moderate pollution levels.

EPA PM2.5 AQI BREAKPOINTS:
  AQI  0–50   → µg/m³  0.0–12.0
  AQI  51–100 → µg/m³ 12.1–35.4
  AQI 101–150 → µg/m³ 35.5–55.4
  AQI 151–200 → µg/m³ 55.5–150.4
  AQI 201–300 → µg/m³ 150.5–250.4
  AQI 301+    → µg/m³ 250.5+
"""

import logging
from pathlib import Path

import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── EPA AQI → µg/m³ breakpoint table ─────────────────────────────────────────
# Each tuple: (aqi_low, aqi_high, conc_low, conc_high)
_AQI_BREAKPOINTS = [
    (0,   50,  0.0,   12.0),
    (51,  100, 12.1,  35.4),
    (101, 150, 35.5,  55.4),
    (151, 200, 55.5,  150.4),
    (201, 300, 150.5, 250.4),
    (301, 500, 250.5, 500.0),
]

# ── Yangon diurnal PM2.5 pattern ──────────────────────────────────────────────
# 24 weights (one per hour 0–23) that sum to 24.0 so the daily mean is preserved.
# Based on Southeast Asian urban combustion profiles.
_DIURNAL_WEIGHTS = np.array([
    0.75,  # 00:00 — low overnight activity
    0.70,  # 01:00
    0.68,  # 02:00
    0.70,  # 03:00
    0.78,  # 04:00 — early charcoal cooking
    0.92,  # 05:00 — pre-dawn burning peak starts
    1.05,  # 06:00 — burning + early commuters
    1.20,  # 07:00 — morning rush begins
    1.35,  # 08:00 — morning rush peak
    1.25,  # 09:00 — rush tapering
    1.05,  # 10:00
    0.92,  # 11:00 — thermal mixing starts
    0.85,  # 12:00 — midday trough
    0.82,  # 13:00
    0.85,  # 14:00
    0.92,  # 15:00 — afternoon secondary rise
    1.02,  # 16:00
    1.15,  # 17:00 — evening rush begins
    1.28,  # 18:00 — evening rush peak
    1.32,  # 19:00 — evening rush + cooking fires peak
    1.20,  # 20:00
    1.05,  # 21:00 — activity declining
    0.90,  # 22:00
    0.78,  # 23:00
], dtype=float)

# Normalise so weights sum exactly to 24.0 — preserves daily mean
_DIURNAL_WEIGHTS = _DIURNAL_WEIGHTS / _DIURNAL_WEIGHTS.sum() * 24.0


def aqi_to_ugm3(aqi: float) -> float | None:
    """
    Convert a PM2.5 AQI index value to µg/m³ using the EPA piecewise formula.

    Formula per EPA:
        C = (C_high - C_low) / (I_high - I_low) × (I - I_low) + C_low
    where I is the AQI value and C is the concentration.

    Args:
        aqi: PM2.5 AQI value (0–500 scale).

    Returns:
        PM2.5 concentration in µg/m³, rounded to 2 decimal places.
        None if AQI is NaN or out of range.
    """
    # Strip whitespace strings — AQICN CSVs use ' ' for missing values
    if isinstance(aqi, str):
        aqi = aqi.strip()
        if aqi == "":
            return None

    if pd.isna(aqi):
        return None

    try:
        aqi = float(aqi)
    except (ValueError, TypeError):
        return None

    for i_low, i_high, c_low, c_high in _AQI_BREAKPOINTS:
        if i_low <= aqi <= i_high:
            conc = (c_high - c_low) / (i_high - i_low) * (aqi - i_low) + c_low
            return round(conc, 2)

    logger.warning("AQI value %.1f is out of all breakpoint ranges — returning None", aqi)
    return None


def disaggregate_daily_to_hourly(
    date: pd.Timestamp,
    daily_pm25_ugm3: float,
) -> pd.DataFrame:
    """
    Expand one daily PM2.5 value into 24 hourly rows using the diurnal pattern.

    The daily value is treated as the 24-hour mean. Each hourly value is:
        hourly_pm25 = daily_mean × diurnal_weight[hour]

    Since weights sum to 24 and there are 24 hours, the mean is preserved.

    Args:
        date:             The date of the daily reading (timezone-naive).
        daily_pm25_ugm3:  Daily mean PM2.5 in µg/m³.

    Returns:
        DataFrame with 24 rows, columns: timestamp (UTC), pm25.
    """
    rows = []
    for hour in range(24):
        ts = pd.Timestamp(date.year, date.month, date.day, hour, 0, 0, tzinfo=datetime.timezone.utc)
        hourly_value = round(daily_pm25_ugm3 * _DIURNAL_WEIGHTS[hour] / 24.0 * 24, 3)
        rows.append({"timestamp": ts, "pm25": hourly_value})
    return pd.DataFrame(rows)


def fetch_waqi_historical(
    csv_path: Path,
    date_col: str = "date",
    pm25_col: str = "pm25",
) -> pd.DataFrame:
    """
    Load the WAQI historical CSV and return hourly PM2.5 in µg/m³.

    Handles:
    - AQICN default export format (M/D/YYYY or YYYY-MM-DD date strings)
    - AQI → µg/m³ conversion
    - Daily → hourly disaggregation using Yangon diurnal pattern
    - Deduplication if any dates appear more than once

    Args:
        csv_path: Path to the WAQI CSV file.
        date_col: Name of the date column (default "date").
        pm25_col: Name of the PM2.5 AQI column (default "pm25").

    Returns:
        DataFrame with columns: timestamp (UTC, hourly), pm25 (µg/m³).
        Empty DataFrame if file not found or no valid data.
    """
    if not csv_path.exists():
        logger.error("WAQI CSV not found at %s", csv_path)
        return pd.DataFrame()

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(csv_path, na_values=["", " ", "N/A", "n/a", "-", "--"])
    except Exception as e:
        logger.error("Failed to read WAQI CSV: %s", e)
        return pd.DataFrame()

    logger.info(
        "WAQI CSV loaded: %d rows, columns: %s",
        len(df_raw), list(df_raw.columns),
    )

    # ── Normalise column names (lowercase + strip whitespace) ─────────────────
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]
    date_col = date_col.lower()
    pm25_col = pm25_col.lower()

    if date_col not in df_raw.columns:
        logger.error(
            "Date column '%s' not found. Available columns: %s",
            date_col, list(df_raw.columns),
        )
        return pd.DataFrame()

    if pm25_col not in df_raw.columns:
        logger.error(
            "PM2.5 column '%s' not found. Available columns: %s",
            pm25_col, list(df_raw.columns),
        )
        return pd.DataFrame()

    # ── Parse dates — handle both M/D/YYYY and YYYY-MM-DD ─────────────────────
    df_raw[date_col] = pd.to_datetime(df_raw[date_col], infer_datetime_format=True)

    # ── Drop rows with missing PM2.5 ──────────────────────────────────────────
    before = len(df_raw)
    df_raw = df_raw.dropna(subset=[pm25_col])
    if len(df_raw) < before:
        logger.warning(
            "Dropped %d rows with missing PM2.5 values.", before - len(df_raw)
        )

    # ── Convert AQI → µg/m³ ───────────────────────────────────────────────────
    df_raw["pm25_ugm3"] = df_raw[pm25_col].apply(aqi_to_ugm3)
    n_failed = df_raw["pm25_ugm3"].isna().sum()
    if n_failed > 0:
        logger.warning(
            "%d rows had AQI values that could not be converted — dropped.", n_failed
        )
        df_raw = df_raw.dropna(subset=["pm25_ugm3"])

    logger.info(
        "AQI → µg/m³ conversion complete. "
        "PM2.5 range: %.1f – %.1f µg/m³ (mean=%.1f)",
        df_raw["pm25_ugm3"].min(),
        df_raw["pm25_ugm3"].max(),
        df_raw["pm25_ugm3"].mean(),
    )

    # ── Deduplicate by date (keep mean if multiple entries per day) ────────────
    df_raw = (
        df_raw.groupby(date_col)["pm25_ugm3"]
        .mean()
        .reset_index()
        .rename(columns={date_col: "date", "pm25_ugm3": "pm25_ugm3"})
    )
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    # ── Disaggregate to hourly ─────────────────────────────────────────────────
    logger.info(
        "Disaggregating %d daily rows → %d hourly rows using Yangon diurnal pattern...",
        len(df_raw), len(df_raw) * 24,
    )

    hourly_frames = []
    for _, row in df_raw.iterrows():
        hourly_frames.append(
            disaggregate_daily_to_hourly(row["date"], row["pm25_ugm3"])
        )

    df_hourly = pd.concat(hourly_frames, ignore_index=True)
    df_hourly = df_hourly.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "WAQI historical: %d hourly records (%s → %s)",
        len(df_hourly),
        df_hourly["timestamp"].min().date(),
        df_hourly["timestamp"].max().date(),
    )

    return df_hourly


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from config.settings import settings

    csv_path = settings.RAW_DATA_DIR / "waqi_yangon.csv"
    df = fetch_waqi_historical(csv_path)
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"PM2.5 range: {df['pm25'].min():.1f} – {df['pm25'].max():.1f} µg/m³")
    print(df.head(24))