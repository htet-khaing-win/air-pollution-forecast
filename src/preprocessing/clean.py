from __future__ import annotations

"""
Data cleaning pipeline for the Air Pollution Forecast project.

WHAT THIS MODULE DOES:
1. Drops rows where PM2.5 is missing — it's the target variable, so rows
   without it are useless for both training and evaluation.
2. Forward-fills small weather gaps (≤3 consecutive hours) — weather
   changes slowly, so a short interpolation is physically reasonable.
   Larger gaps are left as NaN and flagged; imputing them would fabricate data.
3. Removes physically impossible values — sensors malfunction and report
   negative readings or absurdly high spikes. We clip at EPA-defined bounds.
4. Logs outlier statistics — so we can monitor data quality over time
   without silently swallowing bad data.

WHY NOT JUST DROP ALL NaN ROWS:
- Dropping any row with any NaN would eliminate most of our data (Phase 1
  showed many rows have NaN for CO/NO2/O3 because most stations don't
  measure them).
- We only *need* PM2.5 (target) + weather features to train. The other
  pollutants are supplementary.
- Feature engineering (features.py) will handle remaining NaNs via lag
  windows — lags computed over NaN-containing rows are naturally excluded
  by pandas rolling/shift operations.
"""

import logging
from pathlib import Path

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Physical bounds for sensor sanity checks ──────────────────────────────────
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "pm25":        (0.0,    1000.0),
    "pm10":        (0.0,    2000.0),
    "co":          (0.0,   50000.0),
    "no2":         (0.0,    2000.0),
    "o3":          (0.0,    1000.0),
    "temperature": (-20.0,    60.0),
    "humidity":    (0.0,     100.0),
    "wind_speed":  (0.0,     100.0),
    "pressure":    (850.0,  1100.0),
}

# ── Forward-fill cap: only fill gaps up to this many consecutive hours ─────────
MAX_FFILL_HOURS: int = 3


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where PM2.5 (the target variable) is missing.

    We log how many rows are dropped so we can track data quality over time.
    A high drop rate suggests an API or sensor problem upstream.

    Args:
        df: Raw merged DataFrame from ingest.py.

    Returns:
        DataFrame with no NaN in the pm25 column.
    """
    before = len(df)

    if "pm25" not in df.columns:
        logger.error(
            "Column 'pm25' not found in DataFrame. "
            "Available columns: %s", list(df.columns)
        )
        return df

    df = df.dropna(subset=["pm25"]).copy()
    dropped = before - len(df)

    if dropped > 0:
        drop_pct = dropped / before * 100
        logger.warning(
            "Dropped %d rows (%.1f%%) with missing PM2.5 target. "
            "%d rows remain.",
            dropped, drop_pct, len(df),
        )
    else:
        logger.info("No rows dropped — all rows have PM2.5 values.")

    return df


def remove_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace physically impossible sensor readings with NaN.

    WHY NaN INSTEAD OF DROPPING:
    Setting to NaN instead of dropping the whole row preserves the timestamp
    and any valid readings in other columns. For example, a bad PM10 reading
    shouldn't cause us to lose the valid PM2.5 and weather data for that hour.

    The forward-fill step that follows will attempt to recover these NaN cells
    for weather columns.

    Returns:
        DataFrame with out-of-range values replaced by NaN.
    """
    total_replaced = 0

    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue

        mask = (df[col] < lo) | (df[col] > hi)
        count = mask.sum()

        if count > 0:
            logger.warning(
                "Column '%s': %d impossible value(s) outside [%s, %s] → set to NaN. "
                "Sample values: %s",
                col, count, lo, hi,
                df.loc[mask, col].head(5).tolist(),
            )
            df.loc[mask, col] = float("nan")
            total_replaced += count

    if total_replaced == 0:
        logger.info("No physically impossible values found.")
    else:
        logger.info("Total impossible values replaced with NaN: %d", total_replaced)

    return df


def forward_fill_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill small gaps (≤ MAX_FFILL_HOURS) in weather feature columns.

    WHY ONLY WEATHER COLUMNS:
    - Weather changes slowly (temperature doesn't jump 10°C in an hour),
      so short forward-fills are physically valid.
    - Pollution values (pm25, pm10, etc.) can spike suddenly — forward-filling
      them would fabricate data and bias the model. We leave those as NaN.

    WHY LIMIT TO 3 HOURS:
    Filling beyond 3 hours starts to diverge meaningfully from reality,
    especially for wind and humidity during weather transitions.

    HOW IT WORKS:
    pandas ffill(limit=N) fills NaN cells only if they follow a valid value
    with at most N consecutive NaN cells before them. A gap of 4+ NaN hours
    is left untouched.

    Args:
        df: DataFrame sorted by timestamp.

    Returns:
        DataFrame with small weather gaps filled.
    """
    weather_cols = [
        c for c in ["temperature", "humidity", "wind_speed", "pressure"]
        if c in df.columns
    ]

    if not weather_cols:
        logger.warning("No weather columns found to forward-fill.")
        return df

    # Ensure we're sorted by time before filling
    df = df.sort_values("timestamp").reset_index(drop=True)

    before_nulls = df[weather_cols].isnull().sum()
    df[weather_cols] = df[weather_cols].ffill(limit=MAX_FFILL_HOURS)
    after_nulls = df[weather_cols].isnull().sum()

    filled = before_nulls - after_nulls
    filled_total = filled.sum()

    if filled_total > 0:
        logger.info(
            "Forward-filled %d weather cell(s) across columns: %s",
            filled_total,
            filled[filled > 0].to_dict(),
        )
    else:
        logger.info("No weather gaps required forward-filling.")

    # Log remaining large gaps that we deliberately did NOT fill
    remaining = after_nulls[after_nulls > 0]
    if not remaining.empty:
        logger.warning(
            "Weather columns still have NaN after forward-fill "
            "(gaps > %d hours or start-of-series): %s",
            MAX_FFILL_HOURS,
            remaining.to_dict(),
        )

    return df


def log_outlier_statistics(df: pd.DataFrame) -> None:
    """
    Log descriptive statistics for key columns to aid monitoring.

    This does NOT remove outliers — it surfaces them so you can decide
    whether to act. Log these before and after cleaning to track the effect.

    What we report:
    - Count of values beyond 1.5× IQR (statistical outliers, not errors)
    - Min / max / mean for all numeric columns
    - NaN counts per column
    """
    logger.info("── Outlier & null statistics ──────────────────────────────")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            logger.warning("  %s: all values are NaN", col)
            continue

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outlier_count = ((series < lower_fence) | (series > upper_fence)).sum()
        null_count = df[col].isnull().sum()

        logger.info(
            "  %-15s  min=%-8.2f  max=%-8.2f  mean=%-8.2f  "
            "IQR-outliers=%-4d  nulls=%d",
            col,
            series.min(), series.max(), series.mean(),
            outlier_count, null_count,
        )

    logger.info("───────────────────────────────────────────────────────────")


def run_cleaning(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Full cleaning pipeline: load → validate → clean → save.

    Steps (in order):
        1. Load raw CSV from ingest.py output
        2. Log pre-cleaning outlier statistics
        3. Remove physically impossible values
        4. Drop rows with missing PM2.5 (target)
        5. Forward-fill small weather gaps
        6. Log post-cleaning statistics
        7. Save to processed data directory

    Args:
        input_path:  Path to raw CSV. Defaults to settings.RAW_DATA_DIR / RAW_FILENAME.
        output_path: Path to save cleaned CSV. Defaults to PROCESSED_DATA_DIR / "cleaned.csv".

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    if input_path is None:
        input_path = settings.RAW_DATA_DIR / settings.RAW_FILENAME
    if output_path is None:
        output_path = settings.PROCESSED_DATA_DIR / "cleaned.csv"

    logger.info("=" * 60)
    logger.info("Starting data cleaning pipeline")
    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {input_path}. "
            "Run ingest.py first to fetch data."
        )

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), input_path)

    # ── Pre-clean statistics ───────────────────────────────────────────────────
    logger.info("\n[PRE-CLEANING STATISTICS]")
    log_outlier_statistics(df)

    # ── Step 1: Remove impossible sensor values ────────────────────────────────
    logger.info("\n[1/3] Removing physically impossible values...")
    df = remove_impossible_values(df)

    # ── Step 2: Drop rows with missing target ─────────────────────────────────
    logger.info("\n[2/3] Dropping rows with missing PM2.5 target...")
    df = drop_missing_target(df)

    if df.empty:
        logger.error(
            "DataFrame is empty after dropping missing PM2.5 rows. "
            "Check data ingestion pipeline."
        )
        return df

    # ── Step 3: Forward-fill small weather gaps ────────────────────────────────
    logger.info("\n[3/3] Forward-filling small weather gaps (≤%d hours)...", MAX_FFILL_HOURS)
    df = forward_fill_weather(df)

    # ── Post-clean statistics ──────────────────────────────────────────────────
    logger.info("\n[POST-CLEANING STATISTICS]")
    log_outlier_statistics(df)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("\nCleaned data saved: %d rows → %s", len(df), output_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    df = run_cleaning()
    print(f"\nShape: {df.shape}")
    print(df.head())