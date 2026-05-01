from __future__ import annotations

"""
Feature engineering pipeline for the Air Pollution Forecast project.

WHAT THIS MODULE BUILDS:
1. Lag features       — PM2.5 values from t-1, t-3, t-6, t-12, t-24, t-48, t-72 hours ago.
                        All are safe at inference time — they reference past
                        observations known at prediction time T.
                        These let the model learn temporal autocorrelation.
2. Rolling statistics — mean, std, max of PM2.5 over 6h, 12h, 24h windows.
                        Capture recent trend and volatility.
3. Time features      — hour, day of week, month, is_weekend, plus cyclical
                        (sin/cos) encoding for hour and month.
4. Interaction terms  — humidity × temperature, which together determine
                        how well the atmosphere disperses PM2.5 particles.
5. AQI label          — categorical air quality label derived from PM2.5
                        using EPA breakpoints. Useful for classification tasks
                        and human-readable evaluation.

WHY LAGS:
PM2.5 at time T is heavily influenced by PM2.5 at T-1, T-6, T-24.
Pollution has strong autocorrelation — high-smog days tend to cluster.
Without lags, the model only sees current conditions and misses this memory.

WHY ROLLING STATS:
- Rolling mean: captures the recent "baseline" pollution level.
- Rolling std:  captures volatility — a sudden spike looks different from
                a gradual build-up.
- Rolling max:  captures recent peaks — useful for AQI threshold prediction.

WHY CYCLICAL ENCODING FOR HOUR AND MONTH:
One-hot encoding hour_of_day (0–23) creates 24 sparse binary columns.
Label encoding treats hour 23 as "far from" hour 0, but they're adjacent.
Sin/cos encoding maps the cycle onto a unit circle so hour 23 and hour 0
are close together in feature space — which is true physically (midnight
rush doesn't reset just because the hour ticks over).
  sin_hour = sin(2π × hour / 24)
  cos_hour = cos(2π × hour / 24)

WHY HUMIDITY × TEMPERATURE INTERACTION:
Neither feature alone fully explains dispersion. High humidity at low
temperature behaves differently from high humidity at high temperature
(Bangkok's hot-humid afternoons trap pollutants differently from cool foggy
mornings). The product of the two gives the model a direct signal for this
combined effect without needing to learn it implicitly.

"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Lag offsets (hours) ────────────────────────────────────────────────────────
# All lags are safe at inference time — they reference past observations
# already recorded in the master CSV. pm25_lag_1h at T means "pm25 one
# hour ago", which is known at prediction time T regardless of how far
# ahead we are forecasting.
LAG_HOURS: list[int] = [1, 3, 6, 12, 24, 48, 72]

# ── Rolling window sizes (hours) ──────────────────────────────────────────────
ROLLING_WINDOWS: list[int] = [6, 12, 24]

# ── EPA PM2.5 breakpoints (upper bound of each category, µg/m³) ───────────────
# (lower_bound, upper_bound, label)
AQI_BREAKPOINTS: list[tuple[float, float, str]] = [
    (0.0,    12.0,  "Good"),
    (12.1,   35.4,  "Moderate"),
    (35.5,   55.4,  "Unhealthy for Sensitive Groups"),
    (55.5,  150.4,  "Unhealthy"),
    (150.5, 250.4,  "Very Unhealthy"),
    (250.5, 9999.0, "Hazardous"),
]


# ── Individual feature builders ───────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged PM2.5 values as features.

    For each lag L in LAG_HOURS, creates column `pm25_lag_{L}h` containing
    the PM2.5 value from L rows earlier (assuming hourly data, 1 row = 1 hour).

    WHY THIS REQUIRES SORTED DATA:
    pandas shift() works positionally — row N gets the value from row N-L.
    If the data isn't sorted by time, a "1-hour lag" is meaningless.
    We enforce sorting at the start of run_feature_engineering().

    NOTE ON NaN:
    The first L rows will have NaN for the L-hour lag (no history yet).
    These rows are dropped at the end of the pipeline because any model
    training requires complete feature vectors.

    Args:
        df: DataFrame sorted by timestamp with a 'pm25' column.

    Returns:
        DataFrame with added lag columns.
    """
    for lag in LAG_HOURS:
        col_name = f"pm25_lag_{lag}h"
        df[col_name] = df["pm25"].shift(lag)
        logger.debug("Added lag feature: %s", col_name)

    logger.info("Added %d lag features: %s", len(LAG_HOURS), [f"pm25_lag_{l}h" for l in LAG_HOURS])
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window statistics for PM2.5.

    For each window W in ROLLING_WINDOWS, creates:
        pm25_roll_{W}h_mean  — average PM2.5 over the past W hours
        pm25_roll_{W}h_std   — std deviation over the past W hours
        pm25_roll_{W}h_max   — maximum PM2.5 over the past W hours

    WHY min_periods=1:
    At the start of the series, we have fewer than W rows of history.
    min_periods=1 computes stats over whatever history is available
    rather than returning NaN for the first W-1 rows. This is a
    deliberate trade-off: slightly less accurate early stats vs. keeping
    more training rows.

    WHY closed='left':
    We use closed='left' so the window does NOT include the current row's
    value — this prevents data leakage (the model can't know the current
    PM2.5 at prediction time; it only knows past values).

    Args:
        df: DataFrame sorted by timestamp with a 'pm25' column.

    Returns:
        DataFrame with added rolling statistic columns.
    """
    for window in ROLLING_WINDOWS:
        roll = df["pm25"].rolling(window=window, min_periods=1, closed="left")
        df[f"pm25_roll_{window}h_mean"] = roll.mean()
        df[f"pm25_roll_{window}h_std"]  = roll.std()
        df[f"pm25_roll_{window}h_max"]  = roll.max()
        logger.debug("Added rolling features for window=%dh", window)

    new_cols = [
        f"pm25_roll_{w}h_{stat}"
        for w in ROLLING_WINDOWS
        for stat in ["mean", "std", "max"]
    ]
    logger.info("Added %d rolling features.", len(new_cols))
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features from the timestamp column.

    Features added:
        hour_of_day   — 0–23 integer
        day_of_week   — 0=Monday, 6=Sunday
        month         — 1–12 integer
        is_weekend    — 1 if Saturday or Sunday, else 0
        sin_hour      — cyclical sine encoding of hour
        cos_hour      — cyclical cosine encoding of hour
        sin_month     — cyclical sine encoding of month
        cos_month     — cyclical cosine encoding of month

    WHY KEEP BOTH RAW AND CYCLICAL:
    Raw integers (hour_of_day) are useful for tree-based models like XGBoost
    which can create threshold splits (e.g. "hour > 18"). Cyclical encodings
    help if we add distance-based models later (linear regression, KNN).
    Having both costs little and increases model flexibility.

    Args:
        df: DataFrame with a timezone-aware 'timestamp' column.

    Returns:
        DataFrame with added time feature columns.
    """
    ts = df["timestamp"]

    df["hour_of_day"]  = ts.dt.hour
    df["day_of_week"]  = ts.dt.dayofweek
    df["month"]        = ts.dt.month
    df["is_weekend"]   = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding
    df["sin_hour"]  = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["cos_hour"]  = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    logger.info(
        "Added time features: hour_of_day, day_of_week, month, is_weekend, "
        "sin_hour, cos_hour, sin_month, cos_month"
    )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features between weather variables.

    Features added:
        humidity_x_temp  — humidity × temperature product

    PHYSICAL RATIONALE:
    Bangkok's pollution is worst on hot, humid, still days. Humidity alone
    or temperature alone doesn't capture this — their interaction does.
    High humidity + high temperature → strong thermal inversion risk.
    High humidity + low temperature → fog/mist that traps fine particles.
    The product term lets the model distinguish these two regimes.

    Only created if both 'humidity' and 'temperature' columns are present.
    Silently skipped if either is missing (e.g. OWM API failure day).

    Args:
        df: DataFrame with optional 'humidity' and 'temperature' columns.

    Returns:
        DataFrame with added interaction columns.
    """
    if "humidity" in df.columns and "temperature" in df.columns:
        df["humidity_x_temp"] = df["humidity"] * df["temperature"]
        logger.info("Added interaction feature: humidity_x_temp")
    else:
        missing = [c for c in ["humidity", "temperature"] if c not in df.columns]
        logger.warning(
            "Skipping humidity_x_temp interaction — missing columns: %s", missing
        )

    return df



def add_owm_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add delta feature: difference between OpenAQ sensor and OWM model estimate.

    WHY THIS IS VALUABLE:
    When the ground-truth sensor reads much higher than OWM's model expects,
    something unusual is happening locally — a factory fire, a festival with
    firecrackers, an unexpected weather inversion. This gap is a strong signal
    that XGBoost can exploit to predict future spikes.

    A near-zero delta = conditions match the global model = normal day.
    A large positive delta = local event the model didn't anticipate.
    A large negative delta = sensor may be under-reading or OWM overestimating.

    Only created if both 'pm25' and 'owm_pm25' are present and not fully NaN.
    Silently skipped otherwise.

    Args:
        df: DataFrame with optional 'pm25' and 'owm_pm25' columns.

    Returns:
        DataFrame with added 'pm25_owm_delta' column.
    """
    if "pm25" in df.columns and "owm_pm25" in df.columns:
        if df["pm25"].notna().any() and df["owm_pm25"].notna().any():
            df["pm25_owm_delta"] = df["pm25"] - df["owm_pm25"]
            logger.info(
                "Added delta feature: pm25_owm_delta  "
                "(mean=%.2f, std=%.2f)",
                df["pm25_owm_delta"].mean(),
                df["pm25_owm_delta"].std(),
            )
        else:
            logger.warning(
                "Skipping pm25_owm_delta — one or both columns are fully NaN."
            )
    else:
        missing = [c for c in ["pm25", "owm_pm25"] if c not in df.columns]
        logger.warning(
            "Skipping pm25_owm_delta — missing columns: %s", missing
        )

    return df

def add_aqi_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a categorical AQI label from PM2.5 concentration.

    Uses EPA 24-hour PM2.5 AQI breakpoints. The label is derived from the
    *instantaneous* hourly PM2.5, not a true 24-hour rolling average — this
    is a simplification acceptable for a real-time forecast model.

    Categories (stored in 'aqi_category' column):
        Good · Moderate · Unhealthy for Sensitive Groups ·
        Unhealthy · Very Unhealthy · Hazardous

    Also adds 'aqi_numeric' (0–5 integer) for models that need a numeric target.

    Args:
        df: DataFrame with a 'pm25' column.

    Returns:
        DataFrame with added 'aqi_category' and 'aqi_numeric' columns.
    """
    if "pm25" not in df.columns:
        logger.warning("Cannot add AQI label — 'pm25' column not found.")
        return df

    def _classify(pm25: float) -> tuple[str, int]:
        if pd.isna(pm25):
            return ("Unknown", -1)
        for idx, (lo, hi, label) in enumerate(AQI_BREAKPOINTS):
            if lo <= pm25 <= hi:
                return (label, idx)
        return ("Hazardous", 5)

    results = df["pm25"].apply(_classify)
    df["aqi_category"] = results.apply(lambda x: x[0])
    df["aqi_numeric"]  = results.apply(lambda x: x[1])

    distribution = df["aqi_category"].value_counts().to_dict()
    logger.info("AQI label distribution: %s", distribution)

    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_feature_engineering(
    input_path: Path | None = None,
    output_path: Path | None = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline: load → build features → save.

    Steps (in order):
        1. Load cleaned CSV from clean.py output
        2. Sort by timestamp (required for lags and rolling windows)
        3. Add lag features
        4. Add rolling window features
        5. Add time features
        6. Add interaction features
        7. Add AQI label
        8. Optionally drop rows with NaN in any feature column
        9. Save to processed data directory

    Args:
        input_path:       Path to cleaned CSV. Defaults to PROCESSED_DATA_DIR / "cleaned.csv".
        output_path:      Path to save feature CSV. Defaults to PROCESSED_DATA_DIR / "features.csv".
        drop_incomplete:  If True, drop rows where any feature is NaN (default True).
                          Set to False if you want to inspect which rows have gaps.

    Returns:
        DataFrame with all features added, ready for model training.
    """
    if input_path is None:
        input_path = settings.PROCESSED_DATA_DIR / "cleaned.csv"
    if output_path is None:
        output_path = settings.PROCESSED_DATA_DIR / "features.csv"

    logger.info("=" * 60)
    logger.info("Starting feature engineering pipeline")
    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {input_path}. "
            "Run clean.py first."
        )

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # ── Sort — mandatory for lags and rolling windows ─────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Shift target by forecast horizon ─────────────────────────────────────
    # The model predicts PM2.5 at T+FORECAST_HORIZON_HOURS using features at T.
    # We shift pm25 backwards by the horizon so each row's target is the
    # future value the model should predict.
    # Rows at the end of the series (last FORECAST_HORIZON_HOURS rows) will
    # have NaN target and are dropped in the drop_incomplete step.
    horizon = settings.FORECAST_HORIZON_HOURS
    df["pm25"] = df["pm25"].shift(-horizon)
    logger.info(
        "Target shifted by -%d hours (forecasting T+%dh). "
        "Last %d rows will have NaN target and be dropped.",
        horizon, horizon, horizon,
    )

    # ── Feature building ──────────────────────────────────────────────────────
    logger.info("\n[1/6] Adding lag features...")
    df = add_lag_features(df)

    logger.info("\n[2/6] Adding rolling window features...")
    df = add_rolling_features(df)

    logger.info("\n[3/6] Adding time features...")
    df = add_time_features(df)

    logger.info("\n[4/6] Adding interaction features...")
    df = add_interaction_features(df)

    logger.info("\n[5/6] Adding OWM delta features...")
    df = add_owm_delta_features(df)

    logger.info("\n[6/6] Adding AQI label...")
    df = add_aqi_label(df)

    # ── Drop rows with any remaining NaN in feature columns ───────────────────
    if drop_incomplete:
        before = len(df)
        # Exclude non-feature columns and entirely-NaN columns (e.g. owm_*
        # columns on OWM free tier — always NaN, not useful to drop rows for)
        exclude_cols = {"timestamp", "aqi_category"}
        all_null_cols = [
            c for c in df.columns
            if c not in exclude_cols and df[c].isna().all()
        ]
        if all_null_cols:
            logger.warning(
                "Skipping %d fully-NaN column(s) from completeness check "
                "(likely OWM paid-tier features): %s",
                len(all_null_cols), all_null_cols,
            )
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and c not in all_null_cols
        ]
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            logger.info(
                "Dropped %d rows with incomplete features (expected: "
                "first %d rows due to max lag window). %d rows remain.",
                dropped, max(LAG_HOURS), len(df),
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info("Final shape:   %d rows × %d columns", *df.shape)
    logger.info(
        "Date range:    %s → %s",
        df["timestamp"].min(), df["timestamp"].max()
    )
    logger.info("Feature columns (%d):", len(df.columns))
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_note = f"  ← {null_count} NaN" if null_count > 0 else ""
        logger.info("  %-35s%s", col, null_note)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("\nFeature data saved → %s", output_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    df = run_feature_engineering()
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())