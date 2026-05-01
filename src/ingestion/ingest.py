from __future__ import annotations

"""
Data ingestion orchestrator — combines 4 sources into one master CSV.

DATA SOURCES (in priority order for pm25 values):
1. OpenAQ             — ground-truth PM2.5 sensor readings (highest trust)
2. WAQI historical    — daily historical PM2.5 disaggregated to hourly
                        (AQI → µg/m³ converted, covers Dec 2020 – Jan 2025)
3. OWM air pollution  — model-driven PM2.5/PM10/CO/NO2/O3 (owm_* prefix)
4. Open-Meteo         — historical temperature, humidity, wind, pressure

MERGE STRATEGY:
All four DataFrames are outer-joined on the hourly timestamp. For pm25,
OpenAQ takes priority over WAQI — where both exist for the same hour,
OpenAQ wins. WAQI fills the historical gap (Dec 2020 – Dec 2025) where
OpenAQ has no data.

PM2.5 SOURCE PRIORITY:
  OpenAQ  → real ground sensor  → highest trust
  WAQI    → daily AQI converted → medium trust (disaggregated)
  owm_pm25 → OWM model estimate → supplement only (never used as pm25 target)

WAQI DATA PLACEMENT:
Put the WAQI CSV at: data/raw/waqi_yangon.csv
If the file is not found, the pipeline continues without it (graceful skip).
"""

import logging
from pathlib import Path

import pandas as pd

from config.settings import settings
from src.ingestion.openaq import fetch_openaq_data
from src.ingestion.openweather import fetch_owm_air_pollution_history
from src.ingestion.openmeteo import fetch_openmeteo_data
from src.ingestion.waqi_historical import fetch_waqi_historical

logger = logging.getLogger(__name__)

# Expected location of the WAQI CSV file
WAQI_CSV_PATH: Path = settings.RAW_DATA_DIR / "waqi_yangon.csv"


def merge_all_sources(
    df_openaq: pd.DataFrame,
    df_waqi: pd.DataFrame,
    df_owm: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Outer-join all four source DataFrames on hourly timestamp.

    Merge order:
        1. WAQI historical + OpenAQ  →  df_pm25  (pm25 column, OpenAQ wins)
        2. df_pm25 + OWM AQI         →  df_poll  (adds owm_* columns)
        3. df_poll + Open-Meteo      →  df_merged (adds weather columns)

    PM2.5 priority logic:
        Where both WAQI and OpenAQ have a value for the same hour,
        OpenAQ ground-truth takes priority. WAQI fills hours where
        OpenAQ has no reading.

    Args:
        df_openaq:  Ground-truth pollution from OpenAQ.
        df_waqi:    Historical daily→hourly PM2.5 from WAQI.
        df_owm:     Model-driven AQI estimates from OWM (owm_* columns).
        df_weather: Hourly weather from Open-Meteo.

    Returns:
        Merged DataFrame with all columns, sorted by timestamp.
    """
    source_status = {
        "OpenAQ":     not df_openaq.empty,
        "WAQI":       not df_waqi.empty,
        "OWM AQI":    not df_owm.empty,
        "Open-Meteo": not df_weather.empty,
    }
    for name, available in source_status.items():
        if not available:
            logger.warning("%s returned no data — its columns will be NaN.", name)

    # ── Step 1: Merge pm25 from WAQI + OpenAQ (OpenAQ wins) ──────────────────
    if not df_waqi.empty and not df_openaq.empty:
        # Outer join WAQI and OpenAQ on timestamp
        df_pm25 = pd.merge(
            df_waqi.rename(columns={"pm25": "pm25_waqi"}),
            df_openaq,
            on="timestamp",
            how="outer",
        )
        # OpenAQ pm25 takes priority; fall back to WAQI where OpenAQ is NaN
        df_pm25["pm25"] = df_pm25["pm25"].fillna(df_pm25["pm25_waqi"])
        df_pm25 = df_pm25.drop(columns=["pm25_waqi"])
        logger.info(
            "Merged WAQI + OpenAQ: %d rows "
            "(%d from WAQI, %d from OpenAQ, OpenAQ takes priority on overlap)",
            len(df_pm25), len(df_waqi), len(df_openaq),
        )
    elif not df_waqi.empty:
        df_pm25 = df_waqi
        logger.info("Using WAQI only for pm25 (%d rows)", len(df_pm25))
    elif not df_openaq.empty:
        df_pm25 = df_openaq
        logger.info("Using OpenAQ only for pm25 (%d rows)", len(df_pm25))
    else:
        df_pm25 = pd.DataFrame()
        logger.error("Both WAQI and OpenAQ are empty — no pm25 target available.")

    # ── Step 2: Add OWM air pollution supplement ──────────────────────────────
    if df_pm25.empty:
        df_poll = df_owm if not df_owm.empty else pd.DataFrame()
    elif not df_owm.empty:
        df_poll = pd.merge(df_pm25, df_owm, on="timestamp", how="outer")
        logger.info("Merged pm25 sources + OWM: %d rows", len(df_poll))
    else:
        df_poll = df_pm25

    # ── Step 3: Add Open-Meteo weather ───────────────────────────────────────
    if df_poll.empty:
        df_merged = df_weather if not df_weather.empty else pd.DataFrame()
    elif not df_weather.empty:
        df_merged = pd.merge(df_poll, df_weather, on="timestamp", how="outer")
        logger.info(
            "Merged all sources: %d rows, %d columns",
            len(df_merged), len(df_merged.columns),
        )
    else:
        df_merged = df_poll

    if df_merged.empty:
        logger.error("All sources empty — nothing to save.")
        return pd.DataFrame()

    df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)
    return df_merged


def append_to_master_csv(
    df_new: pd.DataFrame,
    filepath: Path | None = None,
) -> Path:
    """
    Append new data to the master CSV, deduplicating by timestamp.

    Strategy:
    1. Load existing CSV if it exists
    2. Concatenate old + new
    3. Drop duplicates — keep last (newest data wins, handles corrections)
    4. Save back

    Args:
        df_new:   New data to append.
        filepath: Path to master CSV. Defaults to settings path.

    Returns:
        Path to the saved CSV file.
    """
    if filepath is None:
        filepath = settings.RAW_DATA_DIR / settings.RAW_FILENAME

    if filepath.exists():
        df_existing = pd.read_csv(filepath, parse_dates=["timestamp"])
        logger.info("Existing master CSV: %d rows", len(df_existing))
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["timestamp"], keep="last")
    else:
        df_combined = df_new

    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(filepath, index=False)

    logger.info("Master CSV saved: %d total rows → %s", len(df_combined), filepath)
    return filepath


def run_ingestion(lookback_days: int | None = None) -> pd.DataFrame:
    """
    Full ingestion pipeline: fetch all sources → merge → save.

    For the daily Airflow run, WAQI is skipped (it's a static historical
    file, not a live API). OpenAQ, OWM, and Open-Meteo handle incremental
    updates. The WAQI data is already in the master CSV from the one-time
    backfill run.

    Args:
        lookback_days: Override lookback window. Defaults to settings.LOOKBACK_DAYS.

    Returns:
        The merged DataFrame that was appended to the master CSV.
    """
    if lookback_days is None:
        lookback_days = settings.LOOKBACK_DAYS

    logger.info("=" * 60)
    logger.info("Starting data ingestion for %s", settings.CITY_NAME)
    logger.info("Lookback: %d days", lookback_days)
    logger.info("=" * 60)

    # ── Source 1: WAQI historical (static file — load once) ───────────────────
    if WAQI_CSV_PATH.exists():
        logger.info("\n[1/4] Loading WAQI historical data from %s...", WAQI_CSV_PATH)
        df_waqi = fetch_waqi_historical(WAQI_CSV_PATH)
        logger.info("WAQI: %d hourly rows", len(df_waqi))
    else:
        logger.info(
            "\n[1/4] WAQI CSV not found at %s — skipping. "
            "Place waqi_yangon.csv in data/raw/ to enable historical backfill.",
            WAQI_CSV_PATH,
        )
        df_waqi = pd.DataFrame()

    # ── Source 2: OpenAQ ground-truth sensor readings ─────────────────────────
    logger.info("\n[2/4] Fetching OpenAQ pollution data...")
    df_openaq = fetch_openaq_data(lookback_days=lookback_days)
    logger.info(
        "OpenAQ: %d rows%s", len(df_openaq),
        f", columns: {list(df_openaq.columns)}" if not df_openaq.empty else " (empty)",
    )

    # ── Source 3: OWM model-driven air pollution supplement ───────────────────
    logger.info("\n[3/4] Fetching OWM air pollution history...")
    df_owm = fetch_owm_air_pollution_history(lookback_days=lookback_days)
    logger.info(
        "OWM AQI: %d rows%s", len(df_owm),
        f", columns: {list(df_owm.columns)}" if not df_owm.empty else " (empty)",
    )

    # ── Source 4: Open-Meteo historical weather ───────────────────────────────
    logger.info("\n[4/4] Fetching Open-Meteo weather...")
    df_weather = fetch_openmeteo_data(lookback_days=lookback_days)
    logger.info(
        "Open-Meteo: %d rows%s", len(df_weather),
        f", columns: {list(df_weather.columns)}" if not df_weather.empty else " (empty)",
    )

    # ── Merge all four ────────────────────────────────────────────────────────
    logger.info("\nMerging all sources...")
    df_merged = merge_all_sources(df_openaq, df_waqi, df_owm, df_weather)

    if df_merged.empty:
        logger.error("No data ingested. Check API connectivity and keys.")
        return df_merged

    csv_path = append_to_master_csv(df_merged)

    logger.info("\n" + "=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Rows in this run: %d", len(df_merged))
    logger.info(
        "Date range: %s → %s",
        df_merged["timestamp"].min(), df_merged["timestamp"].max(),
    )
    logger.info("Columns: %s", list(df_merged.columns))
    logger.info("File: %s", csv_path)
    logger.info("\nMissing values:\n%s", df_merged.isnull().sum().to_string())

    return df_merged


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    run_ingestion()