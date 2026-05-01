from __future__ import annotations

"""
One-time historical backfill script for initial model training.

RUN THIS ONCE before kicking off the daily Airflow pipeline.
After the backfill, the 7-day Airflow DAG handles incremental updates.

WHAT IT DOES:
1. Fetches BACKFILL_DAYS (default 365) of Open-Meteo weather history
2. Fetches BACKFILL_DAYS of OWM air pollution history (if paid key available)
3. Fetches as much OpenAQ history as the API allows (typically 90 days max
   on the free tier — OpenAQ does not store multi-year sensor archives)
4. Merges all three and appends to the master CSV with deduplication

WHY OPENAQ HISTORY IS LIMITED:
OpenAQ v3 stores sensor readings but does not expose a bulk historical
download on the free tier. The measurements endpoint supports date_from/date_to
but returns at most 1000 records per page per sensor. For 365 days at hourly
resolution that would require ~8760 API calls per sensor — impractical.

PRACTICAL BACKFILL STRATEGY:
- Open-Meteo: full 365 days  ✅ (free, single request)
- OWM:        full 365 days  ✅ (paid key, single request)
- OpenAQ:     90 days        ⚠️  (free tier practical limit)

The model will train on whatever overlap exists between the three sources.
With Open-Meteo + OWM covering a full year, XGBoost has enough weather
context even if OpenAQ PM2.5 only covers 90 days.

USAGE:
    python -m src.ingestion.backfill
    python -m src.ingestion.backfill --days 180   # custom window
"""

import argparse
import logging
from datetime import datetime

from config.settings import settings
from src.ingestion.ingest import merge_all_sources, append_to_master_csv
from src.ingestion.openaq import fetch_openaq_data
from src.ingestion.openweather import fetch_owm_air_pollution_history
from src.ingestion.openmeteo import fetch_openmeteo_backfill

logger = logging.getLogger(__name__)

# OpenAQ practical free-tier limit for bulk history
_OPENAQ_MAX_BACKFILL_DAYS = 90


def run_backfill(backfill_days: int | None = None) -> None:
    """
    Execute the one-time historical backfill.

    Args:
        backfill_days: Total days of history to pull.
                       Defaults to settings.BACKFILL_DAYS (365).
    """
    if backfill_days is None:
        backfill_days = settings.BACKFILL_DAYS

    openaq_days = min(backfill_days, _OPENAQ_MAX_BACKFILL_DAYS)

    logger.info("=" * 60)
    logger.info("BACKFILL START — %s", settings.CITY_NAME)
    logger.info("Target window:    %d days", backfill_days)
    logger.info("Open-Meteo:       %d days (full window)", backfill_days)
    logger.info("OWM AQI:          %d days (full window, paid key required)", backfill_days)
    logger.info("OpenAQ:           %d days (free-tier practical limit)", openaq_days)
    logger.info("Started at:       %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # ── Open-Meteo: full window, free ─────────────────────────────────────────
    logger.info("\n[1/3] Open-Meteo weather backfill (%d days)...", backfill_days)
    df_weather = fetch_openmeteo_backfill(backfill_days=backfill_days)
    logger.info(
        "Open-Meteo result: %d hourly records",
        len(df_weather),
    )

    # ── OWM: full window, requires paid key ───────────────────────────────────
    logger.info("\n[2/3] OWM air pollution backfill (%d days)...", backfill_days)
    df_owm = fetch_owm_air_pollution_history(lookback_days=backfill_days)
    if df_owm.empty:
        logger.warning(
            "OWM returned no data (free tier or missing key). "
            "owm_* columns will be NaN. The model can still train without them."
        )
    else:
        logger.info("OWM result: %d hourly records", len(df_owm))

    # ── OpenAQ: capped at practical free-tier limit ───────────────────────────
    logger.info("\n[3/3] OpenAQ pollution backfill (%d days)...", openaq_days)
    df_openaq = fetch_openaq_data(lookback_days=openaq_days)
    if df_openaq.empty:
        logger.warning(
            "OpenAQ returned no data. Check OPENAQ_API_KEY and sensor "
            "availability near %s.", settings.CITY_NAME
        )
    else:
        logger.info("OpenAQ result: %d hourly records", len(df_openaq))

    # ── Merge and save ────────────────────────────────────────────────────────
    logger.info("\nMerging all backfill sources...")
    df_merged = merge_all_sources(df_openaq, df_owm, df_weather)

    if df_merged.empty:
        logger.error("Backfill produced no data. Check API keys and connectivity.")
        return

    csv_path = append_to_master_csv(df_merged)

    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info("Total rows saved: %d", len(df_merged))
    logger.info(
        "Date range: %s → %s",
        df_merged["timestamp"].min(), df_merged["timestamp"].max(),
    )
    logger.info("File: %s", csv_path)
    logger.info("\nMissing values per column:")
    null_summary = df_merged.isnull().sum()
    null_summary = null_summary[null_summary > 0]
    if null_summary.empty:
        logger.info("  None — all columns fully populated!")
    else:
        for col, count in null_summary.items():
            pct = count / len(df_merged) * 100
            logger.info("  %-25s %5d rows  (%.1f%%)", col, count, pct)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="One-time historical backfill")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=f"Days of history to pull (default: settings.BACKFILL_DAYS = {settings.BACKFILL_DAYS})",
    )
    args = parser.parse_args()

    run_backfill(backfill_days=args.days)