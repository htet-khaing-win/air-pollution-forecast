from __future__ import annotations

"""
Airflow DAG — weekly PM2.5 model retraining pipeline.

SCHEDULE: Every Sunday at midnight UTC (0 0 * * 0).

PIPELINE:
  fetch_data → preprocess → train → evaluate → promote_if_better

TASK BREAKDOWN:
  1. fetch_data       — runs ingest.py: pulls 7 days of new data from
                        OpenAQ, OWM, and Open-Meteo, appends to master CSV.
  2. preprocess       — runs clean.py + features.py: cleans raw data,
                        rebuilds all lag/rolling/time features on the full
                        growing dataset.
  3. train            — runs train.py: trains LightGBM with walk-forward CV,
                        logs run + model artifact to MLflow.
  4. evaluate         — runs evaluate.py: compares new model RMSE against
                        @champion, promotes if improvement >= 4%.
  5. promote_if_better — reads the evaluate result and sends a summary
                         log. No-op if model was not promoted.


USAGE:

  To trigger manually:
    airflow dags trigger pm25_retrain_pipeline

  To backfill a missed run:
    airflow dags backfill pm25_retrain_pipeline --start-date 2026-03-16
"""

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ── DAG default arguments ─────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner":            "airflow",
    "depends_on_past":  False,
    "email_on_failure": False,        
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),  # hard kill if task hangs > 2h
}


# ── Task callables ─────────────────────────────────────────────────────────────
# Each callable is a thin wrapper around the existing pipeline functions.
# Wrappers are intentionally minimal — all logic lives in the pipeline modules.

def task_fetch_data(**context) -> dict:
    """
    Fetch 7 days of new data from OpenAQ, OWM, and Open-Meteo.
    Appends to the master CSV with deduplication.

    Returns a summary dict pushed to XCom.
    """
    from src.ingestion.ingest import run_ingestion
    from config.settings import settings

    logger.info("Starting data ingestion for week ending %s", context["ds"])
    df = run_ingestion(lookback_days=settings.LOOKBACK_DAYS)

    summary = {
        "rows_ingested": len(df),
        "date_range_min": str(df["timestamp"].min()) if not df.empty else "N/A",
        "date_range_max": str(df["timestamp"].max()) if not df.empty else "N/A",
        "execution_date": context["ds"],
    }

    if df.empty:
        raise ValueError(
            "Ingestion returned empty DataFrame — no data from any source. "
            "Check API keys and connectivity."
        )

    logger.info(
        "Ingestion complete: %d rows (%s → %s)",
        summary["rows_ingested"],
        summary["date_range_min"],
        summary["date_range_max"],
    )
    return summary


def task_preprocess(**context) -> dict:
    """
    Clean raw data and rebuild all features on the full growing dataset.

    Two steps:
        1. clean.py  — drops missing targets, removes impossible values,
                       forward-fills small weather gaps
        2. features.py — rebuilds lag/rolling/time/interaction features
    """
    from src.preprocessing.clean import run_cleaning
    from src.preprocessing.features import run_feature_engineering

    logger.info("Starting preprocessing...")

    df_clean = run_cleaning()
    if df_clean.empty:
        raise ValueError(
            "Cleaning returned empty DataFrame. "
            "Check that master CSV has valid pm25 rows."
        )

    df_features = run_feature_engineering()
    if df_features.empty:
        raise ValueError(
            "Feature engineering returned empty DataFrame. "
            "Check cleaned.csv has enough rows for lag windows."
        )

    summary = {
        "rows_after_clean":    len(df_clean),
        "rows_after_features": len(df_features),
        "n_feature_cols":      len(df_features.columns),
    }

    logger.info(
        "Preprocessing complete: %d clean rows → %d feature rows (%d columns)",
        summary["rows_after_clean"],
        summary["rows_after_features"],
        summary["n_feature_cols"],
    )
    return summary


def task_train(**context) -> dict:
    """
    Train LightGBM with walk-forward CV and log to MLflow.

    Returns the MLflow run_id and key metrics via XCom.
    """
    from src.training.train import run_training
    import mlflow
    from config.settings import settings

    logger.info("Starting model training...")

    run_id = run_training(register_model=True, run_cv=True)

    # Retrieve logged metrics from the MLflow run for XCom
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    run    = client.get_run(run_id)

    summary = {
        "run_id":          run_id,
        "train_rmse":      run.data.metrics.get("train_rmse"),
        "test_rmse":       run.data.metrics.get("test_rmse"),
        "test_r2":         run.data.metrics.get("test_r2"),
        "cv_mean_rmse":    run.data.metrics.get("cv_mean_rmse"),
        "cv_std_rmse":     run.data.metrics.get("cv_std_rmse"),
        "best_iteration":  run.data.metrics.get("best_iteration"),
    }

    logger.info(
        "Training complete: run_id=%s  test_rmse=%.4f  cv_mean_rmse=%.4f±%.4f",
        run_id,
        summary["test_rmse"] or 0,
        summary["cv_mean_rmse"] or 0,
        summary["cv_std_rmse"] or 0,
    )
    return summary


def task_evaluate(**context) -> dict:
    """
    Compare new model against @champion and promote if RMSE improved >= 4%.

    Returns the evaluation result dict via XCom.
    """
    from src.training.evaluate import run_evaluation

    logger.info("Starting model evaluation...")
    result = run_evaluation()

    logger.info(
        "Evaluation complete: promoted=%s  new_rmse=%.4f  champion_rmse=%s",
        result["promoted"],
        result.get("new_rmse", 0),
        result.get("prod_rmse", "N/A"),
    )
    return result


def task_promote_if_better(**context) -> None:
    """
    Log a structured summary of the pipeline run outcome.

    Reads evaluate result from XCom and logs whether a promotion occurred,
    what the RMSE delta was, and what action was taken.

    This task always succeeds — it is purely observational.
    In production you would replace the logger calls here with a Slack
    message, email alert, or PagerDuty notification.
    """
    ti      = context["task_instance"]
    train   = ti.xcom_pull(task_ids="train")   or {}
    eval_r  = ti.xcom_pull(task_ids="evaluate") or {}
    fetch   = ti.xcom_pull(task_ids="fetch_data") or {}
    preproc = ti.xcom_pull(task_ids="preprocess") or {}

    promoted     = eval_r.get("promoted", False)
    new_rmse     = eval_r.get("new_rmse")
    prod_rmse    = eval_r.get("prod_rmse")
    new_version  = eval_r.get("new_version")
    prod_version = eval_r.get("prod_version")
    reason       = eval_r.get("reason", "")

    logger.info("=" * 60)
    logger.info("WEEKLY PIPELINE SUMMARY — %s", context["ds"])
    logger.info("=" * 60)
    logger.info("Data:      %d rows ingested", fetch.get("rows_ingested", 0))
    logger.info(
        "Features:  %d rows after preprocessing (%d columns)",
        preproc.get("rows_after_features", 0),
        preproc.get("n_feature_cols", 0),
    )
    logger.info(
        "Training:  run_id=%s  test_rmse=%.4f  cv=%.4f±%.4f",
        train.get("run_id", "unknown"),
        train.get("test_rmse") or 0,
        train.get("cv_mean_rmse") or 0,
        train.get("cv_std_rmse") or 0,
    )

    if promoted:
        logger.info(
            "Promotion: ✓ VERSION %s PROMOTED TO @champion  "
            "(RMSE %.4f → %.4f, version %s archived)",
            new_version, prod_rmse or 0, new_rmse or 0, prod_version,
        )
    else:
        logger.info(
            "Promotion: ✗ NO CHANGE  "
            "(new=%.4f vs champion=%.4f)  %s",
            new_rmse or 0, prod_rmse or 0, reason,
        )

    logger.info("=" * 60)

# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="pm25_retrain_pipeline",
    description="Weekly PM2.5 model retraining — fetch → preprocess → train → evaluate → promote",
    schedule_interval="0 0 * * 0",     # every Sunday at midnight UTC
    start_date=datetime(2026, 3, 23),  # first run: Sunday 2026-03-29
    catchup=False,                     # don't backfill missed runs on first deploy
    default_args=DEFAULT_ARGS,
    tags=["ml", "pm25", "yangon", "lightgbm"],
    max_active_runs=1,                 # prevent overlapping runs if a task is slow
    doc_md="""
## PM2.5 Weekly Retraining Pipeline

Runs every Sunday at midnight UTC. Fetches the latest 7 days of air quality
and weather data, rebuilds features on the full dataset, trains a new
LightGBM model with walk-forward CV, and promotes it to @champion if RMSE
improves by ≥ 4% over the current champion.

**City:** Yangon, Myanmar
**Model:** LightGBM PM2.5 regressor
**Forecast horizon:** 24 hours ahead
**MLflow experiment:** air_pollution_forecast
    """,
) as dag:

    fetch_data = PythonOperator(
        task_id="fetch_data",
        python_callable=task_fetch_data,
        doc_md="Fetch 7 days of OpenAQ + OWM + Open-Meteo data and append to master CSV.",
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
        doc_md="Run clean.py + features.py on the full growing dataset.",
    )

    train = PythonOperator(
        task_id="train",
        python_callable=task_train,
        doc_md="Train LightGBM with walk-forward CV and register in MLflow.",
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate,
        doc_md="Compare new model vs @champion. Promote if RMSE improved >= 4%.",
    )

    promote_if_better = PythonOperator(
        task_id="promote_if_better",
        python_callable=task_promote_if_better,
        doc_md="Log pipeline summary. Hook for Slack/email alerts in production.",
    )

    # ── Task dependencies ─────────────────────────────────────────────────────
    fetch_data >> preprocess >> train >> evaluate >> promote_if_better