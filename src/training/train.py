from __future__ import annotations

"""
LightGBM model training pipeline with MLflow tracking + walk-forward CV.

WHAT THIS MODULE DOES:
1. Loads the feature-engineered CSV from Phase 2
2. Drops columns that are 100% NaN for Yangon (pm10, co, no2, o3)
3. Drops pm25_owm_delta — contains pm25 as a component, would leak the target
4. Runs walk-forward cross-validation to measure stability across time periods
5. Performs a final time-based train/test split and trains the production model
6. Logs CV metrics, final metrics, hyperparameters, and model artifact to MLflow
7. Registers the model in the MLflow Model Registry under settings.MODEL_NAME

WHY WALK-FORWARD CV:
A single 80/20 split gives one RMSE number. That number could be accidentally
good or bad depending on which month ended up in the test window. Walk-forward
CV runs 5 successive splits across the full timeline and reports mean + std RMSE.
Low std = the model is stable across time periods, not just lucky on one window.

"""

import logging
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Columns to always drop before training ────────────────────────────────────
ALWAYS_DROP_COLS: list[str] = ["pm10", "co", "no2", "o3"]

# ── Non-feature columns (target, labels, identifiers) ─────────────────────────
NON_FEATURE_COLS: list[str] = [
    "timestamp", "pm25", "aqi_category", "aqi_numeric", "pm25_owm_delta",
]

# ── LightGBM hyperparameters ──────────────────────────────────────────────────
DEFAULT_PARAMS: dict = {
    "n_estimators":       200,
    "learning_rate":      0.05,
    "max_depth":          4,
    "num_leaves":         15,
    "subsample":          0.7,
    "colsample_bytree":   0.7,
    "min_child_samples":  20,
    "reg_alpha":          0.5,
    "reg_lambda":         2.0,
    "random_state":       42,
    "n_jobs":             2,       # -1 hangs on Docker Desktop/Windows — cap at 2
    "verbose":            -1,
}

EARLY_STOPPING_ROUNDS: int = 30


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_features(input_path: Path | None = None) -> pd.DataFrame:
    if input_path is None:
        input_path = settings.PROCESSED_DATA_DIR / "features.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature file not found at {input_path}. "
            "Run clean.py then features.py first."
        )
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), input_path)
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cols_to_drop = [c for c in ALWAYS_DROP_COLS if c in df.columns]
    if cols_to_drop:
        logger.info("Dropping always-NaN columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    all_null = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and df[c].isna().all()
    ]
    if all_null:
        logger.warning("Dropping additional fully-NaN columns: %s", all_null)
        df = df.drop(columns=all_null)

    y = df["pm25"].copy()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()

    logger.info(
        "Feature matrix: %d rows × %d features. Target: pm25 (mean=%.2f, std=%.2f)",
        len(X), len(feature_cols), y.mean(), y.std(),
    )
    return X, y, feature_cols


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    m = {
        f"{prefix}rmse": round(rmse, 4),
        f"{prefix}mae":  round(mae,  4),
        f"{prefix}r2":   round(r2,   4),
    }
    logger.info(
        "%s — RMSE: %.4f  MAE: %.4f  R²: %.4f",
        prefix.upper().rstrip("_") or "METRICS", rmse, mae, r2,
    )
    return m


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    if test_size is None:
        test_size = settings.TEST_SIZE
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(
        "Time-based split: train=%d rows (%.0f%%), test=%d rows (%.0f%%)",
        len(X_train), (1 - test_size) * 100,
        len(X_test),  test_size * 100,
    )
    return X_train, X_test, y_train, y_test, str(split_idx)


def _train_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
) -> tuple[float, float, float]:
    """
    Train one LightGBM fold and return (rmse, mae, r2) on the validation set.
    Internal helper used by walk_forward_cv().
    """
    fit_params = {k: v for k, v in params.items() if k != "verbose"}
    model = LGBMRegressor(**fit_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    mae  = mean_absolute_error(y_val, preds)
    r2   = r2_score(y_val, preds)
    return round(rmse, 4), round(mae, 4), round(r2, 4)


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_folds: int | None = None,
    min_train_size: float | None = None,
) -> dict[str, float]:
    """
    Run walk-forward (expanding window) cross-validation.

    HOW IT WORKS:
    The dataset is divided into (n_folds + 1) equal chunks. Each fold uses
    all data up to chunk k as training and chunk k+1 as validation. The
    training window always starts from row 0 and grows — this mirrors real
    deployment where all historical data is always available.

    Example with 5 folds on 1459 rows (chunks of ~243 rows each):
      Fold 1: Train [0:729]    Validate [729:972]
      Fold 2: Train [0:972]    Validate [972:1215]
      Fold 3: Train [0:1215]   Validate [1215:1459]
      ...

    WHY EXPANDING WINDOW (NOT SLIDING):
    A sliding window discards older data each fold. For PM2.5 forecasting,
    older data is still valuable — seasonal patterns from 3 months ago are
    relevant. Expanding window retains all history, which better reflects
    how the final model will be trained.

    Args:
        X:              Feature matrix, sorted chronologically.
        y:              Target series.
        params:         LightGBM hyperparameters.
        n_folds:        Number of folds. Defaults to settings.CV_N_FOLDS.
        min_train_size: Minimum fraction of data for the first fold's training
                        set. Defaults to settings.CV_MIN_TRAIN_SIZE.

    Returns:
        Dict with per-fold metrics and summary statistics:
          cv_fold_{k}_rmse / _mae / _r2  for k in 1..n_folds
          cv_mean_rmse, cv_std_rmse
          cv_mean_mae,  cv_std_mae
          cv_mean_r2,   cv_std_r2
    """
    if n_folds is None:
        n_folds = settings.CV_N_FOLDS
    if min_train_size is None:
        min_train_size = settings.CV_MIN_TRAIN_SIZE

    n = len(X)
    # First train cutoff respects min_train_size
    first_train_end = int(n * min_train_size)
    # Remaining data split into n_folds equal validation windows
    remaining = n - first_train_end
    fold_size  = remaining // n_folds

    if fold_size < 10:
        logger.warning(
            "Fold size is very small (%d rows). "
            "Consider reducing CV_N_FOLDS or increasing data volume.",
            fold_size,
        )

    logger.info(
        "Walk-forward CV: %d folds, first train end=%d, fold_size~%d",
        n_folds, first_train_end, fold_size,
    )

    fold_metrics: list[tuple] = []

    for k in range(n_folds):
        train_end = first_train_end + k * fold_size
        val_end   = train_end + fold_size

        # Last fold takes all remaining rows to avoid a tiny leftover window
        if k == n_folds - 1:
            val_end = n

        X_tr  = X.iloc[:train_end]
        y_tr  = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]

        logger.info(
            "Fold %d/%d — train=[0:%d] (%d rows)  val=[%d:%d] (%d rows)",
            k + 1, n_folds,
            train_end, len(X_tr),
            train_end, val_end, len(X_val),
        )

        rmse, mae, r2 = _train_fold(X_tr, y_tr, X_val, y_val, params)
        fold_metrics.append((rmse, mae, r2))

        logger.info(
            "  Fold %d result — RMSE: %.4f  MAE: %.4f  R²: %.4f",
            k + 1, rmse, mae, r2,
        )

    rmses = [m[0] for m in fold_metrics]
    maes  = [m[1] for m in fold_metrics]
    r2s   = [m[2] for m in fold_metrics]

    cv_metrics: dict[str, float] = {}

    for k, (rmse, mae, r2) in enumerate(fold_metrics):
        cv_metrics[f"cv_fold_{k+1}_rmse"] = rmse
        cv_metrics[f"cv_fold_{k+1}_mae"]  = mae
        cv_metrics[f"cv_fold_{k+1}_r2"]   = r2

    cv_metrics["cv_mean_rmse"] = round(float(np.mean(rmses)), 4)
    cv_metrics["cv_std_rmse"]  = round(float(np.std(rmses)),  4)
    cv_metrics["cv_mean_mae"]  = round(float(np.mean(maes)),  4)
    cv_metrics["cv_std_mae"]   = round(float(np.std(maes)),   4)
    cv_metrics["cv_mean_r2"]   = round(float(np.mean(r2s)),   4)
    cv_metrics["cv_std_r2"]    = round(float(np.std(r2s)),    4)

    logger.info(
        "CV summary — mean_rmse=%.4f (±%.4f)  mean_r2=%.4f (±%.4f)",
        cv_metrics["cv_mean_rmse"], cv_metrics["cv_std_rmse"],
        cv_metrics["cv_mean_r2"],   cv_metrics["cv_std_r2"],
    )

    return cv_metrics


# ── Main training pipeline ────────────────────────────────────────────────────

def run_training(
    input_path: Path | None = None,
    params: dict | None = None,
    register_model: bool = True,
    run_cv: bool = True,
) -> str:
    """
    Full training pipeline: load → CV → final train → evaluate → log to MLflow.

    Args:
        input_path:     Path to features CSV. Defaults to PROCESSED_DATA_DIR/features.csv.
        params:         LightGBM hyperparameters. Defaults to DEFAULT_PARAMS.
        register_model: Whether to register model in MLflow registry.
        run_cv:         Whether to run walk-forward CV before final training.
                        Set to False to skip CV and train faster (e.g. for quick reruns).

    Returns:
        MLflow run ID of the completed training run.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # ── MLflow connection with retry ──────────────────────────────────────────
    # Inside Docker, MLflow may take a few seconds after container start.
    # We retry up to 5 times with 5s delay before failing hard.
    import time, urllib.error
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    for attempt in range(1, 6):
        try:
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            break
        except Exception as e:
            if attempt == 5:
                raise RuntimeError(
                    f"Cannot connect to MLflow at {settings.MLFLOW_TRACKING_URI} "
                    f"after 5 attempts. Check that the mlflow service is running."
                ) from e
            logger.warning(
                "MLflow not ready (attempt %d/5): %s — retrying in 5s...", attempt, e
            )
            time.sleep(5)

    df = load_features(input_path)
    X, y, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test, split_idx = time_based_split(X, y)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s", run_id)

        mlflow.set_tags({
            "city":                settings.CITY_NAME,
            "model_type":          "lightgbm",
            "split_index":         split_idx,
            "n_features":          len(feature_cols),
            "training_rows":       len(X_train),
            "test_rows":           len(X_test),
            "cv_folds":            settings.CV_N_FOLDS if run_cv else 0,
            "seasonal_note":       "dry+burning season only — monsoon data pending backfill",
        })

        mlflow.log_params(params)
        mlflow.log_param("early_stopping_rounds", EARLY_STOPPING_ROUNDS)
        mlflow.log_param("feature_cols",          feature_cols)
        mlflow.log_param("test_size",             settings.TEST_SIZE)
        mlflow.log_param("n_rows_total",          len(X))
        mlflow.log_param("cv_n_folds",            settings.CV_N_FOLDS if run_cv else 0)
        mlflow.log_param("cv_min_train_size",     settings.CV_MIN_TRAIN_SIZE if run_cv else "n/a")

        # ── Walk-forward CV ────────────────────────────────────────────────────
        if run_cv:
            logger.info("\n── Walk-forward cross-validation (%d folds) ──", settings.CV_N_FOLDS)
            cv_metrics = walk_forward_cv(X_train, y_train, params)
            mlflow.log_metrics(cv_metrics)
            logger.info(
                "CV complete — mean_rmse=%.4f ±%.4f",
                cv_metrics["cv_mean_rmse"], cv_metrics["cv_std_rmse"],
            )
        else:
            logger.info("CV skipped (run_cv=False)")
            cv_metrics = {}

        # ── Final model — trained on full training set ─────────────────────────
        logger.info("\n── Final model training ──")
        fit_params = {k: v for k, v in params.items() if k != "verbose"}
        model = LGBMRegressor(**fit_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                log_evaluation(period=-1),
            ],
        )
        logger.info("Best iteration: %d / %d", model.best_iteration_, params["n_estimators"])

        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)

        train_metrics = compute_metrics(y_train, train_pred, prefix="train_")
        test_metrics  = compute_metrics(y_test,  test_pred,  prefix="test_")

        mlflow.log_metrics({**train_metrics, **test_metrics})
        mlflow.log_metric("best_iteration", model.best_iteration_)

        # ── Feature importance ────────────────────────────────────────────────
        importance = pd.Series(
            model.feature_importances_, index=feature_cols,
        ).sort_values(ascending=False)

        logger.info("Top 10 feature importances:")
        for feat, score in importance.head(10).items():
            logger.info("  %-30s %.4f", feat, score)

        importance_path = settings.PROCESSED_DATA_DIR / "feature_importance.csv"
        try:
            importance_path.parent.mkdir(parents=True, exist_ok=True)
            importance.reset_index().rename(
                columns={"index": "feature", 0: "importance"}
            ).to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path), artifact_path="reports")
        except Exception as e:
            logger.warning("Could not save feature importance artifact: %s", e)

        # ── Log model ─────────────────────────────────────────────────────────
        signature = mlflow.models.infer_signature(X_train, train_pred)
        if register_model:
            mlflow.lightgbm.log_model(
                model,
                artifact_path="model",
                registered_model_name=settings.MODEL_NAME,
                signature=signature,
                input_example=X_train.head(5),
            )
            logger.info("Model registered as '%s'", settings.MODEL_NAME)
        else:
            mlflow.lightgbm.log_model(model, artifact_path="model", signature=signature)

        # ── Final summary ─────────────────────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info("Run ID:           %s", run_id)
        logger.info("Best iteration:   %d", model.best_iteration_)
        if run_cv:
            logger.info(
                "CV RMSE:          %.4f ± %.4f  (std=low is good)",
                cv_metrics["cv_mean_rmse"], cv_metrics["cv_std_rmse"],
            )
        logger.info("Train RMSE:       %.4f", train_metrics["train_rmse"])
        logger.info("Test  RMSE:       %.4f", test_metrics["test_rmse"])
        logger.info("Test  MAE:        %.4f", test_metrics["test_mae"])
        logger.info("Test  R²:         %.4f", test_metrics["test_r2"])
        logger.info("MLflow UI:        %s/#/experiments", settings.MLFLOW_TRACKING_URI)

    return run_id


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    run_training()