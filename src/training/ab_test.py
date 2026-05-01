from __future__ import annotations

"""
A/B Test runner — evaluates all configured models on the same data split.

Reads model configs from ab_test_config.yaml, trains/evaluates each model
on identical train/test splits, and logs all results to MLflow under a
dedicated experiment so runs are directly comparable side-by-side.

USAGE:
  python -m src.training.ab_test
  python -m src.training.ab_test --config path/to/ab_test_config.yaml
  python -m src.training.ab_test --model xgboost lightgbm   # run subset only

"""

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Shared utilities ──────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_features(cfg: dict) -> pd.DataFrame:
    path = settings.PROCESSED_DATA_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"features.csv not found at {path}. Run clean.py + features.py first."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def prepare_tabular(df: pd.DataFrame, cfg: dict) -> tuple:
    """
    Prepare X, y for tabular models (XGBoost, LightGBM, Ridge).
    Drops always-NaN and non-feature columns, returns feature matrix + target.
    """
    exp = cfg["experiment"]
    drop  = [c for c in exp["drop_cols"] if c in df.columns]
    excl  = set(exp["non_feature_cols"])

    df = df.drop(columns=drop)
    all_null = [c for c in df.columns if c not in excl and df[c].isna().all()]
    if all_null:
        logger.warning("Dropping additional fully-NaN columns: %s", all_null)
        df = df.drop(columns=all_null)

    y = df[exp["target_col"]].copy()
    feature_cols = [c for c in df.columns if c not in excl]
    X = df[feature_cols].copy()
    return X, y, feature_cols


def time_split(X: pd.DataFrame, y: pd.Series, test_size: float):
    idx = int(len(X) * (1 - test_size))
    return X.iloc[:idx], X.iloc[idx:], y.iloc[:idx], y.iloc[idx:]


def metrics(y_true, y_pred, prefix="") -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {
        f"{prefix}rmse": round(float(rmse), 4),
        f"{prefix}mae":  round(float(mae),  4),
        f"{prefix}r2":   round(float(r2),   4),
    }



def walk_forward_folds(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> list[tuple]:
    """
    Generate walk-forward (expanding window) train/validation fold indices.

    Used by tabular model runners (XGBoost, LightGBM, Ridge) to evaluate
    stability across time periods. Prophet and Chronos skip CV — they are
    univariate models where a single time split is sufficient.

    Args:
        X:   Feature matrix sorted chronologically.
        y:   Target series.
        cfg: Full config dict (reads cross_validation block).

    Yields:
        Tuples of (X_tr, X_val, y_tr, y_val, fold_number).
    """
    cv_cfg         = cfg.get("cross_validation", {})
    n_folds        = cv_cfg.get("n_folds",        settings.CV_N_FOLDS)
    min_train_size = cv_cfg.get("min_train_size",  settings.CV_MIN_TRAIN_SIZE)
    test_size      = cfg["experiment"]["test_size"]

    # CV runs only on the training portion — keep test set untouched
    n_train    = int(len(X) * (1 - test_size))
    X_tr_pool  = X.iloc[:n_train]
    y_tr_pool  = y.iloc[:n_train]

    n          = len(X_tr_pool)
    first_end  = int(n * min_train_size)
    fold_size  = (n - first_end) // n_folds

    folds = []
    for k in range(n_folds):
        train_end = first_end + k * fold_size
        val_end   = train_end + fold_size if k < n_folds - 1 else n
        folds.append((
            X_tr_pool.iloc[:train_end],
            X_tr_pool.iloc[train_end:val_end],
            y_tr_pool.iloc[:train_end],
            y_tr_pool.iloc[train_end:val_end],
            k + 1,
        ))
    return folds


# ── Model runners ─────────────────────────────────────────────────────────────

def run_xgboost(df: pd.DataFrame, model_cfg: dict, cfg: dict) -> dict:
    from xgboost import XGBRegressor
    params = model_cfg["params"].copy()
    early  = params.pop("early_stopping_rounds", None)
    if early:
        params["early_stopping_rounds"] = early

    X, y, feats = prepare_tabular(df, cfg)
    X_tr, X_te, y_tr, y_te = time_split(X, y, cfg["experiment"]["test_size"])

    # Walk-forward CV
    cv_metrics = {}
    fold_rmses = []
    for X_f_tr, X_f_val, y_f_tr, y_f_val, k in walk_forward_folds(X, y, cfg):
        fold_model = XGBRegressor(**params)
        fold_model.fit(X_f_tr, y_f_tr, eval_set=[(X_f_val, y_f_val)], verbose=False)
        fold_rmse = mean_squared_error(y_f_val, fold_model.predict(X_f_val)) ** 0.5
        cv_metrics[f"cv_fold_{k}_rmse"] = round(float(fold_rmse), 4)
        fold_rmses.append(fold_rmse)
    if fold_rmses:
        cv_metrics["cv_mean_rmse"] = round(float(np.mean(fold_rmses)), 4)
        cv_metrics["cv_std_rmse"]  = round(float(np.std(fold_rmses)),  4)

    # Final model
    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    m = {
        **cv_metrics,
        **metrics(y_tr, model.predict(X_tr), "train_"),
        **metrics(y_te, model.predict(X_te), "test_"),
    }
    return {"metrics": m, "params": params, "n_features": len(feats)}


def run_lightgbm(df: pd.DataFrame, model_cfg: dict, cfg: dict) -> dict:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    params = model_cfg["params"].copy()
    early  = params.pop("early_stopping_rounds", None)
    params.pop("verbose", None)

    X, y, feats = prepare_tabular(df, cfg)
    X_tr, X_te, y_tr, y_te = time_split(X, y, cfg["experiment"]["test_size"])

    def _make_callbacks(e):
        cbs = [log_evaluation(period=-1)]
        if e:
            cbs.append(early_stopping(stopping_rounds=e, verbose=False))
        return cbs

    # Walk-forward CV
    cv_metrics = {}
    fold_rmses = []
    for X_f_tr, X_f_val, y_f_tr, y_f_val, k in walk_forward_folds(X, y, cfg):
        fold_model = LGBMRegressor(**params, verbose=-1)
        fold_model.fit(X_f_tr, y_f_tr, eval_set=[(X_f_val, y_f_val)],
                       callbacks=_make_callbacks(early))
        fold_rmse = mean_squared_error(y_f_val, fold_model.predict(X_f_val)) ** 0.5
        cv_metrics[f"cv_fold_{k}_rmse"] = round(float(fold_rmse), 4)
        fold_rmses.append(fold_rmse)
    if fold_rmses:
        cv_metrics["cv_mean_rmse"] = round(float(np.mean(fold_rmses)), 4)
        cv_metrics["cv_std_rmse"]  = round(float(np.std(fold_rmses)),  4)

    # Final model
    model = LGBMRegressor(**params, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=_make_callbacks(early))

    m = {
        **cv_metrics,
        **metrics(y_tr, model.predict(X_tr), "train_"),
        **metrics(y_te, model.predict(X_te), "test_"),
    }
    params["early_stopping_rounds"] = early
    return {"metrics": m, "params": params, "n_features": len(feats)}


def run_ridge(df: pd.DataFrame, model_cfg: dict, cfg: dict) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    params = model_cfg["params"].copy()
    X, y, feats = prepare_tabular(df, cfg)
    X_tr, X_te, y_tr, y_te = time_split(X, y, cfg["experiment"]["test_size"])

    # Walk-forward CV
    cv_metrics = {}
    fold_rmses = []
    for X_f_tr, X_f_val, y_f_tr, y_f_val, k in walk_forward_folds(X, y, cfg):
        fold_model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(**params))])
        fold_model.fit(X_f_tr, y_f_tr)
        fold_rmse = mean_squared_error(y_f_val, fold_model.predict(X_f_val)) ** 0.5
        cv_metrics[f"cv_fold_{k}_rmse"] = round(float(fold_rmse), 4)
        fold_rmses.append(fold_rmse)
    if fold_rmses:
        cv_metrics["cv_mean_rmse"] = round(float(np.mean(fold_rmses)), 4)
        cv_metrics["cv_std_rmse"]  = round(float(np.std(fold_rmses)),  4)

    # Final model
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(**params))])
    model.fit(X_tr, y_tr)

    m = {
        **cv_metrics,
        **metrics(y_tr, model.predict(X_tr), "train_"),
        **metrics(y_te, model.predict(X_te), "test_"),
    }
    return {"metrics": m, "params": params, "n_features": len(feats)}


def run_prophet(df: pd.DataFrame, model_cfg: dict, cfg: dict) -> dict:
    from prophet import Prophet

    params = model_cfg["params"].copy()
    test_size = cfg["experiment"]["test_size"]

    # Prophet only needs ds (timestamp) and y (target)
    pdf = df[["timestamp", "pm25"]].copy()
    pdf.columns = ["ds", "y"]
    # Prophet requires timezone-naive timestamps
    pdf["ds"] = pdf["ds"].dt.tz_localize(None)

    split_idx = int(len(pdf) * (1 - test_size))
    train_df  = pdf.iloc[:split_idx]
    test_df   = pdf.iloc[split_idx:]

    model = Prophet(**params)
    # Suppress Prophet's verbose Stan output
    import logging as _log
    _log.getLogger("prophet").setLevel(_log.WARNING)
    _log.getLogger("cmdstanpy").setLevel(_log.WARNING)

    model.fit(train_df)

    # Predict on training period (in-sample)
    train_forecast = model.predict(train_df[["ds"]])
    # Predict on test period (out-of-sample)
    test_forecast  = model.predict(test_df[["ds"]])

    m = {
        **metrics(train_df["y"].values, train_forecast["yhat"].values, "train_"),
        **metrics(test_df["y"].values,  test_forecast["yhat"].values,  "test_"),
    }
    return {"metrics": m, "params": params, "n_features": 1}


def run_chronos(df: pd.DataFrame, model_cfg: dict, cfg: dict) -> dict:
    import torch
    from chronos import ChronosPipeline

    params       = model_cfg["params"].copy()
    test_size    = cfg["experiment"]["test_size"]
    pred_length  = params.get("prediction_length", 24)
    num_samples  = params.get("num_samples", 20)
    model_id     = params.get("model_id", "amazon/chronos-t5-small")
    device       = params.get("device", "cpu")
    torch_dtype  = getattr(torch, params.get("torch_dtype", "bfloat16"))

    split_idx  = int(len(df) * (1 - test_size))
    pm25_series = df["pm25"].values.astype(float)
    y_train = pm25_series[:split_idx]
    y_test  = pm25_series[split_idx:]

    logger.info("Loading Chronos model: %s (this may take a minute)...", model_id)
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch_dtype,       # torch_dtype deprecated in newer chronos — use dtype
    )

    # Chronos takes context (training series) and predicts forward
    # We use a sliding window to generate predictions across the full test set
    context_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(0)

    all_preds = []
    # Predict in chunks of pred_length across the test period
    # Chronos >=1.4: predict() takes context_tensor, not context
    steps = 0
    context = y_train.tolist()

    while steps < len(y_test):
        chunk_len = min(pred_length, len(y_test) - steps)
        ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        forecast = pipeline.predict(
            context_tensor=ctx,
            prediction_length=chunk_len,
            num_samples=num_samples,
        )
        # forecast shape: (1, num_samples, chunk_len) → take median
        chunk_pred = forecast[0].median(dim=0).values.numpy()
        all_preds.extend(chunk_pred[:chunk_len].tolist())

        # Extend context with actual values (not predictions) — avoids drift
        context.extend(y_test[steps:steps + chunk_len].tolist())
        steps += chunk_len

    y_pred_test = np.array(all_preds[:len(y_test)])

    # For train metrics use in-sample prediction (context reconstruction)
    train_ctx = torch.tensor(y_train[:-pred_length], dtype=torch.float32).unsqueeze(0)
    train_fc  = pipeline.predict(context_tensor=train_ctx, prediction_length=pred_length, num_samples=num_samples)
    y_pred_train_tail = train_fc[0].median(dim=0).values.numpy()
    # Only compute train metrics on the last pred_length window (only available chunk)
    y_train_tail = y_train[-pred_length:]

    m = {
        **metrics(y_train_tail, y_pred_train_tail, "train_"),
        **metrics(y_test,       y_pred_test,        "test_"),
    }
    return {"metrics": m, "params": params, "n_features": 1}


# ── Model dispatcher ──────────────────────────────────────────────────────────

RUNNERS = {
    "xgboost":  run_xgboost,
    "lightgbm": run_lightgbm,
    "ridge":    run_ridge,
    "prophet":  run_prophet,
    "chronos":  run_chronos,
}


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_ab_test(
    config_path: Path | None = None,
    only_models: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run all enabled models and log each to MLflow.

    Args:
        config_path:  Path to YAML config. Defaults to ab_test_config.yaml
                      in the same directory as this script.
        only_models:  Optional list of model names to run (subset filter).

    Returns:
        DataFrame with one row per model showing all metrics — sorted by test_rmse.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "ab_test_config.yaml"

    cfg = load_config(config_path)
    df  = load_features(cfg)

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg["experiment"]["mlflow_experiment_name"])

    results = []

    for model_cfg in cfg["models"]:
        name    = model_cfg["name"]
        enabled = model_cfg.get("enabled", True)

        if not enabled:
            logger.info("Skipping '%s' (disabled in config)", name)
            continue

        if only_models and name not in only_models:
            logger.info("Skipping '%s' (not in --model filter)", name)
            continue

        runner = RUNNERS.get(model_cfg["type"])
        if runner is None:
            logger.error("Unknown model type '%s' for model '%s'", model_cfg["type"], name)
            continue

        logger.info("\n" + "=" * 60)
        logger.info("Running: %s", name.upper())
        logger.info("=" * 60)

        try:
            with mlflow.start_run(run_name=name):
                mlflow.set_tag("model_type", model_cfg["type"])
                mlflow.set_tag("city", settings.CITY_NAME)

                result = runner(df, model_cfg, cfg)

                mlflow.log_params(result["params"])
                mlflow.log_param("n_features", result["n_features"])
                mlflow.log_metrics(result["metrics"])

                logger.info(
                    "%-12s  train_rmse=%-8.4f  test_rmse=%-8.4f  "
                    "test_mae=%-8.4f  test_r2=%.4f",
                    name,
                    result["metrics"].get("train_rmse", float("nan")),
                    result["metrics"].get("test_rmse",  float("nan")),
                    result["metrics"].get("test_mae",   float("nan")),
                    result["metrics"].get("test_r2",    float("nan")),
                )
                results.append({"model": name, **result["metrics"]})

        except ImportError as e:
            logger.error(
                "Cannot run '%s' — missing dependency: %s. "
                "Install it and retry.", name, e,
            )
        except Exception as e:
            logger.error("'%s' failed with error: %s", name, e, exc_info=True)

    if not results:
        logger.warning("No models completed successfully.")
        return pd.DataFrame()

    # ── Final leaderboard ─────────────────────────────────────────────────────
    leaderboard_df = pd.DataFrame(results)
    # Show the most important columns first in the printed table
    display_cols = ["model", "cv_mean_rmse", "cv_std_rmse", "test_rmse", "test_mae", "test_r2"]
    display_cols = [c for c in display_cols if c in leaderboard_df.columns]
    leaderboard = (
        leaderboard_df
        .sort_values("test_rmse")
        .reset_index(drop=True)
    )
    leaderboard.index += 1  # rank starts at 1

    print("\n" + "=" * 80)
    print("A/B TEST LEADERBOARD  (sorted by test RMSE — lower is better)")
    print("=" * 80)
    print(leaderboard[display_cols].to_string())
    print("=" * 80)
    print("  cv_mean_rmse = average RMSE across walk-forward folds (stability indicator)")
    print("  cv_std_rmse  = std dev across folds — low means consistent across time")
    print(f"\nFull results in MLflow: {settings.MLFLOW_TRACKING_URI}")

    return leaderboard


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="A/B test multiple forecasting models")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: ab_test_config.yaml next to this script)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Run only specific models, e.g. --model xgboost lightgbm",
    )
    args = parser.parse_args()

    run_ab_test(config_path=args.config, only_models=args.model)