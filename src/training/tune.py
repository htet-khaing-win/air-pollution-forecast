from __future__ import annotations

"""
Optuna hyperparameter tuning for the LightGBM PM2.5 forecasting model.

WHAT THIS MODULE DOES:
1. Loads the same feature matrix used by train.py
2. Runs an Optuna study over the 3-phase parameter search space
3. Each trial is evaluated using walk-forward CV mean RMSE — same metric
   used in train.py, so results are directly comparable
4. Every trial is logged to MLflow as a child run under a dedicated
   "pm25_tuning" experiment
5. Prints the best parameters at the end — paste into train.py DEFAULT_PARAMS

WHY OPTUNA:
Manual tuning requires running train.py repeatedly and reading logs.
Optuna uses Tree-structured Parzen Estimator (TPE) — a Bayesian search
that learns which parameter regions are promising and focuses trials there.
50 trials with TPE typically outperforms 200 random trials.

WHY CV AS THE OBJECTIVE (NOT A SINGLE SPLIT):
A single 80/20 split RMSE is noisy — one set of params might look great
because it got a "lucky" test window. CV mean RMSE across 5 folds is a
much more stable signal for the optimizer to follow.

SEARCH SPACE (mirrors your 3-phase tuning framework):
  Phase 1 — Capacity:     max_depth, num_leaves, min_child_samples
  Phase 2 — Learning:     n_estimators, learning_rate
  Phase 3 — Regularization: subsample, colsample_bytree, reg_alpha, reg_lambda

PRUNING:
Optuna's MedianPruner stops unpromising trials early (after fold 2 of CV)
if their intermediate RMSE is worse than the median of completed trials.
This saves time without sacrificing search quality.

USAGE:
  python -m src.training.tune                    # 50 trials (default)
  python -m src.training.tune --trials 100       # more thorough search
  python -m src.training.tune --trials 20        # quick test

OUTPUT:
  Best params printed to terminal + saved to data/processed/best_params.yaml
  All trials visible in MLflow UI under experiment "pm25_tuning"
"""

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from config.settings import settings
from src.training.train import (
    load_features,
    prepare_features,
    EARLY_STOPPING_ROUNDS,
)

logger = logging.getLogger(__name__)

# ── Tuning experiment — separate from main training experiment ─────────────────
TUNING_EXPERIMENT_NAME: str = "pm25_tuning"
DEFAULT_N_TRIALS: int = 50

# ── Search space bounds ────────────────────────────────────────────────────────
# These ranges cover both "tighter for overfitting" and "looser for underfitting"
# directions from your current defaults, as per the 3-phase framework.
SEARCH_SPACE = {
    # Phase 1 — Capacity
    "max_depth":         (3, 8),
    "num_leaves":        (7, 63),       # kept < 2^max_depth by trial logic
    "min_child_samples": (10, 150),

    # Phase 2 — Learning speed
    "n_estimators":      (100, 1000),
    "learning_rate":     (0.005, 0.15),

    # Phase 3 — Regularisation / stochasticity
    "subsample":         (0.4, 1.0),
    "colsample_bytree":  (0.4, 1.0),
    "reg_alpha":         (0.0, 5.0),
    "reg_lambda":        (0.5, 10.0),
}


def get_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int | None = None,
    min_train_size: float | None = None,
) -> list[tuple]:
    """
    Build walk-forward CV folds from the training portion of the data.
    Identical logic to walk_forward_cv() in train.py — kept local to avoid
    circular imports between tune.py and train.py.

    Returns list of (X_tr, X_val, y_tr, y_val) tuples.
    """
    if n_folds is None:
        n_folds = settings.CV_N_FOLDS
    if min_train_size is None:
        min_train_size = settings.CV_MIN_TRAIN_SIZE

    # Use only the training portion (first 80%) — test set is never touched
    n_train   = int(len(X) * (1 - settings.TEST_SIZE))
    X_pool    = X.iloc[:n_train]
    y_pool    = y.iloc[:n_train]

    n          = len(X_pool)
    first_end  = int(n * min_train_size)
    fold_size  = (n - first_end) // n_folds

    folds = []
    for k in range(n_folds):
        train_end = first_end + k * fold_size
        val_end   = (train_end + fold_size) if k < n_folds - 1 else n
        folds.append((
            X_pool.iloc[:train_end].copy(),
            X_pool.iloc[train_end:val_end].copy(),
            y_pool.iloc[:train_end].copy(),
            y_pool.iloc[train_end:val_end].copy(),
        ))
    return folds


def make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple],
    mlflow_run_id: str,
) -> callable:
    """
    Build the Optuna objective function.

    The objective trains one LightGBM model per CV fold with the trial's
    suggested parameters, reports intermediate RMSE after each fold for
    pruning, and returns the mean RMSE across all folds.

    Args:
        X:             Full feature matrix (used only for shape reference).
        y:             Full target series (used only for shape reference).
        folds:         Pre-built CV fold list from get_cv_folds().
        mlflow_run_id: Parent MLflow run ID — each trial is logged as a
                       nested child run under this parent.

    Returns:
        Optuna objective callable.
    """
    def objective(trial: optuna.Trial) -> float:

        # ── Sample parameters (3-phase order) ────────────────────────────────
        max_depth = trial.suggest_int(
            "max_depth", *SEARCH_SPACE["max_depth"]
        )
        # num_leaves must be < 2^max_depth — enforce the constraint
        max_safe_leaves = min(63, 2 ** max_depth - 1)
        num_leaves = trial.suggest_int(
            "num_leaves",
            SEARCH_SPACE["num_leaves"][0],
            max_safe_leaves,
        )
        min_child_samples = trial.suggest_int(
            "min_child_samples", *SEARCH_SPACE["min_child_samples"]
        )
        n_estimators = trial.suggest_int(
            "n_estimators", *SEARCH_SPACE["n_estimators"], step=50
        )
        learning_rate = trial.suggest_float(
            "learning_rate", *SEARCH_SPACE["learning_rate"], log=True
        )
        subsample = trial.suggest_float(
            "subsample", *SEARCH_SPACE["subsample"]
        )
        colsample_bytree = trial.suggest_float(
            "colsample_bytree", *SEARCH_SPACE["colsample_bytree"]
        )
        reg_alpha = trial.suggest_float(
            "reg_alpha", *SEARCH_SPACE["reg_alpha"]
        )
        reg_lambda = trial.suggest_float(
            "reg_lambda", *SEARCH_SPACE["reg_lambda"]
        )

        params = {
            "n_estimators":      n_estimators,
            "learning_rate":     learning_rate,
            "max_depth":         max_depth,
            "num_leaves":        num_leaves,
            "min_child_samples": min_child_samples,
            "subsample":         subsample,
            "colsample_bytree":  colsample_bytree,
            "reg_alpha":         reg_alpha,
            "reg_lambda":        reg_lambda,
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }

        # ── Run CV folds ──────────────────────────────────────────────────────
        fold_rmses: list[float] = []

        with mlflow.start_run(
            run_name=f"trial_{trial.number:03d}",
            nested=True,
        ):
            mlflow.log_params(params)
            mlflow.set_tag("trial_number", trial.number)

            for fold_idx, (X_tr, X_val, y_tr, y_val) in enumerate(folds):
                model = LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        early_stopping(
                            stopping_rounds=EARLY_STOPPING_ROUNDS,
                            verbose=False,
                        ),
                        log_evaluation(period=-1),
                    ],
                )

                preds     = model.predict(X_val)
                fold_rmse = mean_squared_error(y_val, preds) ** 0.5
                fold_rmses.append(fold_rmse)

                mlflow.log_metric(
                    f"cv_fold_{fold_idx + 1}_rmse", round(fold_rmse, 4)
                )

                # Report intermediate value for pruning — Optuna can stop
                # bad trials after the first couple of folds
                trial.report(float(np.mean(fold_rmses)), step=fold_idx)
                if trial.should_prune():
                    mlflow.set_tag("pruned", True)
                    raise optuna.TrialPruned()

            mean_rmse = float(np.mean(fold_rmses))
            std_rmse  = float(np.std(fold_rmses))

            mlflow.log_metric("cv_mean_rmse", round(mean_rmse, 4))
            mlflow.log_metric("cv_std_rmse",  round(std_rmse,  4))

        return mean_rmse

    return objective


def run_tuning(n_trials: int = DEFAULT_N_TRIALS) -> dict:
    """
    Run the full Optuna tuning study.

    Args:
        n_trials: Number of Optuna trials to run.

    Returns:
        Dict of best hyperparameters found.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(TUNING_EXPERIMENT_NAME)

    # ── Load and prepare data (same as train.py) ──────────────────────────────
    df = load_features()
    X, y, feature_cols = prepare_features(df)
    folds = get_cv_folds(X, y)

    logger.info("=" * 60)
    logger.info("OPTUNA HYPERPARAMETER SEARCH")
    logger.info("=" * 60)
    logger.info("Model:      LightGBM")
    logger.info("City:       %s", settings.CITY_NAME)
    logger.info("Features:   %d", len(feature_cols))
    logger.info("CV folds:   %d", settings.CV_N_FOLDS)
    logger.info("Trials:     %d", n_trials)
    logger.info("Objective:  walk-forward CV mean RMSE (minimize)")
    logger.info("=" * 60)

    # ── Run study inside a parent MLflow run ──────────────────────────────────
    with mlflow.start_run(run_name=f"optuna_study_{n_trials}_trials") as parent_run:
        mlflow.set_tags({
            "city":       settings.CITY_NAME,
            "n_trials":   n_trials,
            "cv_folds":   settings.CV_N_FOLDS,
            "model_type": "lightgbm",
        })

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(
                n_startup_trials=10,    # don't prune until 10 trials complete
                n_warmup_steps=2,       # always run at least 2 folds before pruning
            ),
        )

        objective = make_objective(X, y, folds, parent_run.info.run_id)

        # Suppress Optuna's per-trial output — we log to MLflow instead
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info("Starting Optuna study... (check MLflow UI for trial progress)")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # ── Log best result to parent run ─────────────────────────────────────
        best_params  = study.best_params
        best_value   = study.best_value
        best_trial   = study.best_trial.number
        n_pruned     = len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.PRUNED])
        n_complete   = len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_mean_rmse", round(best_value, 4))
        mlflow.log_metric("best_trial_number", best_trial)
        mlflow.log_metric("n_trials_complete", n_complete)
        mlflow.log_metric("n_trials_pruned",   n_pruned)

        # ── Save best params to YAML ──────────────────────────────────────────
        output_path = settings.PROCESSED_DATA_DIR / "best_params.yaml"
        # Add fixed params not searched by Optuna
        full_best_params = {
            **best_params,
            "random_state": 42,
            "n_jobs":       -1,
            "verbose":      -1,
        }
        with open(output_path, "w") as f:
            yaml.dump(full_best_params, f, default_flow_style=False, sort_keys=True)
        mlflow.log_artifact(str(output_path), artifact_path="reports")
        logger.info("Best params saved to %s", output_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 60)
    logger.info("Best trial:       #%d", best_trial)
    logger.info("Best CV RMSE:     %.4f", best_value)
    logger.info("Trials complete:  %d", n_complete)
    logger.info("Trials pruned:    %d", n_pruned)
    logger.info("\nBest parameters:")
    for k, v in sorted(best_params.items()):
        logger.info("  %-22s %s", k, v)

    # ── Print copy-paste block for train.py ───────────────────────────────────
    print("\n" + "=" * 60)
    print("PASTE THIS INTO train.py DEFAULT_PARAMS:")
    print("=" * 60)
    print("DEFAULT_PARAMS: dict = {")
    for k, v in sorted(full_best_params.items()):
        if isinstance(v, float):
            print(f'    "{k}":{" " * max(1, 22 - len(k))}{v},')
        else:
            print(f'    "{k}":{" " * max(1, 22 - len(k))}{v},')
    print("}")
    print(f"\nSaved to: {output_path}")
    print(f"MLflow:   {settings.MLFLOW_TRACKING_URI}/#/experiments")

    return full_best_params


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for LightGBM PM2.5 model"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Number of Optuna trials (default: {DEFAULT_N_TRIALS})",
    )
    args = parser.parse_args()

    run_tuning(n_trials=args.trials)