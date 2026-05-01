from __future__ import annotations

"""
Model evaluation and auto-promotion pipeline.

ALIAS CONVENTIONS USED:
  @champion   — the current production model (equivalent to old "Production")
  @challenger — the previous champion, kept for rollback

PROMOTION LOGIC:
  new_rmse <= champion_rmse * (1 - PROMOTION_THRESHOLD)  ->  promote

FIRST RUN (no @champion yet):
  The latest version is promoted unconditionally.
"""

import logging

import mlflow
from mlflow.tracking import MlflowClient

from config.settings import settings

logger = logging.getLogger(__name__)

# Pulled from settings — change settings.PROMOTION_THRESHOLD to adjust
PROMOTION_THRESHOLD: float = settings.PROMOTION_THRESHOLD


def _get_client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    return MlflowClient()


def get_latest_model_version(client: MlflowClient):
    """
    Get the highest version number registered for settings.MODEL_NAME.
    Works regardless of MLflow version — does not rely on stages or aliases.

    Returns:
        ModelVersion object, or None if model doesn't exist yet.
    """
    try:
        versions = client.search_model_versions(f"name='{settings.MODEL_NAME}'")
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.error(
                "Model '%s' not found in registry. Run train.py first.",
                settings.MODEL_NAME,
            )
            return None
        raise

    if not versions:
        logger.error(
            "No versions found for model '%s'. Run train.py first.",
            settings.MODEL_NAME,
        )
        return None

    # Sort by version number descending, return the latest
    return sorted(versions, key=lambda v: int(v.version), reverse=True)[0]


def get_champion_version(client: MlflowClient):
    """
    Get the model version currently tagged as @champion.

    Returns:
        ModelVersion object, or None if no @champion alias set yet.
    """
    try:
        mv = client.get_model_version_by_alias(settings.MODEL_NAME, "champion")
        return mv
    except mlflow.exceptions.MlflowException:
        return None


def get_run_metric(client: MlflowClient, run_id: str, metric_name: str = "test_rmse"):
    """Retrieve a logged metric from an MLflow run by run_id."""
    try:
        run = client.get_run(run_id)
        value = run.data.metrics.get(metric_name)
        if value is None:
            logger.warning(
                "Metric '%s' not found in run %s. Available: %s",
                metric_name, run_id, list(run.data.metrics.keys()),
            )
        return value
    except mlflow.exceptions.MlflowException as e:
        logger.error("Failed to retrieve run %s: %s", run_id, e)
        return None


def set_alias(client: MlflowClient, version: str, alias: str) -> None:
    """Assign an alias to a specific model version."""
    client.set_registered_model_alias(
        name=settings.MODEL_NAME,
        alias=alias,
        version=version,
    )
    logger.info("Alias '@%s' -> version %s", alias, version)


def delete_alias(client: MlflowClient, alias: str) -> None:
    """Remove an alias if it exists."""
    try:
        client.delete_registered_model_alias(
            name=settings.MODEL_NAME,
            alias=alias,
        )
    except mlflow.exceptions.MlflowException:
        pass  # alias didn't exist — fine


def run_evaluation() -> dict:
    """
    Compare the latest model version against the current @champion.
    Promotes to @champion if RMSE improves by >= PROMOTION_THRESHOLD.

    Returns:
        Dict with promoted (bool), versions, rmse values, and reason string.
    """
    client = _get_client()

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION — %s", settings.MODEL_NAME)
    logger.info("Promotion threshold: %.0f%% RMSE improvement required",
                PROMOTION_THRESHOLD * 100)
    logger.info("=" * 60)

    # ── Get the latest registered version ────────────────────────────────────
    new_version = get_latest_model_version(client)
    if new_version is None:
        return {"promoted": False, "reason": "No model versions found."}

    new_rmse = get_run_metric(client, new_version.run_id, "test_rmse")
    if new_rmse is None:
        msg = (f"test_rmse not found for version {new_version.version}. "
               "Ensure train.py logged this metric.")
        logger.error(msg)
        return {"promoted": False, "reason": msg}

    logger.info(
        "Latest version: %s  run_id=%s  test_rmse=%.4f",
        new_version.version, new_version.run_id, new_rmse,
    )

    # ── Get current champion ──────────────────────────────────────────────────
    champion = get_champion_version(client)

    # ── First deployment — no champion yet ───────────────────────────────────
    if champion is None:
        logger.info(
            "No @champion exists yet. Promoting version %s unconditionally.",
            new_version.version,
        )
        set_alias(client, new_version.version, "champion")
        reason = (
            f"First deployment. Version {new_version.version} "
            f"promoted as @champion (test_rmse={new_rmse:.4f})."
        )
        logger.info(reason)
        return {
            "promoted":      True,
            "new_version":   new_version.version,
            "new_rmse":      new_rmse,
            "prod_version":  None,
            "prod_rmse":     None,
            "reason":        reason,
        }

    # ── Already champion — nothing to compare ────────────────────────────────
    if champion.version == new_version.version:
        reason = (
            f"Version {new_version.version} is already @champion. "
            "Train a new model version first."
        )
        logger.info(reason)
        return {
            "promoted":      False,
            "new_version":   new_version.version,
            "new_rmse":      new_rmse,
            "prod_version":  champion.version,
            "prod_rmse":     new_rmse,
            "reason":        reason,
        }

    champion_rmse = get_run_metric(client, champion.run_id, "test_rmse")
    if champion_rmse is None:
        msg = (f"Could not retrieve test_rmse for @champion "
               f"version {champion.version}. Skipping promotion.")
        logger.warning(msg)
        return {"promoted": False, "reason": msg}

    logger.info(
        "Current @champion: version %s  test_rmse=%.4f",
        champion.version, champion_rmse,
    )

    # ── Compare ───────────────────────────────────────────────────────────────
    improvement    = (champion_rmse - new_rmse) / champion_rmse
    threshold_rmse = champion_rmse * (1 - PROMOTION_THRESHOLD)

    logger.info(
        "RMSE: champion=%.4f  new=%.4f  improvement=%.1f%%  required=%.0f%%",
        champion_rmse, new_rmse, improvement * 100, PROMOTION_THRESHOLD * 100,
    )

    if new_rmse <= threshold_rmse:
        # Promote — move old champion to @challenger for rollback
        delete_alias(client, "challenger")
        set_alias(client, champion.version, "challenger")
        set_alias(client, new_version.version, "champion")

        reason = (
            f"Promoted version {new_version.version} (RMSE={new_rmse:.4f}) "
            f"over version {champion.version} (RMSE={champion_rmse:.4f}). "
            f"Improvement: {improvement * 100:.1f}% >= required "
            f"{PROMOTION_THRESHOLD * 100:.0f}%. "
            f"Previous champion is now @challenger."
        )
        promoted = True
    else:
        reason = (
            f"Version {new_version.version} (RMSE={new_rmse:.4f}) did not "
            f"improve enough over @champion version {champion.version} "
            f"(RMSE={champion_rmse:.4f}). "
            f"Improvement: {improvement * 100:.1f}% < required "
            f"{PROMOTION_THRESHOLD * 100:.0f}%. No change."
        )
        promoted = False

    logger.info("\n" + "=" * 60)
    logger.info("RESULT: %s", "PROMOTED" if promoted else "NO CHANGE")
    logger.info("=" * 60)
    logger.info(reason)

    return {
        "promoted":     promoted,
        "new_version":  new_version.version,
        "new_rmse":     new_rmse,
        "prod_version": champion.version,
        "prod_rmse":    champion_rmse,
        "reason":       reason,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    result = run_evaluation()
    print("\nResult:", result)