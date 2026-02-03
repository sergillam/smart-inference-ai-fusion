#!/usr/bin/env python3
"""Experimento exemplo com verificação formal habilitada via variável de ambiente."""

import logging
import os

from smart_inference_ai_fusion.experiments.common import run_standard_experiment
from smart_inference_ai_fusion.models.random_forest_classifier_model import (
    RandomForestClassifierModel,
)
from smart_inference_ai_fusion.utils.types import (
    DatasetSourceType,
    SklearnDatasetName,
    VerificationConfig,
)

logger = logging.getLogger(__name__)


def run():
    """Executa experimento com verificação condicional baseada em variáveis de ambiente."""

    # Verificar se verificação formal está habilitada via variável de ambiente
    verification_enabled = os.getenv("VERIFICATION_ENABLED", "false").lower() == "true"
    verification_strict = os.getenv("VERIFICATION_STRICT", "false").lower() == "true"

    if verification_enabled:
        logger.info("🔍 Formal verification ENABLED via environment variable")

        # Configurar verificação formal
        verification_config = VerificationConfig(
            enabled=True,
            timeout=30.0,
            fail_on_error=verification_strict,
            constraints={
                "shape_preservation": True,  # Z3-compatible constraint
                "bounds": True,  # Z3-compatible constraint
                "range_check": True,  # Z3-compatible constraint
                "type_safety": True,  # Z3-compatible constraint
                "bounds_tolerance": 0.1 if not verification_strict else 0.05,
            },
        )

        verification_mode = "STRICT" if verification_strict else "FLEXIBLE"
        logger.info("Verification mode: %s", verification_mode)

    else:
        logger.info("🚫 Formal verification DISABLED")
        verification_config = None

    # Executar experimento
    logger.info("🚀 Starting Random Forest experiment on DIGITS dataset...")

    baseline_metrics, inference_metrics = run_standard_experiment(
        model_class=RandomForestClassifierModel,
        model_name="RandomForest_EnvVerification",
        dataset_source=DatasetSourceType.SKLEARN,
        dataset_name=SklearnDatasetName.DIGITS,
        model_params={"n_estimators": 10, "random_state": 42},
        verification_config=verification_config,
    )

    # Relatório de resultados
    logger.info("📊 EXPERIMENT RESULTS:")
    logger.info("  Baseline accuracy: %.4f", baseline_metrics.get("accuracy", 0.0))
    logger.info("  Inference accuracy: %.4f", inference_metrics.get("accuracy", 0.0))
    logger.info("  Baseline time: %.2fs", baseline_metrics.get("execution_time_seconds", 0))
    logger.info("  Inference time: %.2fs", inference_metrics.get("execution_time_seconds", 0))

    if verification_enabled:
        logger.info("✅ Experiment completed WITH formal verification")
    else:
        logger.info("✅ Experiment completed WITHOUT formal verification")

    return {"baseline": baseline_metrics, "inference": inference_metrics}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
