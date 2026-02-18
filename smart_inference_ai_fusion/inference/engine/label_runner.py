"""Label inference engine for applying noise/perturbation techniques to target labels."""

import numpy as np

from smart_inference_ai_fusion.inference.transformations.label.label_confusion_matrix_noise import (
    LabelConfusionMatrixNoise,
)
from smart_inference_ai_fusion.inference.transformations.label.label_flip_near_border import (
    LabelFlipNearBorder,
)
from smart_inference_ai_fusion.inference.transformations.label.label_swap_within_class import (
    LabelSwapWithinClass,
)
from smart_inference_ai_fusion.inference.transformations.label.partial_label_noise import (
    PartialLabelNoise,
)
from smart_inference_ai_fusion.inference.transformations.label.random_label_noise import (
    RandomLabelNoise,
)
from smart_inference_ai_fusion.utils.types import LabelNoiseConfig


def _compute_label_distribution(y: np.ndarray) -> dict:
    """Compute label distribution statistics.

    Args:
        y: Label array

    Returns:
        Dictionary with distribution statistics
    """
    y_arr = np.asarray(y)
    total_samples = len(y_arr)

    # Handle empty label array to avoid division by zero
    if total_samples == 0:
        return {
            "unique_classes": 0,
            "total_samples": 0,
            "class_counts": {},
            "class_fractions": {},
            "class_balance": {
                "min_class_fraction": 0.0,
                "max_class_fraction": 0.0,
                "imbalance_ratio": 0.0,
            },
        }

    unique, counts = np.unique(y_arr, return_counts=True)

    distribution = {
        "unique_classes": len(unique),
        "total_samples": int(total_samples),
        "class_counts": {str(cls): int(cnt) for cls, cnt in zip(unique, counts)},
        "class_fractions": {
            str(cls): float(cnt / total_samples) for cls, cnt in zip(unique, counts)
        },
        "class_balance": {
            "min_class_fraction": float(counts.min() / total_samples) if len(counts) > 0 else 0.0,
            "max_class_fraction": float(counts.max() / total_samples) if len(counts) > 0 else 0.0,
            "imbalance_ratio": (
                float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
            ),
        },
    }
    return distribution


def _compute_label_changes(y_before: np.ndarray, y_after: np.ndarray) -> dict:
    """Compute statistics about label changes.

    Args:
        y_before: Original labels
        y_after: Transformed labels

    Returns:
        Dictionary with change statistics
    """
    y_before_arr = np.asarray(y_before)
    y_after_arr = np.asarray(y_after)

    changed_mask = y_before_arr != y_after_arr
    n_changed = int(changed_mask.sum())
    n_total = len(y_before_arr)

    stats = {
        "labels_changed": n_changed,
        "labels_unchanged": n_total - n_changed,
        "change_fraction": float(n_changed / n_total) if n_total > 0 else 0.0,
    }

    # Compute transition matrix (which classes changed to which)
    if n_changed > 0:
        transitions = {}
        for old, new in zip(y_before_arr[changed_mask], y_after_arr[changed_mask]):
            key = f"{old}->{new}"
            transitions[key] = transitions.get(key, 0) + 1
        stats["transitions"] = transitions
        stats["most_common_transition"] = max(transitions, key=transitions.get)
    else:
        stats["transitions"] = {}
        stats["most_common_transition"] = None

    return stats


# pylint: disable=too-many-positional-arguments
class LabelInferenceEngine:
    """Applies label noise and perturbation techniques to target labels (y) based on configuration.

    Supports: RandomLabelNoise, LabelFlipNearBorder, LabelConfusionMatrixNoise,
        PartialLabelNoise, LabelSwapWithinClass.
    """

    def __init__(self, config: LabelNoiseConfig, X_train=None):
        """Initializes the label inference engine.

        Args:
            config (LabelNoiseConfig):
                Configuration object for label noise techniques.
            X_train (Any, optional):
                Training features, required for certain techniques.
        """
        # This line signals to Pylint that X_train is intentionally used.
        _ = X_train

        self.label_pipeline = []

        # LabelFlipNearBorder is commented out for regression tasks because it requires
        # predict_proba/decision_function
        transformation_map = {
            "label_noise_fraction": RandomLabelNoise,
            "flip_near_border_fraction": LabelFlipNearBorder,
            # Excluded for regression: needs predict_proba
            "confusion_matrix_noise_level": LabelConfusionMatrixNoise,
            "partial_label_fraction": PartialLabelNoise,
            "swap_within_class_fraction": LabelSwapWithinClass,
        }

        for field, cls in transformation_map.items():
            value = getattr(config, field, None)
            if value is not None:
                self.label_pipeline.append(cls(value))

    def apply(
        self,
        y_train,
        y_test,
        model=None,
        X_train=None,
        X_test=None,
        collect_statistics: bool = True,
    ):
        """Applies the configured label transformations to the training and test labels.

        Args:
            y_train (Any):
                Original training labels.
            y_test (Any):
                Original test labels.
            model (Any, optional):
                Model instance, used if required by the technique.
            X_train (Any, optional):
                Training features.
            X_test (Any, optional):
                Test features.
            collect_statistics (bool):
                Whether to collect distribution statistics. Defaults to True.

        Returns:
            tuple: A tuple containing (y_train_perturbed, y_test_perturbed, statistics).
        """
        statistics = {
            "transformations_applied": [],
            "per_transformation_stats": [],
        }

        # Store original labels
        y_train_original = np.asarray(y_train).copy() if collect_statistics else None
        y_test_original = np.asarray(y_test).copy() if collect_statistics else None

        if collect_statistics:
            statistics["original"] = {
                "train": _compute_label_distribution(y_train),
                "test": _compute_label_distribution(y_test),
            }

        for transform in self.label_pipeline:
            transform_name = transform.__class__.__name__

            # Store state before transformation (both train and test)
            y_train_before = np.asarray(y_train).copy() if collect_statistics else None
            y_test_before = np.asarray(y_test).copy() if collect_statistics else None

            if getattr(transform, "requires_model", False):
                y_train = transform.apply(y_train, X=X_train, model=model)
                y_test = transform.apply(y_test, X=X_test, model=model)
            else:
                y_train = transform.apply(y_train)
                y_test = transform.apply(y_test)

            # Track transformation statistics (for both train and test)
            if collect_statistics:
                statistics["transformations_applied"].append(transform_name)
                if y_train_before is not None and y_test_before is not None:
                    transform_stats = {
                        "transformation_name": transform_name,
                        "train": _compute_label_changes(y_train_before, y_train),
                        "test": _compute_label_changes(y_test_before, y_test),
                    }
                    statistics["per_transformation_stats"].append(transform_stats)

        if collect_statistics:
            statistics["transformed"] = {
                "train": _compute_label_distribution(y_train),
                "test": _compute_label_distribution(y_test),
            }
            # Overall change summary
            statistics["overall_changes"] = {
                "train": _compute_label_changes(y_train_original, y_train),
                "test": _compute_label_changes(y_test_original, y_test),
            }

        return y_train, y_test, statistics
