"""Gaussian Mixture Model for the Smart Inference AI Fusion framework.

This module defines the GaussianMixtureModel class, a wrapper for scikit-learn's
GaussianMixture compatible with the BaseModel interface.
"""

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.logging import logger

# pylint: disable=duplicate-code


def _align_clusters_to_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Map predicted cluster ids to ground-truth labels using Hungarian assignment.

    Works with any hashable label dtype (ints/strings).

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted cluster ids.

    Returns:
        np.ndarray: y_pred remapped into y_true label space.
    """
    pred_vals = np.unique(y_pred)
    true_vals = np.unique(y_true)

    pred_index = {v: i for i, v in enumerate(pred_vals)}
    true_index = {v: i for i, v in enumerate(true_vals)}

    w = np.zeros((len(pred_vals), len(true_vals)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        w[pred_index[p], true_index[t]] += 1

    # maximize matches -> minimize (max - w)
    r, c = linear_sum_assignment(w.max() - w)
    mapping = {pred_vals[ri]: true_vals[ci] for ri, ci in zip(r, c)}

    return np.array([mapping[p] for p in y_pred], dtype=true_vals.dtype)


class GaussianMixtureModel(BaseModel):
    """Wrapper for scikit-learn's GaussianMixture, compatible with BaseModel."""

    def __init__(self, params: Optional[dict] = None, **kwargs: Any) -> None:
        """Initialize the GaussianMixtureModel.

        Args:
            params (dict | None): Parameters for the ``GaussianMixture`` constructor.
                If ``None``, an empty dict is used.
            **kwargs: Additional keyword arguments for model parameters
                (merged into ``params``).
        """
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = GaussianMixture(**params)

    def train(self, X_train: ArrayLike, y_train: Optional[ArrayLike] = None) -> None:
        """Fit FastICA on X_train and KMeans on the resulting components.

        Args:
            X_train (ArrayLike): The training features.
            y_train (Optional[ArrayLike]): Optional training labels, used to infer
                the number of clusters if not specified.
        """
        _ = y_train
        x_arr = np.asarray(X_train, dtype=np.float64)

        try:
            self.model.fit(x_arr)
            return
        except ValueError as exc:
            msg = str(exc)
            if "ill-defined empirical covariance" not in msg:
                raise

        # Retry progressivo de regularização
        base_reg = getattr(self.model, "reg_covar", 1e-6)
        for reg in (max(base_reg, 1e-5), 1e-4, 5e-4, 1e-3):
            try:
                self.model.set_params(reg_covar=reg)
                self.model.fit(x_arr)
                return
            except ValueError:
                continue

        # Último recurso: simplifica a estrutura de covariância
        if self.model.covariance_type != "diag":
            self.model.set_params(covariance_type="diag")
            self.model.fit(x_arr)  # se falhar aqui, deixa estourar

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Fit the model to data (for scikit-learn compatibility).

        Args:
            X (ArrayLike): The input data to fit the model on.
            y (Optional[ArrayLike]): Optional labels.
        """
        self.train(X, y)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for the input features.

        Args:
            X (ArrayLike): The input data for which to predict clusters.

        Returns:
            np.ndarray: The predicted cluster labels.
        """
        x_arr = np.asarray(X, dtype=np.float64)
        return self.model.predict(x_arr)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return posterior probabilities of each mixture component for each sample.

        Args:
            X (ArrayLike): The input data for which to predict probabilities.

        Returns:
            np.ndarray: The predicted cluster probabilities for each sample.
        """
        x_arr = np.asarray(X, dtype=np.float64)
        return self.model.predict_proba(x_arr)

    def evaluate(
        self,
        X_test: ArrayLike,
        y_test: Optional[ArrayLike] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on test data using clustering and supervised metrics.

        Args:
            X_test (ArrayLike): The test features.
            y_test (Optional[ArrayLike]): The ground truth labels for the test set.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        labels = self.predict(X_test)
        n_components = float(getattr(self.model, "n_components", 0))

        # Silhouette is defined only if there is >1 cluster and enough samples
        try:
            n_unique = len(np.unique(labels))
            x_arr = np.asarray(X_test, dtype=np.float64)
            sil = silhouette_score(x_arr, labels) if 1 < n_unique < len(x_arr) else None
        except ValueError:
            sil = None

        metrics: Dict[str, Any] = {
            "silhouette_score": sil,
            "n_components": n_components,
            "ari": None,
            "nmi": None,
            "accuracy": None,
            "balanced_accuracy": None,
            "f1": None,
            "accuracy_raw": None,
            "balanced_accuracy_raw": None,
            "f1_raw": None,
        }

        if y_test is not None:
            y_true = np.asarray(y_test)
            y_pred = np.asarray(labels)

            # Label-invariant clustering metrics
            try:
                metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
                metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
            except ValueError as exc:
                logger.warning("[GaussianMixtureModel] ARI/NMI failed: %s", exc)

            # Raw supervised metrics (no alignment)
            try:
                metrics["accuracy_raw"] = float(accuracy_score(y_true, y_pred))
                metrics["balanced_accuracy_raw"] = float(balanced_accuracy_score(y_true, y_pred))
                metrics["f1_raw"] = float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                )
            except ValueError:
                # Keep as None if classes mismatch makes metrics undefined
                pass

            # Aligned supervised metrics (Hungarian)
            try:
                y_pred_aligned = _align_clusters_to_labels(y_true, y_pred)
                metrics["accuracy"] = float(accuracy_score(y_true, y_pred_aligned))
                metrics["balanced_accuracy"] = float(
                    balanced_accuracy_score(y_true, y_pred_aligned)
                )
                metrics["f1"] = float(
                    f1_score(y_true, y_pred_aligned, average="macro", zero_division=0)
                )
            except ValueError as exc:
                logger.warning("[GaussianMixtureModel] Supervised metrics failed: %s", exc)

        return metrics

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                         contained subobjects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params: Any) -> "GaussianMixtureModel":
        """Set the parameters of this estimator.

        Args:
            **params (Any): Estimator parameters.

        Returns:
            GaussianMixtureModel: The estimator instance.
        """
        self.model.set_params(**params)
        return self
