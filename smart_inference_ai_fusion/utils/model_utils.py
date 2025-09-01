"""Utility functions for model implementations."""

from typing import Any, Dict, TypeVar

from numpy.typing import ArrayLike

from smart_inference_ai_fusion.utils.metrics import evaluate_classification

T = TypeVar("T")


def get_estimator_params(estimator: Any, deep: bool = True) -> dict:
    """Get parameters for an estimator.

    Args:
        estimator: The scikit-learn estimator.
        deep (bool): If True, will return the parameters for this estimator and
                     contained subobjects that are estimators.

    Returns:
        dict: Parameter names mapped to their values.
    """
    return estimator.get_params(deep=deep)


def set_estimator_params(estimator: T, **params: Any) -> T:
    """Set the parameters of an estimator.

    Args:
        estimator: The scikit-learn estimator.
        **params (Any): Estimator parameters.

    Returns:
        T: The estimator instance.
    """
    estimator.set_params(**params)
    return estimator


def evaluate_classification_model(
    model: Any, X_test: ArrayLike, y_test: ArrayLike, average: str = "macro"
) -> Dict[str, Any]:
    """Common evaluation logic for classification models.

    Args:
        model: The fitted model with a predict method.
        X_test (ArrayLike): Test features.
        y_test (ArrayLike): True labels.
        average (str): Averaging method for multiclass metrics.

    Returns:
        Dict[str, Any]: Classification metrics.
    """
    y_pred = model.predict(X_test)
    return evaluate_classification(y_test, y_pred, average=average)
