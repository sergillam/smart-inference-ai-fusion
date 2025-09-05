"""Preprocess the dataset for machine learning tasks."""

import pandas as pd

from smart_inference_ai_fusion.utils.report import report_data
from smart_inference_ai_fusion.utils.types import ReportMode


def filter_sklearn_params(params, model_class):
    """Filter a parameter dictionary for keys accepted by a scikit-learn model.

    Args:
        params (dict): Dictionary of parameters (may include extra keys).
        model_class (type): scikit-learn model class (e.g., RandomForestClassifier).

    Returns:
        dict: Only the parameters valid for the model.
    """
    valid_keys = model_class().get_params().keys()
    return {k: v for k, v in params.items() if k in valid_keys}


def _validate_samples_param(
    params: dict,
    param_name: str,
    *,
    int_min: int,
    float_min: float,
    float_max: float,
    default_value,
    model_prefix: str = "",
) -> dict:
    """Helper function to validate min_samples_split and min_samples_leaf parameters.

    Args:
        params (dict): Parameters dictionary to modify.
        param_name (str): Name of the parameter to validate.
        int_min (int): Minimum value for integer type.
        float_min (float): Minimum value for float type.
        float_max (float): Maximum value for float type.
        default_value: Default value to use if invalid.
        model_prefix (str): Prefix for warning messages.

    Returns:
        dict: Modified parameters dictionary.
    """
    if param_name in params:
        value = params[param_name]
        if isinstance(value, int) and value < int_min:
            params[param_name] = default_value
            report_data(
                f"WARNING: Corrected {model_prefix}{param_name} from {value} to {default_value}",
                mode=ReportMode.PRINT,
            )
        elif isinstance(value, float) and (value <= float_min or value > float_max):
            params[param_name] = default_value
            report_data(
                f"WARNING: Corrected {model_prefix}{param_name} from {value} to {default_value}",
                mode=ReportMode.PRINT,
            )
    return params


def _validate_gradient_boosting_params(params: dict) -> dict:
    """Validate and fix GradientBoosting specific parameters.

    Args:
        params (dict): Parameters to validate.

    Returns:
        dict: Corrected parameters.
    """
    corrected_params = params.copy()

    # Validate min_samples_split: >= 2 for int, or (0.0, 1.0] for float
    corrected_params = _validate_samples_param(
        corrected_params,
        "min_samples_split",
        int_min=2,
        float_min=0.0,
        float_max=1.0,
        default_value=2,
        model_prefix="GradientBoosting ",
    )

    # Validate min_samples_leaf: >= 1 for int, or (0.0, 0.5] for float
    corrected_params = _validate_samples_param(
        corrected_params,
        "min_samples_leaf",
        int_min=1,
        float_min=0.0,
        float_max=0.5,
        default_value=1,
        model_prefix="GradientBoosting ",
    )

    # subsample must be in (0.0, 1.0]
    if "subsample" in corrected_params:
        value = corrected_params["subsample"]
        if not isinstance(value, (int, float)) or value <= 0.0 or value > 1.0:
            corrected_params["subsample"] = 0.8
            report_data(
                f"WARNING: Corrected GradientBoosting subsample from {value} to 0.8",
                mode=ReportMode.PRINT,
            )

    # learning_rate must be > 0.0
    if "learning_rate" in corrected_params:
        value = corrected_params["learning_rate"]
        if not isinstance(value, (int, float)) or value <= 0.0:
            corrected_params["learning_rate"] = 0.1
            report_data(
                f"WARNING: Corrected GradientBoosting learning_rate from {value} to 0.1",
                mode=ReportMode.PRINT,
            )

    return corrected_params


def _validate_random_forest_params(params: dict) -> dict:
    """Validate and fix RandomForest specific parameters.

    Args:
        params (dict): Parameters to validate.

    Returns:
        dict: Corrected parameters.
    """
    corrected_params = params.copy()

    # Validate min_samples_split: >= 2 for int, or (0.0, 1.0] for float
    corrected_params = _validate_samples_param(
        corrected_params,
        "min_samples_split",
        int_min=2,
        float_min=0.0,
        float_max=1.0,
        default_value=2,
    )
    # Validate min_samples_leaf: >= 1 for int, or (0.0, 0.5] for float
    corrected_params = _validate_samples_param(
        corrected_params,
        "min_samples_leaf",
        int_min=1,
        float_min=0.0,
        float_max=0.5,
        default_value=1,
    )

    return corrected_params


def validate_sklearn_params(params, model_class):
    """Validate and fix parameter values for scikit-learn models.

    Args:
        params (dict): Dictionary of parameters to validate.
        model_class (type): scikit-learn model class.

    Returns:
        dict: Parameters with invalid values corrected.
    """
    corrected_params = params.copy()

    # Apply model-specific validations
    if "GradientBoosting" in model_class.__name__:
        corrected_params = _validate_gradient_boosting_params(corrected_params)
    elif "RandomForest" in model_class.__name__:
        corrected_params = _validate_random_forest_params(corrected_params)

    return corrected_params


def preprocess_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the Titanic dataset DataFrame for machine learning.

    This includes:
        - Removing irrelevant columns
        - Imputing missing values for key features
        - Encoding categorical variables as numeric codes

    Args:
        df (pd.DataFrame): Raw Titanic DataFrame.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame ready for modeling.
    """
    # Drop columns unlikely to add predictive value
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    # Fill missing values for numerical columns
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Fill missing values for categorical columns
    if "Embarked" in df.columns:
        most_frequent = df["Embarked"].mode()
        if not most_frequent.empty:
            df["Embarked"] = df["Embarked"].fillna(most_frequent[0])

    # Encode categorical columns as integer codes
    for col in ["Sex", "Embarked"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    return df
