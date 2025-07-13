"""Preprocess the dataset for machine learning tasks."""
import pandas as pd

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
