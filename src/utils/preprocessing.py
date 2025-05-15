import pandas as pd

def preprocess_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica pré-processamento específico à base Titanic:
    - Remove colunas irrelevantes
    - Trata valores nulos
    - Codifica colunas categóricas
    """
    # Remover colunas que não agregam valor preditivo direto
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    # Preencher valores nulos
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Codificar variáveis categóricas
    for col in ["Sex", "Embarked"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    return df
