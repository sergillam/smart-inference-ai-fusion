"""Prepare real WiDS and IEEE-CIS datasets for STTT runs.

Expected input files:
- datasets/wids/train.csv
- datasets/ieee_fraud/train_transaction.csv
- datasets/ieee_fraud/train_identity.csv (optional but recommended)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"
OUT_DIR = ROOT / "data"

WIDS_IN = DATASETS_DIR / "wids" / "train.csv"
IEEE_TX_IN = DATASETS_DIR / "ieee_fraud" / "train_transaction.csv"
IEEE_ID_IN = DATASETS_DIR / "ieee_fraud" / "train_identity.csv"

WIDS_OUT = OUT_DIR / "wids_preprocessed.parquet"
IEEE_OUT = OUT_DIR / "ieee_cis_preprocessed.parquet"
SHA_OUT = OUT_DIR / "SHA256SUMS.txt"


def _coerce_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype("category").cat.codes
    return out


def _select_top_k_features(df: pd.DataFrame, target: str, k: int = 50, seed: int = 42) -> pd.DataFrame:
    y = df[target]
    x = df.drop(columns=[target])

    x = _coerce_categoricals(x)

    num_cols = x.columns.tolist()
    imputer = SimpleImputer(strategy="median")
    x_imp = pd.DataFrame(imputer.fit_transform(x), columns=num_cols)

    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(x_imp, y)

    importances = pd.Series(rf.feature_importances_, index=x_imp.columns)
    top_features = importances.sort_values(ascending=False).head(min(k, len(importances))).index.tolist()

    selected = x_imp[top_features].copy()
    selected[target] = y.values
    return selected


def _scale_0_1(df: pd.DataFrame, target: str) -> pd.DataFrame:
    y = df[target].copy()
    x = df.drop(columns=[target])
    scaler = MinMaxScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    x_scaled[target] = y.values
    return x_scaled


def prepare_wids() -> pd.DataFrame:
    if not WIDS_IN.exists():
        raise FileNotFoundError(f"Missing input file: {WIDS_IN}")

    df = pd.read_csv(WIDS_IN)
    target_col = None
    if "hospital_death" in df.columns:
        target_col = "hospital_death"
    elif "diabetes_mellitus" in df.columns:
        target_col = "diabetes_mellitus"
    else:
        raise ValueError(
            "WiDS file must contain target column 'hospital_death' (2020) "
            "or 'diabetes_mellitus' (2021)."
        )

    missing_frac = df.isna().mean()
    drop_cols = missing_frac[missing_frac > 0.40].index.tolist()
    drop_cols = [c for c in drop_cols if c != target_col]
    df = df.drop(columns=drop_cols)

    y = df[target_col].astype(int)
    x = df.drop(columns=[target_col])

    num_cols = x.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in x.columns if c not in num_cols]

    if num_cols:
        x[num_cols] = SimpleImputer(strategy="median").fit_transform(x[num_cols])
    if cat_cols:
        x[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(x[cat_cols])

    merged = x.copy()
    merged[target_col] = y.values

    selected = _select_top_k_features(merged, target=target_col, k=50)
    scaled = _scale_0_1(selected, target=target_col)
    return scaled


def prepare_ieee() -> pd.DataFrame:
    if not IEEE_TX_IN.exists():
        raise FileNotFoundError(f"Missing input file: {IEEE_TX_IN}")

    tx = pd.read_csv(IEEE_TX_IN)
    if "isFraud" not in tx.columns:
        raise ValueError("IEEE transaction file must contain target column 'isFraud'.")

    if IEEE_ID_IN.exists():
        identity = pd.read_csv(IEEE_ID_IN)
        if "TransactionID" in tx.columns and "TransactionID" in identity.columns:
            df = tx.merge(identity, on="TransactionID", how="left")
        else:
            df = tx
    else:
        df = tx

    y = df["isFraud"].astype(int)
    x = df.drop(columns=["isFraud"])

    x = _coerce_categoricals(x)
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=x.columns)

    merged = x.copy()
    merged["isFraud"] = y.values

    selected = _select_top_k_features(merged, target="isFraud", k=50)
    scaled = _scale_0_1(selected, target="isFraud")
    return scaled


def _write_sha(paths: list[Path]) -> None:
    lines: list[str] = []
    for p in paths:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{digest}  {p.name}")
    SHA_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wids = prepare_wids()
    ieee = prepare_ieee()

    # quick protocol sanity: stratified split is feasible
    train_test_split(
        wids.drop(columns=["hospital_death"]),
        wids["hospital_death"],
        test_size=0.3,
        stratify=wids["hospital_death"],
        random_state=42,
    )
    train_test_split(
        ieee.drop(columns=["isFraud"]),
        ieee["isFraud"],
        test_size=0.3,
        stratify=ieee["isFraud"],
        random_state=42,
    )

    wids.to_parquet(WIDS_OUT, index=False)
    ieee.to_parquet(IEEE_OUT, index=False)
    _write_sha([WIDS_OUT, IEEE_OUT])

    print(f"Saved: {WIDS_OUT}")
    print(f"Saved: {IEEE_OUT}")
    print(f"Saved: {SHA_OUT}")


if __name__ == "__main__":
    main()
