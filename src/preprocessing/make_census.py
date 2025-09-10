# src/preprocessing/make_census.py
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------
# Paths (auto-detect repo root)
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

RAW_TRAIN = DATA_RAW / "census_training.csv"
RAW_TEST  = DATA_RAW / "census_training_test.csv"

# ----------------------------
# Dataset schema (your columns)
# ----------------------------
BASE_CATEGORICAL = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
NUMERIC = ["age", "education_num", "hours_per_week"]
TARGET = "high_income"  # "<=50K" or ">50K"

# ----------------------------
# Utilities
# ----------------------------
def _require_files():
    missing = [p for p in [RAW_TRAIN, RAW_TEST] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {', '.join(str(m) for m in missing)}")

def _strip_all_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).str.strip()
    return out

def _clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace '?' / empty strings in categoricals with 'Unknown'
    """
    out = df.copy()
    for col in BASE_CATEGORICAL + [TARGET]:
        if col in out.columns and out[col].dtype == "object":
            out[col] = out[col].replace({"?": "Unknown", " ?": "Unknown", "": "Unknown"})
            out[col] = out[col].fillna("Unknown")
    return out

def _numeric_defaults() -> dict:
    """
    Sensible fallbacks if the TRAIN median is NaN (e.g., train is all-NaN for a column).
    """
    return {"age": 35.0, "education_num": 10.0, "hours_per_week": 40.0}

def _fit_numeric_medians(train: pd.DataFrame) -> dict:
    """
    Compute medians on TRAIN ONLY (after stripping/cleaning).
    If a column has no valid numeric values, use sensible defaults (no warnings).
    """
    meds = {}
    defaults = _numeric_defaults()
    for col in NUMERIC:
        s = pd.to_numeric(train[col], errors="coerce")
        if s.notna().any():               # <- avoid calling median on all-NaN
            m = s.median()                # pandas handles NaNs here
            meds[col] = float(m)
        else:
            meds[col] = float(defaults.get(col, 0.0))
    return meds

def _apply_numeric_medians(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """
    Coerce to numeric; fill NaNs using provided medians.
    """
    out = df.copy()
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(medians[col])
    return out

def _prep_common_train(df: pd.DataFrame):
    """
    TRAIN cleaning pipeline:
      - strip all strings
      - clean categoricals ('?' -> 'Unknown')
      - fit TRAIN medians for numerics
      - apply medians to TRAIN
    Returns: (train_clean, medians_dict)
    """
    df = _strip_all_strings(df)
    df = _clean_categoricals(df)
    medians = _fit_numeric_medians(df)
    df = _apply_numeric_medians(df, medians)
    return df, medians

def _prep_common_test(df: pd.DataFrame, medians: dict):
    """
    TEST cleaning pipeline:
      - strip all strings
      - clean categoricals ('?' -> 'Unknown')
      - apply TRAIN medians to TEST numerics
    """
    df = _strip_all_strings(df)
    df = _clean_categoricals(df)
    df = _apply_numeric_medians(df, medians)
    return df

# ----------------------------
# A) Categorical version (ID3)
# ----------------------------
def _bin_numeric_for_id3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Age bins
    out["age_bin"] = pd.cut(
        out["age"],
        bins=[-np.inf, 24, 34, 44, 54, 64, np.inf],
        labels=["<=24", "25-34", "35-44", "45-54", "55-64", "65+"]
    )

    # Education years bins
    out["education_years_bin"] = pd.cut(
        out["education_num"],
        bins=[-np.inf, 8, 12, 15, np.inf],
        labels=["<=8", "9-12", "13-15", ">=16"]
    )

    # Hours per week bins
    out["hours_bin"] = pd.cut(
        out["hours_per_week"],
        bins=[-np.inf, 29, 40, np.inf],
        labels=["part_time", "standard_40", "overtime"]
    )

    return out

def make_categorical_versions(train_clean: pd.DataFrame, test_clean: pd.DataFrame):
    """
    Inputs are already cleaned (strings stripped, categoricals normalized, numerics filled).
    Output: categorical-only files for from-scratch ID3.
    """
    train_b = _bin_numeric_for_id3(train_clean)
    test_b  = _bin_numeric_for_id3(test_clean)

    id3_cols = BASE_CATEGORICAL + ["age_bin", "education_years_bin", "hours_bin", TARGET]
    train_cat = train_b[id3_cols].copy()
    test_cat  = test_b[id3_cols].copy()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_cat.to_csv(DATA_PROCESSED / "census_train_categorical.csv", index=False)
    test_cat.to_csv(DATA_PROCESSED / "census_test_categorical.csv", index=False)

    return train_cat, test_cat

# ----------------------------
# B) One-hot + numeric (sklearn)
# ----------------------------
def make_ml_versions(train_clean: pd.DataFrame, test_clean: pd.DataFrame):
    """
    One-hot encode categoricals; keep numerics (already filled); align columns.
    """
    # One-hot on categoricals
    Xtr_cat = pd.get_dummies(train_clean[BASE_CATEGORICAL], drop_first=True)
    Xte_cat = pd.get_dummies(test_clean[BASE_CATEGORICAL], drop_first=True)

    # Align OHE columns
    Xte_cat = Xte_cat.reindex(columns=Xtr_cat.columns, fill_value=0)

    # Concatenate numeric features
    Xtr = pd.concat([train_clean[NUMERIC].reset_index(drop=True),
                     Xtr_cat.reset_index(drop=True)], axis=1)
    Xte = pd.concat([test_clean[NUMERIC].reset_index(drop=True),
                     Xte_cat.reset_index(drop=True)], axis=1)

    # Binarize target
    ytr = (train_clean[TARGET] == ">50K").astype(int)
    yte = (test_clean[TARGET]  == ">50K").astype(int)

    train_ml = pd.concat([Xtr, ytr.rename(TARGET)], axis=1)
    test_ml  = pd.concat([Xte, yte.rename(TARGET)], axis=1)

    train_ml.to_csv(DATA_PROCESSED / "census_train_ml.csv", index=False)
    test_ml.to_csv(DATA_PROCESSED / "census_test_ml.csv", index=False)

    # Save feature list for reference
    (DATA_PROCESSED / "ml_feature_columns.txt").write_text(
        "\n".join(train_ml.drop(columns=[TARGET]).columns), encoding="utf-8"
    )

    return train_ml, test_ml

# ----------------------------
# Main
# ----------------------------
def main():
    _require_files()

    train_raw = pd.read_csv(RAW_TRAIN)
    test_raw  = pd.read_csv(RAW_TEST)

    # Clean with train-only medians
    train_clean, medians = _prep_common_train(train_raw)
    test_clean = _prep_common_test(test_raw, medians)

    # Write all outputs
    train_cat, test_cat = make_categorical_versions(train_clean, test_clean)
    train_ml,  test_ml  = make_ml_versions(train_clean, test_clean)

    print("Wrote:")
    print(" - data/processed/census_train_categorical.csv")
    print(" - data/processed/census_test_categorical.csv")
    print(" - data/processed/census_train_ml.csv")
    print(" - data/processed/census_test_ml.csv")
    print(" - data/processed/ml_feature_columns.txt")
    print("\nShapes:")
    print(f" categorical: train={train_cat.shape}, test={test_cat.shape}")
    print(f" ml        : train={train_ml.shape},  test={test_ml.shape}")

if __name__ == "__main__":
    main()