"""Cleaning and feature preparation.

Cleaning decisions (each one is explained in the README):
1. Work on a copy - the raw file and the raw DataFrame are never changed.
2. TotalCharges is converted from text to float. Its blank values belong to
   brand-new customers (tenure = 0) who have not been billed yet, so a missing
   value is filled with MonthlyCharges x tenure - which is 0.0 for them.
3. Churn is encoded as Yes -> 1, No -> 0.
4. Rows with a repeated customerID (first one kept) or a Churn label other
   than Yes/No are removed. The IBM file has none, but a new CSV might.
5. customerID stays in the cleaned table for traceability, but is not a feature.
6. Scaling and one-hot encoding are part of the model pipeline
   (`build_preprocessor`), so they are fitted on the training rows only.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def convert_total_charges(series: pd.Series) -> pd.Series:
    """TotalCharges as float; blank or non-numeric text becomes NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def clean_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a cleaned copy of the raw data plus one plain-English note per step."""
    df = raw_df.copy()
    notes = []

    df["TotalCharges"] = convert_total_charges(df["TotalCharges"])
    blank = df["TotalCharges"].isna()
    filled_tenures = sorted(set(df.loc[blank, "tenure"].tolist()))
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    notes.append(
        f"TotalCharges converted from text to float; {int(blank.sum())} blank value(s) "
        f"filled with MonthlyCharges x tenure (tenure of those customers: {filled_tenures})"
    )

    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map(config.TARGET_MAPPING)
    duplicate = df[config.ID_COLUMN].duplicated(keep="first")
    bad_label = df[config.TARGET_COLUMN].isna() & ~duplicate
    df = df.loc[~duplicate & ~bad_label].reset_index(drop=True)
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(int)
    notes.append(f"Duplicate customer IDs removed (first row kept): {int(duplicate.sum())}")
    notes.append(f"Rows removed because Churn was not Yes/No: {int(bad_label.sum())}")
    notes.append("Churn encoded as 1 = Yes (churned), 0 = No (stayed)")
    return df, notes


def split_features_target(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """X = the feature columns, y = Churn (0/1), ids = customerID (for the output only).

    Selecting the feature list explicitly (instead of dropping columns) means
    customerID, Churn and any unexpected extra column can never become a feature.
    """
    X = clean_df[config.FEATURE_COLUMNS]
    y = clean_df[config.TARGET_COLUMN]
    ids = clean_df[config.ID_COLUMN]
    return X, y, ids


def split_train_test(X, y, ids, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """Reproducible, stratified split; customer IDs travel with their rows.

    Returns X_train, X_test, y_train, y_test, ids_train, ids_test.
    """
    return train_test_split(X, y, ids, test_size=test_size, random_state=random_state, stratify=y)


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns, pass SeniorCitizen (0/1) through, one-hot encode text columns.

    handle_unknown="ignore": a category never seen during training (for example
    in a new CSV) becomes all zeros instead of crashing the prediction.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), config.NUMERIC_FEATURES),
            ("binary", "passthrough", config.BINARY_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
