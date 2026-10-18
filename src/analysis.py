"""Exploratory analysis of the cleaned data (Churn already encoded as 1/0).

These numbers describe associations in this sample. They do not show that a
column causes churn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

SEGMENT_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
TENURE_BINS = [-1, 12, 24, 48, np.inf]  # np.inf: any longer tenure lands in the last band
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]


def overall_churn_rate(df: pd.DataFrame) -> float:
    """Share of customers who churned, in percent."""
    return float(df[config.TARGET_COLUMN].mean() * 100)


def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of `df` with a `tenure_band` column (0-12, 13-24, 25-48, 49+ months)."""
    banded = df.copy()
    banded["tenure_band"] = pd.cut(banded["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    return banded


def churn_rate_by(df: pd.DataFrame, column: str, sort: bool = True) -> pd.DataFrame:
    """Customers, churned customers and churn rate (%) for each value of `column`."""
    grouped = df.groupby(column, observed=True)[config.TARGET_COLUMN]
    table = pd.DataFrame({
        "customers": grouped.size(),
        "churned": grouped.sum(),
        "churn_rate_%": (grouped.mean() * 100).round(1),
    })
    return table.sort_values("churn_rate_%", ascending=False) if sort else table


def monthly_charges_by_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Count, mean and median MonthlyCharges for customers who stayed vs churned."""
    table = df.groupby(config.TARGET_COLUMN)["MonthlyCharges"].agg(["count", "mean", "median"])
    return table.rename(index={0: "Stayed (0)", 1: "Churned (1)"}).round(2)


def segment_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Churn-rate tables for the EDA questions (tenure bands kept in time order)."""
    tables = {column: churn_rate_by(df, column) for column in SEGMENT_COLUMNS}
    tables["tenure_band"] = churn_rate_by(add_tenure_band(df), "tenure_band", sort=False)
    return tables


def describe_findings(df: pd.DataFrame) -> list[str]:
    """Observations written from the computed numbers - nothing is hard-coded."""
    churned = int(df[config.TARGET_COLUMN].sum())
    findings = [
        f"{overall_churn_rate(df):.1f}% of customers churned ({churned:,} of {len(df):,}), "
        "so the two classes are imbalanced."
    ]
    for column, table in segment_tables(df).items():
        rates = ", ".join(f"{group} {rate:.1f}%" for group, rate in table["churn_rate_%"].items())
        findings.append(f"Churn rate by {column}: {rates}.")
    charges = monthly_charges_by_churn(df)["mean"]
    higher_or_lower = "higher" if charges["Churned (1)"] > charges["Stayed (0)"] else "lower"
    findings.append(
        f"Customers who churned had {higher_or_lower} average MonthlyCharges "
        f"({charges['Churned (1)']:.2f} vs {charges['Stayed (0)']:.2f} for customers who stayed)."
    )
    return findings
