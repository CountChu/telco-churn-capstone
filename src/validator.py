"""Data-quality checks for the raw Telco churn data.

Each check is a small function that returns plain Python data, so it can be
tested on its own. `validate_raw_data` runs them all and collects the answers
in a `ValidationReport`, which prints as a plain-English summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config

SERVICE_COLUMNS = ["PhoneService", "MultipleLines", "InternetService"] + config.INTERNET_ADDON_COLUMNS


# ---------------------------------------------------------------- single checks
def find_missing_columns(df: pd.DataFrame, required: list[str] = config.REQUIRED_COLUMNS) -> list[str]:
    """Required columns that are not in the data."""
    return [column for column in required if column not in df.columns]


def find_duplicate_ids(df: pd.DataFrame, id_column: str = config.ID_COLUMN) -> list[str]:
    """Customer IDs that appear more than once (each listed once, sorted)."""
    is_duplicate = df[id_column].duplicated(keep=False)
    return sorted(set(df.loc[is_duplicate, id_column].astype(str)))


def count_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Missing values per column: real NaN cells plus blank or whitespace-only text.

    pandas turns an empty cell into NaN, but a cell holding a single space is
    read as the text " " - it looks filled in, yet carries no value.
    """
    counts = {}
    for column in df.columns:
        values = df[column]
        missing = int(values.isna().sum())
        if not pd.api.types.is_numeric_dtype(values):
            missing += int((values.dropna().astype(str).str.strip() == "").sum())
        if missing:
            counts[column] = missing
    return counts


def find_non_numeric_values(series: pd.Series) -> pd.Series:
    """Values that cannot be converted to a number (NaN cells are not included)."""
    as_number = pd.to_numeric(series, errors="coerce")
    return series[as_number.isna() & series.notna()]


def numeric_columns_stored_as_text(df: pd.DataFrame, columns: list[str] = config.NUMERIC_FEATURES) -> list[str]:
    """Numeric columns that pandas could not read as numbers."""
    return [
        column for column in columns
        if column in df.columns and not pd.api.types.is_numeric_dtype(df[column])
    ]


def count_negative_values(df: pd.DataFrame, columns: list[str] = config.NON_NEGATIVE_COLUMNS) -> dict[str, int]:
    """Values below zero per column (text is converted to numbers first)."""
    counts = {}
    for column in columns:
        if column in df.columns:
            negative = int((pd.to_numeric(df[column], errors="coerce") < 0).sum())
            if negative:
                counts[column] = negative
    return counts


def count_out_of_range(series: pd.Series, low: float, high: float) -> int:
    """Number of values outside the closed range [low, high]."""
    values = pd.to_numeric(series, errors="coerce")
    return int(((values < low) | (values > high)).sum())


def find_invalid_values(series: pd.Series, allowed: set) -> dict:
    """{unexpected value: count} for values not in `allowed` (NaN is ignored)."""
    counts = series.dropna().value_counts()
    return {_plain(value): int(n) for value, n in counts.items() if value not in allowed}


def find_unexpected_categories(df: pd.DataFrame, allowed_values: dict = config.ALLOWED_VALUES) -> dict[str, dict]:
    """Columns holding values outside the data dictionary, with those values."""
    problems = {}
    for column, allowed in allowed_values.items():
        if column in df.columns:
            invalid = find_invalid_values(df[column], allowed)
            if invalid:
                problems[column] = invalid
    return problems


def count_service_inconsistencies(df: pd.DataFrame) -> int:
    """Rows that break the service business rules.

    No internet service <-> every internet add-on says "No internet service".
    No phone service    <-> MultipleLines says "No phone service".
    """
    no_internet = df["InternetService"] == "No"
    addon_says_no_internet = df[config.INTERNET_ADDON_COLUMNS] == "No internet service"
    internet_mismatch = addon_says_no_internet.ne(no_internet, axis=0).any(axis=1)
    phone_mismatch = (df["MultipleLines"] == "No phone service") != (df["PhoneService"] == "No")
    return int((internet_mismatch | phone_mismatch).sum())


def _plain(value):
    """NumPy scalars (e.g. np.int64) -> plain Python values, for readable output."""
    return value.item() if hasattr(value, "item") else value


def _format_counts(counts: dict) -> str:
    return ", ".join(f"{key} ({value})" for key, value in counts.items())


# ---------------------------------------------------------------- the report
@dataclass
class ValidationReport:
    """Answers to the validation questions, with helpers to judge and print them."""

    n_rows: int
    n_columns: int
    dtypes: dict[str, str]
    missing_columns: list[str] = field(default_factory=list)
    text_numeric_columns: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    target_counts: dict = field(default_factory=dict)
    invalid_target_values: dict = field(default_factory=dict)
    total_charges_issues: int = 0
    total_charges_issue_tenures: list = field(default_factory=list)
    negative_values: dict[str, int] = field(default_factory=dict)
    out_of_range_tenure: int = 0
    unexpected_categories: dict[str, dict] = field(default_factory=dict)
    service_inconsistencies: int = 0

    @property
    def issues(self) -> list[str]:
        """Every problem found, in plain English (empty list = nothing to fix)."""
        low, high = config.TENURE_RANGE
        checks = [
            (self.missing_columns, f"Required columns missing: {', '.join(self.missing_columns)}"),
            (self.duplicate_ids, f"{len(self.duplicate_ids)} customer ID(s) appear more than once"),
            (self.missing_values, f"Missing or blank values in: {_format_counts(self.missing_values)}"),
            (self.invalid_target_values, f"Churn labels other than Yes/No: {self.invalid_target_values}"),
            (self.total_charges_issues,
             f"TotalCharges has {self.total_charges_issues} value(s) that are not numbers"),
            (self.negative_values, f"Negative values in: {_format_counts(self.negative_values)}"),
            (self.out_of_range_tenure, f"{self.out_of_range_tenure} tenure value(s) outside {low}-{high} months"),
            (self.unexpected_categories, f"Unexpected category values: {self.unexpected_categories}"),
            (self.service_inconsistencies,
             f"{self.service_inconsistencies} row(s) break the phone/internet service rules"),
        ]
        return [message for found, message in checks if found]

    @property
    def status(self) -> str:
        return "PASS" if not self.issues else "REVIEW"

    def to_text(self) -> str:
        """The report as plain text that a non-programmer can follow."""
        low, high = config.TENURE_RANGE
        type_counts = pd.Series(self.dtypes).value_counts().to_dict()
        total_charges = "none"
        if self.total_charges_issues:
            tenures = ", ".join(str(tenure) for tenure in self.total_charges_issue_tenures)
            total_charges = (f"{self.total_charges_issues} blank/non-numeric value(s); "
                             f"tenure of those customers: {tenures}")
        lines = [
            "DATA VALIDATION REPORT",
            "----------------------",
            f"Rows: {self.n_rows}",
            f"Columns: {self.n_columns}",
            f"Required columns missing: {', '.join(self.missing_columns) or 'none'}",
            f"Column types: {_format_counts(type_counts)}",
            f"Numeric columns stored as text: {', '.join(self.text_numeric_columns) or 'none'}",
            f"Duplicate customer IDs: {len(self.duplicate_ids)}",
            f"Missing-value fields: {_format_counts(self.missing_values) or 'none'}",
            f"Target labels (Churn): {_format_counts(self.target_counts) or 'n/a'}",
            f"Invalid target values: {self.invalid_target_values or 'none'}",
            f"TotalCharges conversion issues: {total_charges}",
            f"Negative numeric values: {_format_counts(self.negative_values) or 'none'}",
            f"Tenure outside {low}-{high} months: {self.out_of_range_tenure}",
            f"Unexpected categorical values: {self.unexpected_categories or 'none'}",
            f"Service consistency issues: {self.service_inconsistencies}",
            f"Overall status: {self.status}",
        ]
        if self.issues:
            lines.append("To handle before modelling:")
            lines += [f"  - {issue}" for issue in self.issues]
        return "\n".join(lines)


# ---------------------------------------------------------------- run everything
def validate_raw_data(df: pd.DataFrame) -> ValidationReport:
    """Run every check on the raw data. The DataFrame is only read, never changed."""
    columns = set(df.columns)
    report = ValidationReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        dtypes=df.dtypes.astype(str).to_dict(),
        missing_columns=find_missing_columns(df),
        text_numeric_columns=numeric_columns_stored_as_text(df),
        missing_values=count_missing_values(df),
        negative_values=count_negative_values(df),
        unexpected_categories=find_unexpected_categories(df),
    )
    if config.ID_COLUMN in columns:
        report.duplicate_ids = find_duplicate_ids(df)
    if config.TARGET_COLUMN in columns:
        target = df[config.TARGET_COLUMN]
        report.target_counts = {_plain(label): int(n) for label, n in target.value_counts().items()}
        report.invalid_target_values = find_invalid_values(target, config.TARGET_LABELS)
    if "TotalCharges" in columns:
        not_numeric = find_non_numeric_values(df["TotalCharges"])
        report.total_charges_issues = len(not_numeric)
        if "tenure" in columns:
            report.total_charges_issue_tenures = sorted(set(df.loc[not_numeric.index, "tenure"].tolist()))
    if "tenure" in columns:
        report.out_of_range_tenure = count_out_of_range(df["tenure"], *config.TENURE_RANGE)
    if set(SERVICE_COLUMNS) <= columns:
        report.service_inconsistencies = count_service_inconsistencies(df)
    return report


def check_clean_data(df: pd.DataFrame) -> list[str]:
    """Rules the cleaned table must meet before modelling (empty list = PASS)."""
    problems = []
    missing = count_missing_values(df)
    if missing:
        problems.append(f"missing values remain: {missing}")
    not_numeric = numeric_columns_stored_as_text(df)
    if not_numeric:
        problems.append(f"columns not numeric: {not_numeric}")
    negative = count_negative_values(df)
    if negative:
        problems.append(f"negative values: {negative}")
    if not set(df[config.TARGET_COLUMN].unique()) <= {0, 1}:
        problems.append("Churn is not encoded as 0/1")
    if find_duplicate_ids(df):
        problems.append("duplicate customer IDs remain")
    return problems


def save_report(report: ValidationReport, path: str | Path) -> Path:
    """Write the text report to `path` (folders are created when needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_text() + "\n", encoding="utf-8")
    return path
