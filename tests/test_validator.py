import pandas as pd

from src import config
from src.validator import (
    count_missing_values,
    find_duplicate_ids,
    find_invalid_values,
    find_missing_columns,
    find_unexpected_categories,
    validate_raw_data,
)


def test_required_columns_exist(raw_sample):
    assert find_missing_columns(raw_sample) == []


def test_a_missing_required_column_is_reported(raw_sample):
    report = validate_raw_data(raw_sample.drop(columns=["Contract"]))

    assert report.missing_columns == ["Contract"]
    assert report.status == "REVIEW"


def test_churn_contains_only_expected_labels_before_encoding(raw_sample):
    assert find_invalid_values(raw_sample["Churn"], config.TARGET_LABELS) == {}

    broken = raw_sample.copy()
    broken.loc[0, "Churn"] = "Maybe"
    assert find_invalid_values(broken["Churn"], config.TARGET_LABELS) == {"Maybe": 1}


def test_duplicate_customer_ids_are_detected(raw_sample):
    assert find_duplicate_ids(raw_sample) == []

    with_duplicate = pd.concat([raw_sample, raw_sample.iloc[[5]]], ignore_index=True)
    assert find_duplicate_ids(with_duplicate) == [raw_sample.loc[5, "customerID"]]


def test_blank_total_charges_are_reported_although_pandas_sees_no_nan(raw_sample):
    assert raw_sample["TotalCharges"].isna().sum() == 0  # the trap: " " is not NaN

    assert count_missing_values(raw_sample) == {"TotalCharges": 2}
    report = validate_raw_data(raw_sample)
    assert report.total_charges_issues == 2
    assert report.total_charges_issue_tenures == [0]


def test_negative_charges_and_impossible_tenure_are_detected(raw_sample):
    broken = raw_sample.copy()
    broken.loc[3, "MonthlyCharges"] = -10.0
    broken.loc[4, "tenure"] = 999

    report = validate_raw_data(broken)

    assert report.negative_values == {"MonthlyCharges": 1}
    assert report.out_of_range_tenure == 1


def test_unexpected_category_values_are_reported(raw_sample):
    broken = raw_sample.copy()
    broken.loc[0, "Contract"] = "Three year"

    assert find_unexpected_categories(broken) == {"Contract": {"Three year": 1}}


def test_report_passes_once_nothing_needs_fixing(raw_sample):
    fixed = raw_sample.copy()
    fixed.loc[:1, "TotalCharges"] = "0"

    report = validate_raw_data(fixed)

    assert report.issues == []
    assert report.status == "PASS"
