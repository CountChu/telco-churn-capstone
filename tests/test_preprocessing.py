import pandas as pd

from src.preprocessing import clean_data, split_features_target, split_train_test


def test_total_charges_is_numeric_after_cleaning(raw_sample):
    clean, _ = clean_data(raw_sample)

    assert pd.api.types.is_float_dtype(clean["TotalCharges"])
    assert clean["TotalCharges"].notna().all()
    # the blank values belonged to tenure-0 customers, so they become 0.0
    assert (clean.loc[clean["tenure"] == 0, "TotalCharges"] == 0).all()


def test_cleaned_data_has_no_negative_charges(raw_sample):
    clean, _ = clean_data(raw_sample)

    assert (clean[["MonthlyCharges", "TotalCharges"]] >= 0).all().all()


def test_churn_is_encoded_as_1_for_yes_and_0_for_no(raw_sample):
    clean, _ = clean_data(raw_sample)

    assert set(clean["Churn"].unique()) == {0, 1}
    assert clean["Churn"].sum() == (raw_sample["Churn"] == "Yes").sum()


def test_cleaning_never_modifies_the_raw_dataframe(raw_sample):
    before = raw_sample.copy(deep=True)

    clean_data(raw_sample)

    pd.testing.assert_frame_equal(raw_sample, before)


def test_duplicate_ids_and_invalid_labels_are_removed(raw_sample):
    dirty = pd.concat([raw_sample, raw_sample.iloc[[0]]], ignore_index=True)  # one repeated customer
    dirty.loc[1, "Churn"] = "Unknown"  # one row without a usable label

    clean, _ = clean_data(dirty)

    assert len(clean) == len(raw_sample) - 1
    assert clean["customerID"].is_unique


def test_customer_id_is_kept_out_of_the_features(raw_sample):
    clean, _ = clean_data(raw_sample)

    X, y, ids = split_features_target(clean)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert list(ids) == list(clean["customerID"])


def test_train_test_split_is_reproducible_and_stratified(raw_sample):
    clean, _ = clean_data(raw_sample)
    X, y, ids = split_features_target(clean)

    _, _, _, y_test, _, ids_test = split_train_test(X, y, ids)
    _, _, _, _, _, ids_test_again = split_train_test(X, y, ids)

    assert list(ids_test) == list(ids_test_again)  # same customers every run
    assert abs(y_test.mean() - y.mean()) < 0.05  # churn share kept in the test set
