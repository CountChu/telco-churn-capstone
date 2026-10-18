import pandas as pd
import pytest

from src import config
from src.data_loader import load_raw_data
from src.validator import find_duplicate_ids, find_invalid_values, find_missing_columns


def test_load_raw_data_reads_every_row_and_column(tmp_path, raw_sample):
    csv_path = tmp_path / "customers.csv"
    raw_sample.to_csv(csv_path, index=False)

    loaded = load_raw_data(csv_path)

    assert loaded.shape == raw_sample.shape
    assert list(loaded.columns) == list(raw_sample.columns)
    # loaded as stored: the blank cells keep TotalCharges a text column
    assert not pd.api.types.is_numeric_dtype(loaded["TotalCharges"])


def test_load_raw_data_gives_a_clear_error_for_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        load_raw_data(tmp_path / "missing.csv")


@pytest.mark.skipif(not config.RAW_DATA_PATH.exists(), reason="real CSV not downloaded")
def test_real_dataset_matches_the_data_dictionary():
    df = load_raw_data(config.RAW_DATA_PATH)

    assert find_missing_columns(df) == []
    assert find_duplicate_ids(df) == []
    assert find_invalid_values(df["Churn"], config.TARGET_LABELS) == {}
