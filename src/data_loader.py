"""Load the raw Telco CSV and run basic input checks."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from src import config


def download_dataset(destination: str | Path = config.RAW_DATA_PATH, url: str = config.DATASET_URL) -> Path:
    """Download the IBM CSV to `destination` unless the file is already there.

    An existing file is never overwritten, so an instructor-provided copy wins.
    """
    destination = Path(destination)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)  # only a complete download gets the real name
    return destination


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the CSV exactly as stored on disk.

    No type conversion happens here on purpose: the validator has to see the
    raw values (for example, TotalCharges arrives as text).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}. Put Telco-Customer-Churn.csv in "
            "data/raw/ (see README) or pass another file with --input."
        )
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.name}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"The input file has no data rows: {path}")
    return df


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: data type, filled cells, NaN cells and distinct values."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing_nan": df.isna().sum(),
        "unique_values": df.nunique(),
    })
