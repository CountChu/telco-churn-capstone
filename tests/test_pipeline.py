import hashlib

import pandas as pd

from src.charts import save_eda_charts
from src.evaluation import PREDICTION_COLUMNS
from src.pipeline import run_pipeline
from src.preprocessing import clean_data


def _fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_runs_end_to_end_and_leaves_the_raw_csv_untouched(tmp_path, raw_sample):
    raw_csv = tmp_path / "raw.csv"
    raw_sample.to_csv(raw_csv, index=False)
    fingerprint_before = _fingerprint(raw_csv)

    run_pipeline(input_path=raw_csv, output_dir=tmp_path / "output",
                 processed_path=tmp_path / "clean.csv", save_charts=False)

    predictions = pd.read_csv(tmp_path / "output" / "predictions.csv")
    assert list(predictions.columns) == PREDICTION_COLUMNS
    assert (tmp_path / "output" / "validation_report.txt").exists()
    assert (tmp_path / "output" / "metrics.json").exists()
    assert _fingerprint(raw_csv) == fingerprint_before


def test_eda_charts_are_saved_as_png_files(tmp_path, raw_sample):
    clean, _ = clean_data(raw_sample)

    paths = save_eda_charts(clean, tmp_path)

    assert len(paths) >= 3
    assert all(path.suffix == ".png" and path.stat().st_size > 0 for path in paths)
