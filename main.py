"""Command-line entry point for the churn pipeline.

    python main.py --input data/raw/Telco-Customer-Churn.csv
"""
from __future__ import annotations

import argparse
import sys

from src import config
from src.data_loader import download_dataset
from src.pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Telco churn CSV, train churn models and write predictions."
    )
    parser.add_argument("--input", default=str(config.RAW_DATA_PATH),
                        help="raw CSV file (default: data/raw/Telco-Customer-Churn.csv)")
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR),
                        help="folder for the report, metrics, predictions and charts (default: output/)")
    parser.add_argument("--download", action="store_true",
                        help="download the IBM CSV to --input first if the file is missing")
    parser.add_argument("--no-charts", action="store_true", help="skip writing the PNG charts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.download:
            download_dataset(args.input)
        run_pipeline(input_path=args.input, output_dir=args.output_dir, save_charts=not args.no_charts)
    except (OSError, ValueError) as error:  # missing file, failed download, invalid data
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
