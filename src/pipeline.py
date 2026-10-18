"""Run the whole workflow end to end.

CSV -> load -> validate -> clean -> explore -> split -> train -> evaluate -> save outputs
"""
from __future__ import annotations

from pathlib import Path

from src import analysis, charts, config, data_loader, evaluation, model, preprocessing, validator


def _relative(path: str | Path) -> str:
    """Show a path relative to the current folder when possible (shorter, no user name)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _step(number: int, title: str) -> None:
    print(f"\n[{number}] {title}...")


def run_pipeline(
    input_path: str | Path = config.RAW_DATA_PATH,
    output_dir: str | Path = config.OUTPUT_DIR,
    processed_path: str | Path = config.PROCESSED_DATA_PATH,
    save_charts: bool = True,
) -> dict:
    """Run every stage, print a readable log and write the output files.

    Returns the main intermediate objects so a notebook or a test can inspect them.
    """
    output_dir = Path(output_dir)
    processed_path = Path(processed_path)
    print("=" * 44)
    print("TELCO CUSTOMER CHURN ANALYSIS")
    print("=" * 44)

    _step(1, "Loading data")
    raw_df = data_loader.load_raw_data(input_path)
    print(f"Rows loaded: {len(raw_df)} | Columns: {raw_df.shape[1]} | File: {_relative(input_path)}")

    _step(2, "Validating data")
    report = validator.validate_raw_data(raw_df)
    print(report.to_text())
    print(f"Saved: {_relative(validator.save_report(report, output_dir / 'validation_report.txt'))}")
    if report.missing_columns:
        raise ValueError(f"Cannot continue - required columns are missing: {report.missing_columns}")

    _step(3, "Cleaning data")
    clean_df, notes = preprocessing.clean_data(raw_df)
    for note in notes:
        print(f"- {note}")
    problems = validator.check_clean_data(clean_df)
    if problems:
        raise ValueError("Cleaned data failed its checks: " + "; ".join(problems))
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(processed_path, index=False)
    print(f"Cleaned data checks: PASS ({len(clean_df)} rows) | Saved: {_relative(processed_path)}")

    _step(4, "Exploring data")
    for finding in analysis.describe_findings(clean_df):
        print(f"- {finding}")
    if save_charts:
        chart_paths = charts.save_eda_charts(clean_df, output_dir / "figures")
        print(f"Saved {len(chart_paths)} charts to: {_relative(output_dir / 'figures')}")

    _step(5, "Training models")
    X, y, ids = preprocessing.split_features_target(clean_df)
    X_train, X_test, y_train, y_test, _, ids_test = preprocessing.split_train_test(X, y, ids)
    print(f"Features: {X.shape[1]} columns (customerID and Churn are not features)")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} "
          f"(test_size={config.TEST_SIZE}, random_state={config.RANDOM_STATE}, stratified)")
    models = {name: model.train_model(X_train, y_train, name) for name in model.MODEL_NAMES}
    print(f"Models trained: {', '.join(models)}")

    _step(6, "Evaluating models")
    outputs = {name: model.predict(fitted, X_test) for name, fitted in models.items()}
    results = {name: evaluation.calculate_metrics(y_test, predicted) for name, (predicted, _) in outputs.items()}
    primary = results[config.PRIMARY_MODEL]
    print(f"Primary model: {config.PRIMARY_MODEL}")
    print(evaluation.format_metrics(primary))
    print("Confusion matrix (test set):")
    print(evaluation.confusion_matrix_table(primary).to_string())
    print("Model comparison (test set):")
    print(evaluation.compare_models(results).to_string())
    if save_charts:
        charts.save_confusion_matrix(evaluation.confusion_matrix_table(primary),
                                     output_dir / "figures" / "confusion_matrix.png")

    _step(7, "Writing predictions")
    predicted, probability = outputs[config.PRIMARY_MODEL]
    predictions = evaluation.build_predictions_frame(ids_test, y_test, predicted, probability)
    print(f"Saved: {_relative(evaluation.save_predictions(predictions, output_dir / 'predictions.csv'))}")
    summary = {
        "primary_model": config.PRIMARY_MODEL,
        "split": {"test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE, "stratified": True,
                  "train_rows": len(X_train), "test_rows": len(X_test)},
        "models": results,
    }
    print(f"Saved: {_relative(evaluation.save_metrics(summary, output_dir / 'metrics.json'))}")

    print("\nPipeline completed successfully.")
    return {"report": report, "clean_df": clean_df, "models": models, "metrics": results, "predictions": predictions}
