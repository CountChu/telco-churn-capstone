"""Metrics, the confusion matrix and the output files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src import config

PREDICTION_COLUMNS = ["customerID", "actual_churn", "predicted_churn", "churn_probability", "risk_level"]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]


def calculate_metrics(y_true, y_pred) -> dict:
    """Accuracy, precision, recall and F1 (churn = positive class) plus the confusion counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def confusion_matrix_table(metrics: dict) -> pd.DataFrame:
    """2 x 2 table: rows = what really happened, columns = what the model predicted."""
    return pd.DataFrame(
        [[metrics["true_negatives"], metrics["false_positives"]],
         [metrics["false_negatives"], metrics["true_positives"]]],
        index=["Actual: stayed", "Actual: churned"],
        columns=["Predicted: stay", "Predicted: churn"],
    )


def format_metrics(metrics: dict) -> str:
    """Aligned metric lines for the console."""
    return "\n".join([
        f"Accuracy : {metrics['accuracy']:.3f}",
        f"Precision: {metrics['precision']:.3f}",
        f"Recall   : {metrics['recall']:.3f}",
        f"F1 Score : {metrics['f1']:.3f}",
    ])


def explain_metrics(metrics: dict) -> list[str]:
    """The metrics in business language (churn = the case we want to find)."""
    tp, fp = metrics["true_positives"], metrics["false_positives"]
    fn, tn = metrics["false_negatives"], metrics["true_negatives"]
    return [
        f"Accuracy {metrics['accuracy']:.1%}: {tp + tn:,} of the {tp + fp + fn + tn:,} test customers "
        "are labelled correctly.",
        f"Precision {metrics['precision']:.1%}: of the {tp + fp:,} customers flagged as likely to churn, "
        f"{tp:,} really churned.",
        f"Recall {metrics['recall']:.1%}: of the {tp + fn:,} customers who really churned, "
        f"the model caught {tp:,}.",
        f"False negatives: {fn:,} customers churned although the model said they would stay - "
        "a retention team would never have contacted them.",
        f"False positives: {fp:,} customers who stayed were flagged - the cost is an unneeded retention offer.",
    ]


def compare_models(results: dict) -> pd.DataFrame:
    """One row per model with the four headline metrics."""
    return pd.DataFrame(results).T[METRIC_NAMES].astype(float).round(3)


def build_predictions_frame(customer_ids, y_true, y_pred, churn_probability) -> pd.DataFrame:
    """The predictions table, highest churn risk first."""
    predictions = pd.DataFrame({
        "customerID": np.asarray(customer_ids),
        "actual_churn": np.asarray(y_true, dtype=int),
        "predicted_churn": np.asarray(y_pred, dtype=int),
        "churn_probability": np.round(np.asarray(churn_probability, dtype=float), 4),
    })
    predictions["risk_level"] = pd.cut(
        predictions["churn_probability"], bins=config.RISK_BINS,
        labels=config.RISK_LABELS, include_lowest=True,
    ).astype(str)
    return predictions.sort_values("churn_probability", ascending=False, ignore_index=True)


def risk_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Customers and actual churn rate per risk band - does the risk score rank customers well?"""
    summary = predictions.groupby("risk_level")["actual_churn"].agg(["size", "mean"])
    summary.columns = ["customers", "actual_churn_%"]
    summary["actual_churn_%"] = (summary["actual_churn_%"] * 100).round(1)
    return summary.loc[[label for label in config.RISK_LABELS if label in summary.index]]


def save_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    """Write the predictions table as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return path


def save_metrics(summary: dict, path: str | Path) -> Path:
    """Write the metrics (and the settings that produced them) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
