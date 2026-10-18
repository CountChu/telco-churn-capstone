"""Matplotlib charts for the EDA and the model evaluation.

`plot_*` functions draw on an Axes you pass in, so the notebook can show them.
`save_*` functions build their own Figure and write PNG files; they never open
a window, so they also work in a terminal or in CI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from src import analysis, config

# One hue for single-series bars; a fixed second hue whenever churned is compared with stayed.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQUENTIAL_BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"])

CHART_TITLES = {
    "Contract": "Churn rate by contract type",
    "InternetService": "Churn rate by internet service",
    "PaymentMethod": "Churn rate by payment method",
    "TechSupport": "Churn rate by tech support",
    "tenure_band": "Churn rate by tenure (months as a customer)",
}


def _tidy(ax, grid_axis: str) -> None:
    """Quiet axes: hairline grid behind the data, no top/right frame."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=SECONDARY_INK, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_churn_rate(table: pd.DataFrame, title: str, ax, overall_rate: float | None = None):
    """Horizontal bars for a `churn_rate_by()` table, first row on top."""
    rates = table["churn_rate_%"]
    bars = ax.barh([str(group) for group in table.index], rates, height=0.6, color=BLUE)
    ax.bar_label(bars, labels=[f"{rate:.1f}%" for rate in rates], padding=3, fontsize=9, color=INK)
    ax.invert_yaxis()
    if overall_rate is not None:
        ax.axvline(overall_rate, color=SECONDARY_INK, linewidth=1, label=f"All customers ({overall_rate:.1f}%)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, max(rates.max(), overall_rate or 0) * 1.2)
    ax.set_xlabel("Churn rate (%)", color=SECONDARY_INK, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    _tidy(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    return ax


def plot_monthly_charges(df: pd.DataFrame, ax):
    """MonthlyCharges distribution of customers who stayed vs churned (% of each group)."""
    bins = np.linspace(df["MonthlyCharges"].min(), df["MonthlyCharges"].max(), 25)
    for value, label, color in ((0, "Stayed", BLUE), (1, "Churned", ORANGE)):
        charges = df.loc[df[config.TARGET_COLUMN] == value, "MonthlyCharges"]
        weights = np.full(len(charges), 100 / max(len(charges), 1))  # each group adds up to 100%
        ax.hist(charges, bins=bins, weights=weights, histtype="step", linewidth=2, color=color,
                label=f"{label} (mean {charges.mean():.1f})")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("Monthly charges", color=SECONDARY_INK, fontsize=9)
    ax.set_ylabel("% of the group", color=SECONDARY_INK, fontsize=9)
    ax.set_title("Monthly charges: stayed vs churned", loc="left", fontsize=11, color=INK)
    _tidy(ax, grid_axis="y")
    return ax


def plot_confusion_matrix(matrix: pd.DataFrame, ax, title: str = "Confusion matrix (test set)"):
    """Counts of a 2 x 2 `confusion_matrix_table()`; darker cell = more customers."""
    values = matrix.to_numpy()
    ax.imshow(values, cmap=SEQUENTIAL_BLUES)
    middle = (values.min() + values.max()) / 2
    for (row, col), count in np.ndenumerate(values):
        ax.text(col, row, f"{count:,}", ha="center", va="center", fontsize=13,
                color="white" if count > middle else INK)
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.tick_params(length=0, colors=SECONDARY_INK, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    return ax


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    return path


def save_eda_charts(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Write one PNG per EDA question and return the file paths."""
    output_dir = Path(output_dir)
    overall = analysis.overall_churn_rate(df)
    paths = []
    for name, table in analysis.segment_tables(df).items():
        fig = Figure(figsize=(7, 3.2), layout="constrained", facecolor=SURFACE)
        plot_churn_rate(table, CHART_TITLES[name], fig.add_subplot(), overall)
        paths.append(_save(fig, output_dir / f"churn_by_{name.lower()}.png"))
    fig = Figure(figsize=(7, 3.6), layout="constrained", facecolor=SURFACE)
    plot_monthly_charges(df, fig.add_subplot())
    paths.append(_save(fig, output_dir / "monthly_charges_by_churn.png"))
    return paths


def save_confusion_matrix(matrix: pd.DataFrame, path: str | Path) -> Path:
    """Write the confusion-matrix chart as a PNG file."""
    fig = Figure(figsize=(5.5, 4), layout="constrained", facecolor=SURFACE)
    plot_confusion_matrix(matrix, fig.add_subplot())
    return _save(fig, path)
