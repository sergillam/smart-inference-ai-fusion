#!/usr/bin/env python
"""Analysis script for SIP-Q Case 4 results."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

from scripts.results_io import load_json_records


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all Case 4 JSON result files into a normalized DataFrame."""
    records = load_json_records(results_dir, "case4_results_*.json")
    if not records:
        raise ValueError(f"No valid case4 results found in {results_dir}.")

    frame = pd.json_normalize(records)
    frame["metric_type"] = np.where(
        frame.get("quantized_accuracy").notna(), "supervised", "unsupervised"
    )
    frame["baseline_metric"] = np.where(
        frame["metric_type"] == "supervised",
        frame.get("baseline_accuracy"),
        frame.get("baseline_silhouette"),
    )
    frame["quantized_metric"] = np.where(
        frame["metric_type"] == "supervised",
        frame.get("quantized_accuracy"),
        frame.get("quantized_silhouette"),
    )
    frame["delta_abs"] = frame["quantized_metric"] - frame["baseline_metric"]
    frame["delta_rel_pct"] = np.where(
        frame["baseline_metric"] != 0,
        (frame["delta_abs"] / frame["baseline_metric"]) * 100.0,
        np.nan,
    )
    return frame


def bootstrap_ci(series: pd.Series, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the sample mean."""
    values = series.dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means.append(float(np.mean(sample)))
    lower = np.percentile(boot_means, (alpha / 2.0) * 100.0)
    upper = np.percentile(boot_means, (1.0 - alpha / 2.0) * 100.0)
    return (float(lower), float(upper))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction and return adjusted p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    prev = 0.0
    for rank, (idx, p_value) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p_value)
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return adjusted


def paired_significance_test(series: pd.Series) -> tuple[str, float]:
    """Run one-sample paired test on delta values against zero."""
    values = series.dropna().to_numpy(dtype=np.float64)
    if values.size < 3:
        return ("insufficient", float("nan"))

    if values.size >= 10 and not np.allclose(values, values[0]):
        stat = wilcoxon(values, zero_method="wilcox", correction=False, alternative="two-sided")
        return ("wilcoxon", float(stat.pvalue))

    stat = ttest_1samp(values, popmean=0.0, alternative="two-sided")
    return ("ttest_1samp", float(stat.pvalue))


def summarize_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aggregate table and significance table from raw records."""
    grouping = ["dataset_name", "algorithm_name", "metadata.mode", "bit_width", "metric_type"]
    summary = (
        df.groupby(grouping, dropna=False)
        .agg(
            n=("delta_abs", "count"),
            baseline_mean=("baseline_metric", "mean"),
            quantized_mean=("quantized_metric", "mean"),
            delta_abs_mean=("delta_abs", "mean"),
            delta_rel_pct_mean=("delta_rel_pct", "mean"),
            compression_ratio_mean=("compression_ratio", "mean"),
            quantization_mse_mean=("quantization_mse", "mean"),
        )
        .reset_index()
    )

    ci_bounds = summary.apply(
        lambda row: bootstrap_ci(
            df.loc[
                (df["dataset_name"] == row["dataset_name"])
                & (df["algorithm_name"] == row["algorithm_name"])
                & (df["metadata.mode"] == row["metadata.mode"])
                & (df["bit_width"] == row["bit_width"])
                & (df["metric_type"] == row["metric_type"]),
                "delta_abs",
            ]
        ),
        axis=1,
    )
    summary["delta_abs_ci95_low"] = [bound[0] for bound in ci_bounds]
    summary["delta_abs_ci95_high"] = [bound[1] for bound in ci_bounds]

    sig_records: list[dict[str, Any]] = []
    p_values: list[float] = []
    for _, row in summary.iterrows():
        mask = (
            (df["dataset_name"] == row["dataset_name"])
            & (df["algorithm_name"] == row["algorithm_name"])
            & (df["metadata.mode"] == row["metadata.mode"])
            & (df["bit_width"] == row["bit_width"])
            & (df["metric_type"] == row["metric_type"])
        )
        test_name, p_value = paired_significance_test(df.loc[mask, "delta_abs"])
        sig_records.append(
            {
                "dataset_name": row["dataset_name"],
                "algorithm_name": row["algorithm_name"],
                "mode": row["metadata.mode"],
                "bit_width": row["bit_width"],
                "metric_type": row["metric_type"],
                "test": test_name,
                "p_value": p_value,
            }
        )
        p_values.append(p_value if np.isfinite(p_value) else 1.0)

    adjusted = holm_bonferroni(p_values)
    for record, adj in zip(sig_records, adjusted):
        record["p_value_holm"] = adj
        record["significant"] = bool(adj < 0.05)
    significance = pd.DataFrame(sig_records)
    return summary, significance


def _load_plotting_modules() -> tuple[Any, Any]:
    """Load plotting dependencies lazily and fail with a clear message."""
    try:
        plt = importlib.import_module("matplotlib.pyplot")
        sns = importlib.import_module("seaborn")
        return plt, sns
    except ImportError as exc:
        raise RuntimeError(
            "Plotting dependencies are required. Install with: pip install '.[viz]'"
        ) from exc


def plot_accuracy_vs_bitwidth(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot quantized metric vs bit-width by algorithm."""
    plt, _ = _load_plotting_modules()

    plt.figure(figsize=(12, 6))
    grouped = df.groupby(["algorithm_name", "bit_width"], as_index=False)["quantized_metric"].mean()
    for algorithm in grouped["algorithm_name"].unique():
        part = grouped[grouped["algorithm_name"] == algorithm]
        plt.plot(part["bit_width"], part["quantized_metric"], marker="o", label=algorithm)
    plt.xlabel("Bit-width")
    plt.ylabel("Quantized score (accuracy or silhouette)")
    plt.title("Quantized score vs bit-width")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_bitwidth.png", dpi=300)
    plt.close()


def plot_pareto_frontier(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot Pareto frontier (compression ratio vs quantized score)."""
    plt, _ = _load_plotting_modules()

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df["compression_ratio"],
        df["quantized_metric"],
        c=df["bit_width"],
        cmap="viridis",
        alpha=0.7,
        s=40,
    )
    plt.colorbar(scatter, label="Bit-width")
    plt.xlabel("Compression ratio")
    plt.ylabel("Quantized score (accuracy/silhouette)")
    plt.title("Pareto frontier: compression vs quality")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_frontier.png", dpi=300)
    plt.close()


def plot_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    """Plot heatmap for mean absolute degradation by dataset and algorithm."""
    plt, sns = _load_plotting_modules()

    pivot = summary.pivot_table(
        values="delta_abs_mean",
        index="dataset_name",
        columns="algorithm_name",
        aggfunc="mean",
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn")
    plt.title("Mean absolute delta (quantized - baseline)")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_degradation.png", dpi=300)
    plt.close()


def plot_grouped_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    """Plot grouped bars comparing supervised vs unsupervised deltas."""
    plt, _ = _load_plotting_modules()

    grouped = (
        summary.groupby(["metric_type", "bit_width"], as_index=False)["delta_abs_mean"]
        .mean()
        .sort_values(["metric_type", "bit_width"])
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    bits = sorted(grouped["bit_width"].unique())
    x = np.arange(len(bits))

    for idx, metric_type in enumerate(["supervised", "unsupervised"]):
        vals = []
        part = grouped[grouped["metric_type"] == metric_type]
        for bit in bits:
            val = part.loc[part["bit_width"] == bit, "delta_abs_mean"]
            vals.append(float(val.iloc[0]) if not val.empty else np.nan)
        ax.bar(x + (idx - 0.5) * width, vals, width=width, label=metric_type)

    ax.set_xticks(x)
    ax.set_xticklabels([str(bit) for bit in bits])
    ax.set_xlabel("Bit-width")
    ax.set_ylabel("Mean absolute delta")
    ax.set_title("Degradation by paradigm and bit-width")
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "bars_by_paradigm.png", dpi=300)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SIP-Q Case 4 results")
    parser.add_argument("--results-dir", default="results/case4")
    parser.add_argument("--output-dir", default="results/case4/analysis")
    return parser.parse_args()


def main() -> None:
    """Analyze Case 4 results and write summary tables and plots."""
    args = _parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    summary, significance = summarize_results(df)

    summary.to_csv(output_dir / "summary.csv", index=False)
    significance.to_csv(output_dir / "significance.csv", index=False)

    plot_accuracy_vs_bitwidth(df, output_dir)
    plot_pareto_frontier(df, output_dir)
    plot_heatmap(summary, output_dir)
    plot_grouped_bars(summary, output_dir)

    print(f"Analyzed {len(df)} records. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
