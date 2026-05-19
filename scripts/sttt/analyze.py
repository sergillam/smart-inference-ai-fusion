"""Statistical analysis and LaTeX/PDF artifact generation for STTT runs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kruskal, shapiro, spearmanr, wilcoxon
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar

from scripts.sttt.telemetry import load_table

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze STTT outputs and generate LaTeX tables")
    parser.add_argument("--results-dir", default="results/sttt")
    parser.add_argument("--out-dir", default="results/sttt/analysis")
    return parser.parse_args()


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def _a12(x: list[float], y: list[float]) -> float | None:
    if not x or not y:
        return None
    wins = 0.0
    total = 0
    for xv in x:
        for yv in y:
            total += 1
            if xv > yv:
                wins += 1.0
            elif xv == yv:
                wins += 0.5
    return wins / total if total else None


def _a12_label(a12: float | None) -> str:
    if a12 is None:
        return "N/A"
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "N"
    if d < 0.14:
        return "S"
    if d < 0.21:
        return "M"
    return "L"


def _write_table(out_file: Path, caption: str, header: list[str], body: list[list[str]]) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{" + "l" * len(header) + "}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
    ]
    for row in body:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_dir(path: Path) -> list[dict[str, Any]]:
    return load_table(path)


def _extract_sipv_detection_by_type(payload: dict[str, Any]) -> dict[str, bool]:
    result = {"data": False, "label": False, "param": False}
    experiments = payload.get("experiments", {}) if isinstance(payload, dict) else {}
    mapping = {
        "data": "data_only",
        "label": "label_only",
        "param": "param_only",
    }
    for p_type, key in mapping.items():
        block = experiments.get(key, {})
        summary = block.get("verification_summary", {}) if isinstance(block, dict) else {}
        if not isinstance(summary, dict):
            continue
        for section in ("pre_perturbation", "post_perturbation", "model_integrity"):
            sec = summary.get(section, {})
            if not isinstance(sec, dict):
                continue
            for solver_payload in sec.values():
                if not isinstance(solver_payload, dict):
                    continue
                status = str(solver_payload.get("status", "")).upper()
                if status == "FAILURE":
                    result[p_type] = True
                    break
            if result[p_type]:
                break
    return result


def _analyze_rq1(results_dir: Path, out_dir: Path) -> None:
    sipv_rows = [r for r in _load_dir(results_dir / "sipv") if str(r.get("solver", "")).lower() == "z3"]
    pandera_rows = _load_dir(results_dir / "pandera")
    pandera_map = {row.get("run_id"): row for row in pandera_rows}

    body: list[list[str]] = []
    for p_type in ("data", "label", "param"):
        b = c = sipv_hits = pandera_hits = total = 0
        for row in sipv_rows:
            run_id = str(row.get("run_id", ""))
            p_row = pandera_map.get(run_id.replace(":sipv_z3", ":pandera"))
            if p_row is None:
                continue
            sipv_flag = _extract_sipv_detection_by_type(row.get("payload", {})).get(p_type, False)
            pandera_flag = bool((p_row.get("failures_by_type") or {}).get(p_type, 0))
            total += 1
            sipv_hits += int(sipv_flag)
            pandera_hits += int(pandera_flag)
            if sipv_flag and not pandera_flag:
                b += 1
            elif pandera_flag and not sipv_flag:
                c += 1

        p_value = None
        if b + c > 0:
            contingency = [[0, b], [c, 0]]
            p_value = float(mcnemar(contingency, exact=True).pvalue)

        body.append([
            p_type,
            str(total),
            _fmt(sipv_hits / total if total else None),
            _fmt(pandera_hits / total if total else None),
            str(b),
            str(c),
            _fmt(p_value),
        ])

    _write_table(
        out_dir / "table_rq1_detection.tex",
        "Detection rates by perturbation type — SIP-V vs Pandera",
        ["Type", "N", "SIP-V rate", "Pandera rate", "b", "c", "McNemar p"],
        body,
    )


def _analyze_rq2(results_dir: Path, out_dir: Path) -> None:
    sipv_rows = [r for r in _load_dir(results_dir / "sipv") if str(r.get("solver", "")).lower() == "z3"]
    pandera_rows = _load_dir(results_dir / "pandera")
    p_map = {row.get("run_id"): row for row in pandera_rows}

    sipv_times: list[float] = []
    pandera_times: list[float] = []
    for row in sipv_rows:
        run_id = str(row.get("run_id", ""))
        p_row = p_map.get(run_id.replace(":sipv_z3", ":pandera"))
        if not p_row:
            continue
        s_time = row.get("wall_clock_ms")
        p_time = p_row.get("wall_clock_ms")
        if isinstance(s_time, (int, float)) and isinstance(p_time, (int, float)):
            sipv_times.append(float(s_time))
            pandera_times.append(float(p_time))

    p_norm = p_wilcoxon = None
    a12 = None
    if sipv_times and len(sipv_times) == len(pandera_times):
        if len(sipv_times) >= 3:
            p_norm = float(shapiro(np.asarray(sipv_times) - np.asarray(pandera_times)).pvalue)
        if np.any((np.asarray(sipv_times) - np.asarray(pandera_times)) != 0):
            p_wilcoxon = float(wilcoxon(sipv_times, pandera_times, zero_method="wilcox").pvalue)
        a12 = _a12(sipv_times, pandera_times)

    _write_table(
        out_dir / "table_rq2_overhead.tex",
        "Overhead comparison — median times with p-values and A12",
        ["Pair", "N", "Median SIP-V (ms)", "Median Pandera (ms)", "Shapiro p", "Wilcoxon p", "A12"],
        [[
            "SIP-V vs Pandera",
            str(len(sipv_times)),
            _fmt(_median(sipv_times), 2),
            _fmt(_median(pandera_times), 2),
            _fmt(p_norm),
            _fmt(p_wilcoxon),
            f"{_fmt(a12)} ({_a12_label(a12)})",
        ]],
    )

    if plt is not None and sipv_times and pandera_times:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.boxplot([sipv_times, pandera_times], labels=["SIP-V", "Pandera"])
        ax.set_ylabel("wall_clock_ms")
        ax.set_title("RQ2 Overhead")
        fig.tight_layout()
        fig.savefig(out_dir / "figure_rq2_boxplot.pdf", format="pdf")
        plt.close(fig)


def _analyze_rq3(results_dir: Path, out_dir: Path) -> None:
    sipv_rows = _load_dir(results_dir / "sipv")
    models = sorted({str(r.get("model", "")) for r in sipv_rows})

    # Kruskal-Wallis on solve_time_ms by model
    groups: list[list[float]] = []
    for model in models:
        vals = [
            float(r.get("solve_time_ms"))
            for r in sipv_rows
            if str(r.get("model", "")) == model and isinstance(r.get("solve_time_ms"), (int, float))
        ]
        if vals:
            groups.append(vals)

    kruskal_p = None
    if len(groups) >= 2:
        kruskal_p = float(kruskal(*groups).pvalue)

    body: list[list[str]] = []
    all_constraints: list[float] = []
    all_solve: list[float] = []
    for model in models:
        rows = [r for r in sipv_rows if str(r.get("model", "")) == model]
        constraints = [float(r.get("num_constraints")) for r in rows if isinstance(r.get("num_constraints"), (int, float))]
        solve_times = [float(r.get("solve_time_ms")) for r in rows if isinstance(r.get("solve_time_ms"), (int, float))]
        all_constraints.extend(constraints)
        all_solve.extend(solve_times[: len(constraints)] if constraints else [])
        body.append([model, str(len(rows)), _fmt(_median(constraints), 1), _fmt(_median(solve_times), 2)])

    rho = rho_p = None
    if all_constraints and all_solve and len(all_constraints) == len(all_solve):
        corr = spearmanr(all_constraints, all_solve)
        rho = float(corr.correlation)
        rho_p = float(corr.pvalue)

    body.append(["kruskal_p", "-", "-", _fmt(kruskal_p)])
    body.append(["spearman", "-", _fmt(rho), _fmt(rho_p)])

    _write_table(
        out_dir / "table_rq3_tractability.tex",
        "Solver tractability by model type",
        ["Model", "N", "Median constraints", "Median solve time (ms)"],
        body,
    )

    # Sub-RQ3.1 Z3 vs CVC5
    z3_map = {r.get("run_id"): r for r in sipv_rows if str(r.get("solver", "")).lower() == "z3"}
    cvc5_rows = [r for r in sipv_rows if str(r.get("solver", "")).lower() == "cvc5"]
    verdict_pairs: list[tuple[str, str]] = []
    z3_t: list[float] = []
    cvc5_t: list[float] = []

    for c in cvc5_rows:
        c_id = str(c.get("run_id", ""))
        z_id = c_id.replace(":sipv_cvc5", ":sipv_z3")
        z = z3_map.get(z_id)
        if not z:
            continue
        z_status = str(z.get("status", "UNKNOWN"))
        c_status = str(c.get("status", "UNKNOWN"))
        verdict_pairs.append((z_status, c_status))
        if isinstance(z.get("solve_time_ms"), (int, float)) and isinstance(c.get("solve_time_ms"), (int, float)):
            z3_t.append(float(z["solve_time_ms"]))
            cvc5_t.append(float(c["solve_time_ms"]))

    agreement = None
    kappa = None
    p_w = None
    if verdict_pairs:
        z_labels = [z for z, _ in verdict_pairs]
        c_labels = [c for _, c in verdict_pairs]
        agreement = float(np.mean(np.asarray(z_labels) == np.asarray(c_labels)))
        kappa = float(cohen_kappa_score(z_labels, c_labels))
    if z3_t and len(z3_t) == len(cvc5_t) and np.any((np.asarray(z3_t) - np.asarray(cvc5_t)) != 0):
        p_w = float(wilcoxon(z3_t, cvc5_t, zero_method="wilcox").pvalue)

    _write_table(
        out_dir / "table_rq3_solver_agreement.tex",
        "Solver agreement and performance — Z3 vs CVC5",
        ["N", "Agreement", "Kappa", "Median Z3 solve (ms)", "Median CVC5 solve (ms)", "Wilcoxon p"],
        [[
            str(len(verdict_pairs)),
            _fmt(agreement),
            _fmt(kappa),
            _fmt(_median(z3_t), 2),
            _fmt(_median(cvc5_t), 2),
            _fmt(p_w),
        ]],
    )

    if plt is not None and all_constraints and all_solve and len(all_constraints) == len(all_solve):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        h = ax.hist2d(all_constraints, all_solve, bins=20)
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel("num_constraints")
        ax.set_ylabel("solve_time_ms")
        ax.set_title("RQ3 constraints vs solve time")
        fig.tight_layout()
        fig.savefig(out_dir / "figure_rq3_heatmap.pdf", format="pdf")
        plt.close(fig)


def _analyze_rq4(results_dir: Path, out_dir: Path) -> None:
    rows = _load_dir(results_dir / "sipq")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("model", "unknown"))
        grouped.setdefault(key, []).append(row)

    body: list[list[str]] = []
    for model, model_rows in sorted(grouped.items()):
        f1_deg = [float(r.get("f1_degradation")) for r in model_rows if isinstance(r.get("f1_degradation"), (int, float))]
        flips = [float(r.get("decision_flip_rate")) for r in model_rows if isinstance(r.get("decision_flip_rate"), (int, float))]
        wass = [float(r.get("wasserstein_distance")) for r in model_rows if isinstance(r.get("wasserstein_distance"), (int, float))]
        body.append([model, str(len(model_rows)), _fmt(_median(f1_deg)), _fmt(_median(flips)), _fmt(_median(wass))])

    if not body:
        body = [["N/A", "0", "N/A", "N/A", "N/A"]]

    _write_table(
        out_dir / "table_rq4_quantization.tex",
        "Quantization impact — int8 accuracy trade-off per model and dataset",
        ["Model", "N", "Median f1 degr.", "Median flip rate", "Median Wasserstein"],
        body,
    )


def _write_summary(out_dir: Path) -> None:
    summary = {
        "generated": [
            "table_rq1_detection.tex",
            "table_rq2_overhead.tex",
            "table_rq3_tractability.tex",
            "table_rq3_solver_agreement.tex",
            "table_rq4_quantization.tex",
            "figure_rq2_boxplot.pdf",
            "figure_rq3_heatmap.pdf",
        ]
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    _analyze_rq1(results_dir, out_dir)
    _analyze_rq2(results_dir, out_dir)
    _analyze_rq3(results_dir, out_dir)
    _analyze_rq4(results_dir, out_dir)
    _write_summary(out_dir)


if __name__ == "__main__":
    main()
