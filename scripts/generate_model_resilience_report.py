#!/usr/bin/env python
"""Generate a model resilience report showing how each model reacts to each perturbation type.

Reads impact-analysis results (produced with --impact-analysis) from
case study JSON result files and generates:
  1. A per-model × per-perturbation degradation table (Markdown + optional LaTeX)
  2. A ranking of the most resilient models for each perturbation type
  3. A combined resilience score for model selection guidance

Usage:
  python scripts/generate_model_resilience_report.py                          # all cases
  python scripts/generate_model_resilience_report.py --cases case1 case2      # specific cases
  python scripts/generate_model_resilience_report.py --output results/model_resilience.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CASE_DIRS = [
    "case1",
    "case2",
    "case3",
    "case4",
    "case5_sip_sipv",
    "case6_sip_sipq",
    "case7_sipv_sipq",
    "case8_sip_sipv_sipq",
]

PERTURBATION_LABELS = {
    "data_perturbation": "Perturbação de Dados",
    "label_perturbation": "Perturbação de Labels",
    "param_perturbation": "Perturbação de Parâmetros",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _latest_file(base_dir: Path, pattern: str) -> Path | None:
    files = sorted(base_dir.glob(pattern))
    return files[-1] if files else None


def _load_records_from_case(results_root: Path, case: str) -> list[dict[str, Any]]:
    """Load experiment-level records from a case study's all_results JSON."""
    case_dir = results_root / case
    if not case_dir.is_dir():
        return []

    # Try various naming conventions
    candidates = [
        f"{case}_all_results_*.json",
        "case*_all_results_*.json",
    ]
    all_results_file: Path | None = None
    for pat in candidates:
        all_results_file = _latest_file(case_dir, pat)
        if all_results_file:
            break
    if not all_results_file:
        return []

    with open(all_results_file, encoding="utf-8") as fh:
        payload = json.load(fh)

    # Flatten: could be a plain list (case1/2/3) or a dict with nested records (case5-8)
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        # combo case format: look for sip_or_sipv -> records
        sip_records = payload.get("sip_or_sipv", {}).get("records", [])
        if sip_records:
            return [r for r in sip_records if isinstance(r, dict)]
        # fallback: if top-level list-valued keys exist
        for key in ("results", "all_results"):
            if isinstance(payload.get(key), list):
                return [r for r in payload[key] if isinstance(r, dict)]
    return []


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

# Keyed as (algorithm, dataset) -> perturbation_type -> list of accuracy_drop_pct
AggKey = tuple[str, str]
AggMap = dict[AggKey, dict[str, list[float]]]


def _extract_impact_rows(records: list[dict[str, Any]]) -> AggMap:
    """Extract per-model per-perturbation accuracy-drop lists from records."""
    agg: AggMap = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.get("status") != "success":
            continue
        impact = rec.get("impact_analysis")
        if not impact or not isinstance(impact, dict):
            continue
        algo = rec.get("algorithm", "?")
        dataset = rec.get("dataset", "?")
        key: AggKey = (algo, dataset)
        isolated = impact.get("isolated_impacts", {})
        for ptype in ("data_perturbation", "label_perturbation", "param_perturbation"):
            entry = isolated.get(ptype, {})
            drop_pct = entry.get("accuracy_drop_pct")
            if drop_pct is not None:
                agg[key][ptype].append(float(drop_pct))
    return agg


def _aggregate_by_model(agg: AggMap) -> dict[str, dict[str, list[float]]]:
    """Collapse dataset dimension: group all drops by algorithm only."""
    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (algo, _ds), ptypes in agg.items():
        for ptype, vals in ptypes.items():
            by_model[algo][ptype].extend(vals)
    return by_model


def _aggregate_by_dataset(agg: AggMap) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Group by dataset -> algorithm -> perturbation."""
    by_ds: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for (algo, ds), ptypes in agg.items():
        for ptype, vals in ptypes.items():
            by_ds[ds][algo][ptype].extend(vals)
    return by_ds


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# Report generation (Markdown)
# ---------------------------------------------------------------------------


def _fmt(val: float) -> str:
    return f"{val:.2f}"


def generate_report(
    agg: AggMap,
    *,
    cases_used: list[str],
) -> str:
    """Generate full Markdown resilience report."""
    by_model = _aggregate_by_model(agg)
    by_dataset = _aggregate_by_dataset(agg)
    ptypes = ["data_perturbation", "label_perturbation", "param_perturbation"]
    lines: list[str] = []

    lines.append("# Relatório de Resiliência dos Modelos às Perturbações")
    lines.append("")
    lines.append(f"> Gerado a partir dos casos: {', '.join(cases_used)}")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 1 – Global table (model × perturbation)
    # ------------------------------------------------------------------
    lines.append("## 1. Degradação Média por Modelo e Tipo de Perturbação")
    lines.append("")
    lines.append(
        "A tabela abaixo mostra, para cada modelo, a **queda percentual média de acurácia** "
        "provocada por cada tipo de perturbação isolada (média ± desvio sobre todos os datasets "
        "e seeds)."
    )
    lines.append("")
    lines.append(
        "| Modelo | Pert. Dados (%) | Pert. Labels (%) | Pert. Parâmetros (%) | "
        "Queda Combinada (%) |"
    )
    lines.append("|--------|----------------:|------------------:|---------------------:|---------------------:|")

    # Pre-compute combined drop for each model
    model_combined: dict[str, float] = {}
    for algo in sorted(by_model):
        drops = [_mean(by_model[algo].get(pt, [])) for pt in ptypes]
        combined = _mean([d for d in drops if d])
        model_combined[algo] = combined

    for algo in sorted(by_model):
        row_vals = []
        for pt in ptypes:
            vals = by_model[algo].get(pt, [])
            if vals:
                row_vals.append(f"{_fmt(_mean(vals))} ± {_fmt(_std(vals))}")
            else:
                row_vals.append("—")
        combined_str = _fmt(model_combined[algo])
        lines.append(f"| {algo} | {row_vals[0]} | {row_vals[1]} | {row_vals[2]} | {combined_str} |")

    lines.append("")

    # ------------------------------------------------------------------
    # Section 2 – Per-dataset breakdown
    # ------------------------------------------------------------------
    lines.append("## 2. Degradação por Dataset e Modelo")
    lines.append("")

    for ds in sorted(by_dataset):
        algo_map = by_dataset[ds]
        lines.append(f"### Dataset: {ds}")
        lines.append("")
        lines.append(
            "| Modelo | Pert. Dados (%) | Pert. Labels (%) | Pert. Parâmetros (%) |"
        )
        lines.append("|--------|----------------:|------------------:|---------------------:|")
        for algo in sorted(algo_map):
            row_vals = []
            for pt in ptypes:
                vals = algo_map[algo].get(pt, [])
                if vals:
                    row_vals.append(f"{_fmt(_mean(vals))} ± {_fmt(_std(vals))}")
                else:
                    row_vals.append("—")
            lines.append(f"| {algo} | {row_vals[0]} | {row_vals[1]} | {row_vals[2]} |")
        lines.append("")

    # ------------------------------------------------------------------
    # Section 3 – Rankings
    # ------------------------------------------------------------------
    lines.append("## 3. Ranking de Resiliência por Tipo de Perturbação")
    lines.append("")
    lines.append(
        "Modelos ordenados do **mais resiliente** (menor queda) ao **menos resiliente** "
        "(maior queda) para cada tipo de perturbação."
    )
    lines.append("")

    for pt in ptypes:
        label = PERTURBATION_LABELS[pt]
        ranked = sorted(
            ((algo, _mean(by_model[algo].get(pt, []))) for algo in by_model),
            key=lambda x: x[1],
        )
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Posição | Modelo | Queda Média (%) |")
        lines.append("|--------:|--------|----------------:|")
        for i, (algo, drop) in enumerate(ranked, 1):
            lines.append(f"| {i} | {algo} | {_fmt(drop)} |")
        lines.append("")

    # ------------------------------------------------------------------
    # Section 4 – Combined resilience score
    # ------------------------------------------------------------------
    lines.append("## 4. Pontuação de Resiliência Combinada")
    lines.append("")
    lines.append(
        "Indica qual modelo sofre **menos degradação global** considerando os três "
        "tipos de perturbação. Quanto menor a pontuação, mais resiliente o modelo."
    )
    lines.append("")
    lines.append("| Posição | Modelo | Pontuação (soma médias %) |")
    lines.append("|--------:|--------|-------------------------:|")

    scores: list[tuple[str, float]] = []
    for algo in by_model:
        total = sum(_mean(by_model[algo].get(pt, [])) for pt in ptypes)
        scores.append((algo, total))
    scores.sort(key=lambda x: x[1])

    for i, (algo, score) in enumerate(scores, 1):
        lines.append(f"| {i} | {algo} | {_fmt(score)} |")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 5 – Recommendations
    # ------------------------------------------------------------------
    lines.append("## 5. Recomendações para Seleção de Modelo")
    lines.append("")

    if scores:
        best = scores[0][0]
        worst = scores[-1][0]
        lines.append(f"- **Modelo mais resiliente globalmente**: `{best}`")
        lines.append(f"- **Modelo menos resiliente globalmente**: `{worst}`")
        lines.append("")

    for pt in ptypes:
        label = PERTURBATION_LABELS[pt]
        ranked = sorted(
            ((algo, _mean(by_model[algo].get(pt, []))) for algo in by_model),
            key=lambda x: x[1],
        )
        if ranked:
            lines.append(f"- Melhor modelo para **{label}**: `{ranked[0][0]}` (queda {_fmt(ranked[0][1])}%)")

    lines.append("")
    lines.append(
        "> **Nota**: Se o seu cenário apresenta predominância de ruído num tipo específico "
        "(ex.: dados ruidosos vs. labels incorretos), priorize o ranking desse tipo de perturbação "
        "em vez da pontuação combinada."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX output (optional)
# ---------------------------------------------------------------------------


def generate_latex_table(agg: AggMap) -> str:
    """Generate a LaTeX table with the global model × perturbation degradation."""
    by_model = _aggregate_by_model(agg)
    ptypes = ["data_perturbation", "label_perturbation", "param_perturbation"]

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Degradação percentual média de acurácia por modelo e tipo de perturbação}")
    lines.append(r"\label{tab:model_resilience}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Modelo} & \textbf{Pert. Dados (\%)} & "
        r"\textbf{Pert. Labels (\%)} & \textbf{Pert. Parâmetros (\%)} \\"
    )
    lines.append(r"\midrule")

    for algo in sorted(by_model):
        cells: list[str] = []
        for pt in ptypes:
            vals = by_model[algo].get(pt, [])
            if vals:
                cells.append(f"${_fmt(_mean(vals))} \\pm {_fmt(_std(vals))}$")
            else:
                cells.append("--")
        lines.append(f"{algo} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model resilience report from impact-analysis results.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root results directory (default: results/)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help=f"Case directories to include (default: all). Options: {CASE_DIRS}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output Markdown file (default: print to stdout)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also emit a LaTeX resilience table at the end.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_root = Path(args.results_dir)
    cases = args.cases or CASE_DIRS

    all_records: list[dict[str, Any]] = []
    cases_found: list[str] = []
    for case in cases:
        records = _load_records_from_case(results_root, case)
        if records:
            all_records.extend(records)
            cases_found.append(case)

    if not all_records:
        print(
            "Nenhum resultado com impact_analysis encontrado. "
            "Execute os case studies com --impact-analysis primeiro.",
            file=sys.stderr,
        )
        sys.exit(1)

    agg = _extract_impact_rows(all_records)
    if not agg:
        print(
            "Resultados encontrados, mas nenhum contém impact_analysis. "
            "Re-execute os scripts com --impact-analysis.",
            file=sys.stderr,
        )
        sys.exit(1)

    report = generate_report(agg, cases_used=cases_found)

    if args.latex:
        report += "\n---\n\n## Tabela LaTeX\n\n```latex\n"
        report += generate_latex_table(agg)
        report += "\n```\n"

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Relatório salvo em: {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
