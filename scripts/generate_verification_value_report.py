#!/usr/bin/env python
"""Generate a report on what formal verification (SIP-V) added to the pipeline.

Analyses case study results (especially case3, case5, case7, case8) and answers:
  1. What did the verifiers detect?  (constraints satisfied vs violated per phase)
  2. What is the computational cost? (overhead %)
  3. What should be flagged?          (violations, drift, solver disagreements)
  4. Which models/datasets need attention? (per-model violation rates)

Data sources
------------
- case3 results: dedicated SIP-V evaluation (none / Z3 / CVC5 modes)
- case5/7/8 results: combo experiments that include SIP-V records
- case1/2 impact-mode results: contain ``verification_summary`` when
  verification was globally enabled

Usage::

    python scripts/generate_verification_value_report.py
    python scripts/generate_verification_value_report.py --cases case3 case5_sip_sipv
    python scripts/generate_verification_value_report.py --output results/verification_value.md --latex
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
    "case1_test_impact",
    "case2",
    "case3",
    "case5_sip_sipv",
    "case7_sipv_sipq",
    "case8_sip_sipv_sipq",
]

VERIFICATION_PHASES = [
    "pre_perturbation",
    "post_perturbation",
    "model_integrity",
]

PHASE_LABELS = {
    "pre_perturbation": "Pré-Perturbação",
    "post_perturbation": "Pós-Perturbação",
    "model_integrity": "Integridade do Modelo",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _latest_file(base_dir: Path, pattern: str) -> Path | None:
    files = sorted(base_dir.glob(pattern))
    return files[-1] if files else None


def _load_records(results_root: Path, case: str) -> list[dict[str, Any]]:
    """Load flat experiment records from a case directory."""
    case_dir = results_root / case
    if not case_dir.is_dir():
        return []

    for pat in (f"{case}_all_results_*.json", "case*_all_results_*.json"):
        fpath = _latest_file(case_dir, pat)
        if fpath:
            break
    else:
        return []

    with open(fpath, encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        # combo format
        sip_recs = payload.get("sip_or_sipv", {}).get("records", [])
        if sip_recs:
            return [r for r in sip_recs if isinstance(r, dict)]
    return []


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _solver_keys(phase_data: dict) -> list[str]:
    """Return solver keys present in a phase result dict (e.g. 'Z3', 'CVC5')."""
    return [k for k in phase_data if k not in ("comparison",) and isinstance(phase_data[k], dict)]


def _extract_constraint_info(solver_data: dict) -> dict[str, Any]:
    """Extract constraint counts from a single solver result."""
    satisfied = solver_data.get("constraints_satisfied", [])
    violated = solver_data.get("constraints_violated", [])
    checked = solver_data.get("constraints_checked", [])
    return {
        "satisfied": list(satisfied),
        "violated": list(violated),
        "checked": list(checked),
        "n_satisfied": len(satisfied),
        "n_violated": len(violated),
        "n_checked": len(checked) or (len(satisfied) + len(violated)),
        "status": solver_data.get("status", "unknown"),
        "execution_time": solver_data.get("execution_time", 0.0),
    }


# ---------------------------------------------------------------------------
# Aggregation structures
# ---------------------------------------------------------------------------


class VerificationAggregator:
    """Collects and aggregates verification signals from experiment records."""

    def __init__(self) -> None:
        # phase -> solver -> {"satisfied": int, "violated": int, "total": int}
        self.phase_solver_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"satisfied": 0, "violated": 0, "total": 0})
        )
        # constraint_name -> {"satisfied": int, "violated": int}
        self.constraint_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"satisfied": 0, "violated": 0}
        )
        # (model, dataset) -> phase -> {"satisfied": int, "violated": int}
        self.per_model_dataset: dict[tuple[str, str], dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"satisfied": 0, "violated": 0})
        )
        # solver -> list of execution_time
        self.solver_times: dict[str, list[float]] = defaultdict(list)
        # solver disagreements: list of dicts
        self.disagreements: list[dict[str, Any]] = []
        # flagged violations with context
        self.flagged_violations: list[dict[str, Any]] = []
        # overhead: (model, dataset) -> {"baseline_ms": [], "verified_ms": []}
        self.overhead_data: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: {"baseline_ms": [], "verified_ms": []}
        )

        self._records_processed = 0
        self._records_with_verification = 0

    # -- public API --

    def ingest_case3_records(self, records: list[dict[str, Any]]) -> None:
        """Ingest case3-style records (verification_mode + verification_results)."""
        for rec in records:
            if rec.get("status") != "success":
                continue

            mode = rec.get("verification_mode", "")
            model = rec.get("model", rec.get("algorithm", "?"))
            dataset = rec.get("dataset", "?")
            exec_ms = rec.get("total_execution_time_ms", 0.0)

            if mode == "none":
                self.overhead_data[(model, dataset)]["baseline_ms"].append(exec_ms)
                continue

            self.overhead_data[(model, dataset)]["verified_ms"].append(exec_ms)
            vr = rec.get("verification_results")
            if not vr or not isinstance(vr, dict):
                continue

            self._records_processed += 1
            self._records_with_verification += 1

            solver_key = mode.upper()  # "z3" -> "Z3"
            for phase in VERIFICATION_PHASES:
                phase_data = vr.get(phase, {})
                if not isinstance(phase_data, dict):
                    continue
                for sk in _solver_keys(phase_data):
                    self._ingest_solver_result(
                        phase, sk, phase_data[sk], model, dataset, rec.get("seed")
                    )

    def ingest_impact_records(self, records: list[dict[str, Any]]) -> None:
        """Ingest case1/2/5-8 records that may contain verification_summary."""
        for rec in records:
            if rec.get("status") != "success":
                continue

            self._records_processed += 1
            model = rec.get("algorithm", rec.get("model", "?"))
            dataset = rec.get("dataset", "?")
            seed = rec.get("seed")

            # Check inference_metrics for verification_summary
            for metrics_key in ("inference_metrics", "metrics"):
                metrics = rec.get(metrics_key, {})
                if not isinstance(metrics, dict):
                    continue
                vs = metrics.get("verification_summary")
                if vs and isinstance(vs, dict):
                    self._records_with_verification += 1
                    for phase in VERIFICATION_PHASES:
                        phase_data = vs.get(phase, {})
                        if not isinstance(phase_data, dict):
                            continue
                        for sk in _solver_keys(phase_data):
                            self._ingest_solver_result(
                                phase, sk, phase_data[sk], model, dataset, seed
                            )

            # Also check isolated_experiments for verification_summary
            for iso_key in ("data_only", "label_only", "param_only"):
                iso = (rec.get("isolated_experiments") or {}).get(iso_key, {})
                if not isinstance(iso, dict):
                    continue
                vs = iso.get("verification_summary")
                if vs and isinstance(vs, dict):
                    for phase in VERIFICATION_PHASES:
                        phase_data = vs.get(phase, {})
                        if not isinstance(phase_data, dict):
                            continue
                        for sk in _solver_keys(phase_data):
                            self._ingest_solver_result(
                                phase, sk, phase_data[sk], model, dataset, seed
                            )

    # -- internal --

    def _ingest_solver_result(
        self,
        phase: str,
        solver: str,
        solver_data: dict,
        model: str,
        dataset: str,
        seed: int | None,
    ) -> None:
        info = _extract_constraint_info(solver_data)

        # phase × solver counts
        bucket = self.phase_solver_counts[phase][solver]
        bucket["satisfied"] += info["n_satisfied"]
        bucket["violated"] += info["n_violated"]
        bucket["total"] += info["n_checked"]

        # per-constraint
        for c in info["satisfied"]:
            self.constraint_stats[c]["satisfied"] += 1
        for c in info["violated"]:
            self.constraint_stats[c]["violated"] += 1

        # per-model-dataset
        md = self.per_model_dataset[(model, dataset)][phase]
        md["satisfied"] += info["n_satisfied"]
        md["violated"] += info["n_violated"]

        # timing
        self.solver_times[solver].append(info["execution_time"])

        # flag violations
        if info["n_violated"] > 0:
            self.flagged_violations.append({
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "phase": phase,
                "solver": solver,
                "violated": info["violated"],
                "status": info["status"],
            })


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _fmt(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(agg: VerificationAggregator, *, cases_used: list[str]) -> str:
    lines: list[str] = []

    lines.append("# Relatório de Valor da Verificação Formal (SIP-V) no Pipeline")
    lines.append("")
    lines.append(f"> Gerado a partir dos casos: {', '.join(cases_used)}")
    lines.append(f"> Registos processados: {agg._records_processed} "
                 f"(com verificação: {agg._records_with_verification})")
    lines.append("")

    # ==================================================================
    # Section 1: What did verifiers detect?
    # ==================================================================
    lines.append("## 1. O que os verificadores detetaram?")
    lines.append("")
    lines.append(
        "Para cada fase do pipeline e cada solver, a tabela mostra quantas constraints "
        "foram **satisfeitas** (UNSAT — a propriedade mantém-se) vs **violadas** (SAT — "
        "a propriedade foi quebrada pela perturbação)."
    )
    lines.append("")

    all_solvers = sorted({s for phases in agg.phase_solver_counts.values() for s in phases})
    if all_solvers:
        # Header
        cols = " | ".join(f"{s} Satisf. | {s} Violadas" for s in all_solvers)
        lines.append(f"| Fase | {cols} | Taxa Violação |")
        sep = "|------|" + "|".join("---------:|----------:" for _ in all_solvers) + "|--------------:|"
        lines.append(sep)

        for phase in VERIFICATION_PHASES:
            label = PHASE_LABELS.get(phase, phase)
            cells: list[str] = []
            total_sat = 0
            total_viol = 0
            for s in all_solvers:
                b = agg.phase_solver_counts[phase][s]
                cells.append(f"{b['satisfied']}")
                cells.append(f"{b['violated']}")
                total_sat += b["satisfied"]
                total_viol += b["violated"]
            total = total_sat + total_viol
            rate = f"{(total_viol / total * 100):.1f}%" if total else "—"
            lines.append(f"| {label} | {' | '.join(cells)} | {rate} |")
        lines.append("")
    else:
        lines.append("*Nenhum resultado de verificação encontrado.*")
        lines.append("")

    # ==================================================================
    # Section 2: Per-constraint breakdown
    # ==================================================================
    lines.append("## 2. Análise por Tipo de Constraint")
    lines.append("")
    lines.append(
        "Cada constraint verificada pelo solver tem uma taxa de violação. "
        "Constraints frequentemente violadas indicam propriedades sensíveis às perturbações."
    )
    lines.append("")

    if agg.constraint_stats:
        lines.append("| Constraint | Satisfeitas | Violadas | Total | Taxa Violação |")
        lines.append("|------------|------------:|---------:|------:|--------------:|")
        for cname in sorted(agg.constraint_stats, key=lambda c: agg.constraint_stats[c]["violated"], reverse=True):
            cs = agg.constraint_stats[cname]
            total = cs["satisfied"] + cs["violated"]
            rate = f"{(cs['violated'] / total * 100):.1f}%" if total else "—"
            lines.append(f"| `{cname}` | {cs['satisfied']} | {cs['violated']} | {total} | {rate} |")
        lines.append("")
    else:
        lines.append("*Sem dados de constraints individuais.*")
        lines.append("")

    # ==================================================================
    # Section 3: Computational overhead
    # ==================================================================
    lines.append("## 3. Custo Computacional da Verificação")
    lines.append("")

    if agg.solver_times:
        lines.append("| Solver | Chamadas | Tempo Médio (s) | Desvio (s) | Máx (s) |")
        lines.append("|--------|:--------:|----------------:|-----------:|--------:|")
        for solver in sorted(agg.solver_times):
            times = agg.solver_times[solver]
            lines.append(
                f"| {solver} | {len(times)} | {_fmt(_mean(times), 4)} | "
                f"{_fmt(_std(times), 4)} | {_fmt(max(times), 4)} |"
            )
        lines.append("")

    # Overhead from case3 baseline vs verified
    overhead_entries: list[tuple[str, str, float, float, float]] = []
    for (model, dataset), data in agg.overhead_data.items():
        if data["baseline_ms"] and data["verified_ms"]:
            bl = _mean(data["baseline_ms"])
            vr = _mean(data["verified_ms"])
            if bl > 0:
                overhead_entries.append((model, dataset, bl, vr, ((vr - bl) / bl) * 100))

    if overhead_entries:
        lines.append("### Overhead por Modelo/Dataset (case3)")
        lines.append("")
        lines.append("| Modelo | Dataset | Baseline (ms) | Com Verificação (ms) | Overhead (%) |")
        lines.append("|--------|---------|:-------------:|:--------------------:|:------------:|")
        for model, dataset, bl, vr, pct in sorted(overhead_entries, key=lambda x: -x[4]):
            lines.append(f"| {model} | {dataset} | {_fmt(bl)} | {_fmt(vr)} | +{_fmt(pct)}% |")
        lines.append("")

    # ==================================================================
    # Section 4: What to flag — violations
    # ==================================================================
    lines.append("## 4. O que sinalizar? Violações detetadas")
    lines.append("")

    if agg.flagged_violations:
        lines.append(
            f"**{len(agg.flagged_violations)} violações** foram registadas. "
            "Cada uma representa uma propriedade formal que a perturbação quebrou."
        )
        lines.append("")

        # Group by (model, dataset, phase)
        groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for fv in agg.flagged_violations:
            groups[(fv["model"], fv["dataset"], fv["phase"])].append(fv)

        lines.append("| Modelo | Dataset | Fase | Solver | Constraints Violadas | Ocorrências |")
        lines.append("|--------|---------|------|--------|----------------------|:-----------:|")
        for (model, dataset, phase), items in sorted(groups.items()):
            label = PHASE_LABELS.get(phase, phase)
            # consolidate by solver and violated set
            by_solver: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for it in items:
                key = ", ".join(sorted(it["violated"]))
                by_solver[it["solver"]][key] += 1
            for solver, violated_counts in sorted(by_solver.items()):
                for violated_str, count in sorted(violated_counts.items(), key=lambda x: -x[1]):
                    lines.append(
                        f"| {model} | {dataset} | {label} | {solver} | "
                        f"`{violated_str}` | {count} |"
                    )
        lines.append("")
    else:
        lines.append("Nenhuma violação detetada — todas as constraints foram satisfeitas.")
        lines.append("")

    # ==================================================================
    # Section 5: Per-model-dataset sensitivity
    # ==================================================================
    lines.append("## 5. Sensibilidade por Modelo e Dataset")
    lines.append("")
    lines.append(
        "Taxa de violação por modelo/dataset em cada fase. "
        "Modelos com alta taxa de violação são mais sensíveis às perturbações — "
        "a verificação formal é **mais valiosa** nesses cenários."
    )
    lines.append("")

    if agg.per_model_dataset:
        lines.append("| Modelo | Dataset | Fase | Satisfeitas | Violadas | Taxa Violação |")
        lines.append("|--------|---------|------|:-----------:|:--------:|:-------------:|")
        for (model, dataset) in sorted(agg.per_model_dataset):
            for phase in VERIFICATION_PHASES:
                d = agg.per_model_dataset[(model, dataset)][phase]
                total = d["satisfied"] + d["violated"]
                if total == 0:
                    continue
                rate = f"{(d['violated'] / total * 100):.1f}%"
                label = PHASE_LABELS.get(phase, phase)
                lines.append(
                    f"| {model} | {dataset} | {label} | "
                    f"{d['satisfied']} | {d['violated']} | {rate} |"
                )
        lines.append("")

    # ==================================================================
    # Section 6: Interpretation & recommendations
    # ==================================================================
    lines.append("## 6. Interpretação: O que a verificação formal agregou?")
    lines.append("")

    total_checked = sum(
        b["total"]
        for phases in agg.phase_solver_counts.values()
        for b in phases.values()
    )
    total_violated = sum(
        b["violated"]
        for phases in agg.phase_solver_counts.values()
        for b in phases.values()
    )
    total_satisfied = total_checked - total_violated

    if total_checked:
        viol_rate = total_violated / total_checked * 100
        lines.append(f"- **Total de constraints verificadas**: {total_checked}")
        lines.append(f"- **Satisfeitas**: {total_satisfied} ({100 - viol_rate:.1f}%)")
        lines.append(f"- **Violadas**: {total_violated} ({viol_rate:.1f}%)")
        lines.append("")

        if total_violated > 0:
            # Most violated constraint
            worst_constraint = max(
                agg.constraint_stats.items(),
                key=lambda x: x[1]["violated"],
            )
            lines.append(
                f"- **Constraint mais frequentemente violada**: "
                f"`{worst_constraint[0]}` ({worst_constraint[1]['violated']} violações)"
            )

            # Most affected phase
            phase_violations = {
                phase: sum(b["violated"] for b in solvers.values())
                for phase, solvers in agg.phase_solver_counts.items()
            }
            worst_phase = max(phase_violations, key=phase_violations.get)
            lines.append(
                f"- **Fase mais afetada**: "
                f"{PHASE_LABELS.get(worst_phase, worst_phase)} "
                f"({phase_violations[worst_phase]} violações)"
            )

            # Most affected model
            model_violations: dict[str, int] = defaultdict(int)
            for (model, _), phases in agg.per_model_dataset.items():
                for d in phases.values():
                    model_violations[model] += d["violated"]
            if model_violations:
                worst_model = max(model_violations, key=model_violations.get)
                lines.append(
                    f"- **Modelo com mais violações**: `{worst_model}` "
                    f"({model_violations[worst_model]} violações)"
                )
        lines.append("")

    lines.append("### O que a verificação formal permite concluir")
    lines.append("")
    lines.append(
        "1. **Deteção de anomalias**: Constraints violadas indicam que as perturbações "
        "quebraram propriedades esperadas (tipos, limites, drift). Sem verificação, "
        "essas violações passariam despercebidas."
    )
    lines.append(
        "2. **Confiança nos resultados**: Constraints satisfeitas provam formalmente "
        "que certas propriedades se mantêm apesar das perturbações."
    )
    lines.append(
        "3. **Guia para seleção de modelo**: Modelos com baixa taxa de violação "
        "são mais robustos e adequados para ambientes ruidosos."
    )
    lines.append(
        "4. **Diagnóstico de pipeline**: A taxa de violação por fase revela "
        "qual etapa do pipeline é mais vulnerável."
    )
    lines.append("")

    lines.append(
        "> **Recomendação**: Em produção, ative a verificação formal pelo menos na "
        "fase pós-perturbação (`post_perturbation`) para monitorizar drift de parâmetros, "
        "e na fase de integridade do modelo (`model_integrity`) para garantir que o "
        "modelo instanciado é válido."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def generate_latex_table(agg: VerificationAggregator) -> str:
    all_solvers = sorted({s for phases in agg.phase_solver_counts.values() for s in phases})
    lines: list[str] = []
    n_cols = 1 + len(all_solvers) * 2 + 1  # phase + (sat+viol)*solvers + rate

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Resultados de verificação formal por fase do pipeline}")
    lines.append(r"\label{tab:verification_value}")
    col_spec = "l" + "rr" * len(all_solvers) + "r"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header_parts = [r"\textbf{Fase}"]
    for s in all_solvers:
        header_parts.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{{s}}}}}")
    header_parts.append(r"\textbf{Violação}")
    lines.append(" & ".join(header_parts) + r" \\")

    sub_parts = [""]
    for _ in all_solvers:
        sub_parts.extend(["Satisf.", "Violadas"])
    sub_parts.append("(%)")
    lines.append(" & ".join(sub_parts) + r" \\")
    lines.append(r"\midrule")

    for phase in VERIFICATION_PHASES:
        label = PHASE_LABELS.get(phase, phase)
        cells = [label]
        total_s = 0
        total_v = 0
        for s in all_solvers:
            b = agg.phase_solver_counts[phase][s]
            cells.append(str(b["satisfied"]))
            cells.append(str(b["violated"]))
            total_s += b["satisfied"]
            total_v += b["violated"]
        total = total_s + total_v
        rate = f"{(total_v / total * 100):.1f}" if total else "--"
        cells.append(rate)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate verification value report from SIP-V results.",
    )
    p.add_argument("--results-dir", default="results")
    p.add_argument("--cases", nargs="+", default=None,
                   help=f"Case dirs to include (default: auto-detect). Options include: {CASE_DIRS}")
    p.add_argument("--output", default=None, help="Output Markdown file (default: stdout)")
    p.add_argument("--latex", action="store_true", help="Append LaTeX table")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results_root = Path(args.results_dir)
    cases = args.cases or CASE_DIRS

    agg = VerificationAggregator()
    cases_found: list[str] = []

    for case in cases:
        records = _load_records(results_root, case)
        if not records:
            continue
        cases_found.append(case)

        # Case3 has a dedicated structure with verification_mode
        if "case3" in case:
            agg.ingest_case3_records(records)
        else:
            agg.ingest_impact_records(records)

    if not cases_found:
        print(
            "Nenhum resultado de verificação encontrado. "
            "Execute case3 ou outros cases com verificação ativada.",
            file=sys.stderr,
        )
        sys.exit(1)

    report = generate_report(agg, cases_used=cases_found)

    if args.latex:
        report += "\n---\n\n## Tabela LaTeX\n\n```latex\n"
        report += generate_latex_table(agg)
        report += "\n```\n"

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Relatório salvo em: {out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
