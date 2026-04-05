#!/usr/bin/env python
"""Generate a consolidated markdown report for case1-8 artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CASES = [
    "case1",
    "case2",
    "case3",
    "case4",
    "case5_sip_sipv",
    "case6_sip_sipq",
    "case7_sipv_sipq",
    "case8_sip_sipv_sipq",
]


def _latest_file(base_dir: Path, pattern: str) -> Path | None:
    files = sorted(base_dir.glob(pattern))
    return files[-1] if files else None


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_seconds(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}s"
    return "-"


def _collect_case_row(results_root: Path, case: str) -> tuple[str, dict[str, str]]:
    case_dir = results_root / case
    summary_file = _latest_file(case_dir, f"{case}_summary_*.json")
    all_results_file = _latest_file(case_dir, f"{case}_all_results_*.json")
    if case == "case4":
        summary_file = _latest_file(case_dir, "case4_summary_*.json")
        all_results_file = _latest_file(case_dir, "case4_all_results_*.json")

    log_files = sorted((case_dir / "logs").rglob("*")) if (case_dir / "logs").exists() else []
    log_files = [p for p in log_files if p.is_file()]

    if not summary_file or not all_results_file:
        return case, {
            "status": "MISSING",
            "records": "-",
            "success": "-",
            "errors": "-",
            "elapsed": "-",
            "summary": "-",
            "all_results": "-",
            "log_count": str(len(log_files)),
        }

    summary_data = _load_json(summary_file)
    all_data = _load_json(all_results_file)

    records = "-"
    success = "-"
    errors = "-"
    elapsed = "-"

    if case in {"case1", "case2", "case3"} and isinstance(summary_data, dict):
        stats = summary_data.get("overall_stats", {})
        records = str(stats.get("successful", 0) + stats.get("failed", 0))
        success = str(stats.get("successful", "-"))
        errors = str(stats.get("failed", "-"))
        elapsed = _fmt_seconds(stats.get("total_time_seconds"))
    elif case == "case4" and isinstance(summary_data, dict):
        stats = summary_data.get("overall_stats", {})
        records = str(stats.get("records_generated", "-"))
        success = records
        errors = "0"
        elapsed = _fmt_seconds(stats.get("total_time_seconds"))
    elif isinstance(summary_data, dict):
        records = str(summary_data.get("sip_records_total", "-"))
        success = str(summary_data.get("sip_records_success", "-"))
        errors = str(summary_data.get("sip_records_error", "-"))
        elapsed = _fmt_seconds(summary_data.get("elapsed_seconds"))

    if case == "case4" and isinstance(all_data, list):
        if records == "-":
            records = str(len(all_data))

    status = "OK"
    return case, {
        "status": status,
        "records": records,
        "success": success,
        "errors": errors,
        "elapsed": elapsed,
        "summary": str(summary_file),
        "all_results": str(all_results_file),
        "log_count": str(len(log_files)),
    }


def _build_report(rows: list[tuple[str, dict[str, str]]], results_root: Path) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Relatorio de Execucao dos Cases 1-8",
        "",
        f"- Gerado em: `{generated_at}`",
        f"- Pasta de resultados: `{results_root}`",
        "",
        "## Resumo Geral",
        "",
        "| Case | Status | Records | Success | Errors | Tempo | Logs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for case, data in rows:
        lines.append(
            f"| `{case}` | {data['status']} | {data['records']} | {data['success']} | "
            f"{data['errors']} | {data['elapsed']} | {data['log_count']} |"
        )

    lines.extend(
        [
            "",
            "## Artefatos",
            "",
        ]
    )

    for case, data in rows:
        lines.append(f"### {case}")
        lines.append(f"- Summary: `{data['summary']}`")
        lines.append(f"- All Results: `{data['all_results']}`")
        lines.append(f"- Total de logs no case: `{data['log_count']}`")
        lines.append("")

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate consolidated report for case1-8.")
    parser.add_argument("--results-root", default="results", help="Root directory for results.")
    parser.add_argument("--output", default="relatorio.md", help="Output markdown file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_root = Path(args.results_root)

    rows = [_collect_case_row(results_root, case) for case in CASES]
    report = _build_report(rows, results_root)

    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
