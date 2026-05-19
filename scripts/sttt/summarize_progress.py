"""Summarize STTT matrix execution progress and failures."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_DATASETS = ["wids", "ieee"]
EXPECTED_MODELS = ["lr", "dt", "rf", "mlp"]
EXPECTED_SEEDS = list(range(1, 31))

ROOT = Path("results/sttt")
RUNS = ROOT / "runs"
FAILS = ROOT / "run_all_failures.log"
OUT = ROOT / "progress_summary.json"


def main() -> None:
    completed: set[tuple[str, str, int]] = set()
    success: set[tuple[str, str, int]] = set()
    failed: set[tuple[str, str, int]] = set()
    if RUNS.exists():
        for path in sorted(RUNS.glob("*.json")):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            d = obj.get("dataset")
            m = obj.get("model")
            s = obj.get("seed")
            if d in EXPECTED_DATASETS and m in EXPECTED_MODELS and isinstance(s, int):
                completed.add((d, m, s))
                if obj.get("status") == "failed":
                    failed.add((d, m, s))
                else:
                    success.add((d, m, s))

    all_expected = {(d, m, s) for d in EXPECTED_DATASETS for m in EXPECTED_MODELS for s in EXPECTED_SEEDS}
    missing = sorted(all_expected - completed)

    fail_lines: list[str] = []
    if FAILS.exists():
        fail_lines = [ln.strip() for ln in FAILS.read_text(encoding="utf-8").splitlines() if ln.strip()]

    pattern = re.compile(r"--dataset\s+(\S+)\s+--model\s+(\S+)\s+--seed\s+(\d+)")
    fail_counter_combo: Counter[tuple[str, str]] = Counter()
    fail_seeds: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for ln in fail_lines:
        m = pattern.search(ln)
        if not m:
            continue
        d, md, s = m.group(1), m.group(2), int(m.group(3))
        fail_counter_combo[(d, md)] += 1
        fail_seeds[(d, md)].append(s)

    by_combo_completed: Counter[tuple[str, str]] = Counter((d, m) for d, m, _ in completed)

    summary = {
        "expected_total": len(all_expected),
        "attempted_total": len(completed),
        "attempted_pct": round((len(completed) / len(all_expected)) * 100, 2),
        "success_total": len(success),
        "success_pct": round((len(success) / len(all_expected)) * 100, 2),
        "failed_total_from_manifests": len(failed),
        "failed_pct_from_manifests": round((len(failed) / len(all_expected)) * 100, 2),
        "completed_by_combo": {
            f"{d}/{m}": by_combo_completed.get((d, m), 0) for d in EXPECTED_DATASETS for m in EXPECTED_MODELS
        },
        "failure_count": len(fail_lines),
        "failure_by_combo": {
            f"{d}/{m}": fail_counter_combo.get((d, m), 0) for d in EXPECTED_DATASETS for m in EXPECTED_MODELS
        },
        "failure_seeds": {
            f"{d}/{m}": sorted(vals) for (d, m), vals in sorted(fail_seeds.items())
        },
        "missing_count": len(missing),
        "missing_preview": [
            {"dataset": d, "model": m, "seed": s} for d, m, s in missing[:40]
        ],
    }

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
