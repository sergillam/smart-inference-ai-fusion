"""Run full STTT matrix using in-process run_single execution."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from scripts.sttt.config import DATASETS, MODELS, SEEDS
from scripts.sttt.run_single import main as _run_single_main


def _parse_args() -> argparse.Namespace:
    env_num_seeds = os.getenv("NUM_SEEDS")
    default_seeds = SEEDS
    if env_num_seeds and env_num_seeds.isdigit():
        default_seeds = list(range(1, int(env_num_seeds) + 1))

    parser = argparse.ArgumentParser(description="Run full STTT matrix")
    parser.add_argument("--datasets", nargs="+", default=["wids", "ieee"])
    parser.add_argument("--models", nargs="+", default=sorted(MODELS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=default_seeds)
    parser.add_argument("--output-dir", default="results/sttt")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--failure-log",
        default="results/sttt/run_all_failures.log",
        help="Append failed run_single invocations to this log file.",
    )
    return parser.parse_args()


def _invoke_run_single(dataset: str, model: str, seed: int, output_dir: str, skip_existing: bool) -> None:
    import sys

    argv = [
        "scripts.sttt.run_single",
        "--dataset",
        dataset,
        "--model",
        model,
        "--seed",
        str(seed),
        "--output-dir",
        output_dir,
    ]
    if skip_existing:
        argv.append("--skip-existing")

    prev = sys.argv
    try:
        sys.argv = argv
        _run_single_main()
    finally:
        sys.argv = prev


def main() -> None:
    args = _parse_args()
    failures: list[str] = []
    failure_log_path = Path(args.failure_log)
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                total += 1
                try:
                    _invoke_run_single(dataset, model, seed, args.output_dir, args.skip_existing)
                except Exception:
                    failed_cmd = (
                        f"--dataset {dataset} --model {model} --seed {seed} --output-dir {args.output_dir}"
                    )
                    failures.append(failed_cmd)
                    with open(failure_log_path, "a", encoding="utf-8") as handle:
                        handle.write(failed_cmd + "\n")
                        handle.write(traceback.format_exc().strip() + "\n")

    if failures:
        print(f"Completed {total} runs with {len(failures)} failures")
    else:
        print(f"Completed {total} runs with no failures")


if __name__ == "__main__":
    main()
