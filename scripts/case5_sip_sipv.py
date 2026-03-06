#!/usr/bin/env python
"""Case Study 5: SIP + SIP-V (without quantization)."""
# pylint: disable=wrong-import-position,duplicate-code

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.case4_sip_q import SEEDS
from scripts.case_combo_common import (
    configure_logger,
    default_case4_algorithms,
    default_case4_datasets,
    run_sip_matrix,
    save_combined_artifacts,
    timed_run,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case Study 5: SIP + SIP-V")
    parser.add_argument("--datasets", nargs="+", default=default_case4_datasets())
    parser.add_argument("--algorithms", nargs="+", default=default_case4_algorithms())
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--solver", choices=["z3", "cvc5", "both", "auto"], default="z3")
    parser.add_argument("--output-dir", default="results/case5_sip_sipv")
    parser.add_argument("--log-dir", default="logs/case5_sip_sipv")
    return parser.parse_args()


def main() -> None:
    """Run SIP + SIP-V matrix with the same dataset/model family as case4."""
    args = _parse_args()
    logger, log_file = configure_logger("case5_sip_sipv", args.log_dir, "case5_sip_sipv")
    logger.info("Case5 log file: %s", log_file)

    sip_records, elapsed = timed_run(
        run_sip_matrix,
        datasets=args.datasets,
        algorithms=args.algorithms,
        seeds=args.seeds,
        verification_enabled=True,
        solver=args.solver,
    )
    all_results_file, summary_file = save_combined_artifacts(
        output_dir=args.output_dir,
        file_prefix="case5_sip_sipv",
        combination_name="SIP + SIP-V",
        quant_summary=None,
        quant_results_file=None,
        sip_records=sip_records,
        elapsed_seconds=elapsed,
    )
    logger.info("All results saved to: %s", all_results_file)
    logger.info("Summary saved to: %s", summary_file)


if __name__ == "__main__":
    main()
