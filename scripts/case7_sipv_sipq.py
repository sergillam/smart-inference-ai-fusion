#!/usr/bin/env python
"""Case Study 7: SIP-V + SIP-Q."""
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
    relocate_new_default_artifacts,
    run_quantization_matrix,
    run_sip_matrix,
    save_combined_artifacts,
    snapshot_default_artifacts,
    timed_run,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case Study 7: SIP-V + SIP-Q")
    parser.add_argument("--datasets", nargs="+", default=default_case4_datasets())
    parser.add_argument("--algorithms", nargs="+", default=default_case4_algorithms())
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--method", default="uniform")
    parser.add_argument("--dtype-profile", default="integer", choices=["integer", "float16"])
    parser.add_argument("--solver", choices=["z3", "cvc5", "both", "auto"], default="z3")
    parser.add_argument("--output-dir", default="results/case7_sipv_sipq")
    parser.add_argument("--log-dir", default="logs/case7_sipv_sipq")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run SIP-V + SIP-Q matrix and save combined artifacts."""
    args = _parse_args()
    artifact_snapshot = snapshot_default_artifacts()
    logger, log_file = configure_logger("case7_sipv_sipq", args.log_dir, "case7_sipv_sipq")
    logger.info("Case7 log file: %s", log_file)

    quant_output_dir = Path(args.output_dir) / "quantization"
    (quant_pair, quant_elapsed) = timed_run(
        run_quantization_matrix,
        output_dir=str(quant_output_dir),
        datasets=args.datasets,
        algorithms=args.algorithms,
        bits=args.bits,
        seeds=args.seeds,
        method=args.method,
        dtype_profile=args.dtype_profile,
        resume=args.resume,
    )
    quant_summary, quant_file = quant_pair

    sip_records, sip_elapsed = timed_run(
        run_sip_matrix,
        datasets=args.datasets,
        algorithms=args.algorithms,
        seeds=args.seeds,
        verification_enabled=True,
        solver=args.solver,
    )
    notes = [
        "Current SIP-V runtime in this framework is coupled to the inference pipeline (SIP).",
        "Therefore SIP-V metrics here are produced through SIP-enabled inference runs.",
    ]
    all_results_file, summary_file = save_combined_artifacts(
        output_dir=args.output_dir,
        file_prefix="case7_sipv_sipq",
        combination_name="SIP-V + SIP-Q",
        quant_summary=quant_summary,
        quant_results_file=quant_file,
        sip_records=sip_records,
        notes=notes,
        elapsed_seconds=quant_elapsed + sip_elapsed,
    )
    logger.info("All results saved to: %s", all_results_file)
    logger.info("Summary saved to: %s", summary_file)
    moved_artifacts = relocate_new_default_artifacts(
        snapshot=artifact_snapshot, output_dir=args.output_dir
    )
    if moved_artifacts:
        logger.info("Relocated %d auxiliary artifacts to %s", len(moved_artifacts), args.output_dir)


if __name__ == "__main__":
    main()
