#!/usr/bin/env python
"""Collect and summarize SIP-Q results across cases 4, 6, 7, 8."""
import json
import glob
from collections import defaultdict
from pathlib import Path


def load_latest(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return json.load(open(files[-1]))


def analyze_case4():
    data = load_latest("results/case4/case4_all_results_*.json")
    if not data:
        print("No case4 results found")
        return
    print(f"=== CASE 4 (SIP-Q standalone): {len(data)} records ===\n")

    # Detailed table
    agg = defaultdict(list)
    baselines = defaultdict(list)
    for r in data:
        algo = r["metadata"]["algorithm"]
        ds = r["metadata"]["dataset"]
        bits = r["bit_width"]
        mode = r["metadata"]["mode"]
        deg = r.get("accuracy_degradation")
        if deg is None:
            deg = r.get("silhouette_degradation")
        if deg is not None:
            agg[(algo, ds, bits, mode)].append(deg)
        bl = r.get("baseline_accuracy") or r.get("baseline_silhouette") or 0
        baselines[(algo, ds)].append(bl)

    print(
        f"{'Algo':4s} | {'Dataset':10s} | Bits | {'Mode':10s} | MeanDegrad | Baseline"
    )
    print("-" * 72)
    for (algo, ds, bits, mode), vals in sorted(agg.items()):
        mean_d = sum(vals) / len(vals)
        bl_vals = baselines[(algo, ds)]
        bl = sum(bl_vals) / len(bl_vals) if bl_vals else 0
        print(
            f"{algo:4s} | {ds:10s} | {bits:2d}   | {mode:10s} | {mean_d:+9.4f}  | {bl:.4f}"
        )

    # Summary by algo
    print("\n--- By Algorithm ---")
    algo_agg = defaultdict(list)
    for r in data:
        d = r.get("accuracy_degradation")
        if d is None:
            d = r.get("silhouette_degradation")
        if d is not None:
            algo_agg[r["metadata"]["algorithm"]].append(d)
    for algo in sorted(algo_agg):
        v = algo_agg[algo]
        print(
            f"  {algo}: mean={sum(v)/len(v):+.4f} max={max(v):+.4f} min={min(v):+.4f} n={len(v)}"
        )

    # Summary by bits
    print("\n--- By Bit Width ---")
    bits_agg = defaultdict(list)
    for r in data:
        d = r.get("accuracy_degradation")
        if d is None:
            d = r.get("silhouette_degradation")
        if d is not None:
            bits_agg[r["bit_width"]].append(d)
    for b in sorted(bits_agg):
        v = bits_agg[b]
        print(f"  {b}-bit: mean={sum(v)/len(v):+.4f} max={max(v):+.4f} min={min(v):+.4f}")

    # Summary by mode
    print("\n--- By Mode ---")
    mode_agg = defaultdict(list)
    for r in data:
        d = r.get("accuracy_degradation")
        if d is None:
            d = r.get("silhouette_degradation")
        if d is not None:
            mode_agg[r["metadata"]["mode"]].append(d)
    for m in sorted(mode_agg):
        v = mode_agg[m]
        print(f"  {m}: mean={sum(v)/len(v):+.4f} max={max(v):+.4f} min={min(v):+.4f}")

    return data


def analyze_combo_case(case_name, result_dir, case_label):
    """Analyze combined case (6, 7, 8) results."""
    pattern = f"{result_dir}/{case_name}_all_results_*.json"
    combo = load_latest(pattern)
    if not combo:
        print(f"\nNo {case_label} results found")
        return

    quant_file = combo.get("quantization", {}).get("all_results_file")
    sip_records = combo.get("sip_or_sipv", {}).get("records", [])

    print(f"\n=== {case_label}: {combo['combination']} ===")

    # SIP/SIP-V records
    success = [r for r in sip_records if r.get("status") == "success"]
    error = [r for r in sip_records if r.get("status") == "error"]
    print(f"SIP records: {len(success)} success, {len(error)} error")

    if success:
        print(f"\n{'Algo':4s} | {'Dataset':10s} | Baseline Acc | Perturbed Acc | Degrad")
        print("-" * 65)
        for r in success:
            algo = r["algorithm"]
            ds = r["dataset"]
            bl = r.get("baseline_metrics", {})
            inf = r.get("inference_metrics", {})
            bl_acc = bl.get("accuracy", bl.get("silhouette_score", "N/A"))
            inf_acc = inf.get("accuracy", inf.get("silhouette_score", "N/A"))
            if isinstance(bl_acc, (int, float)) and isinstance(inf_acc, (int, float)):
                deg = inf_acc - bl_acc
                print(
                    f"{algo:4s} | {ds:10s} | {bl_acc:11.4f}  | {inf_acc:12.4f}  | {deg:+.4f}"
                )
            else:
                print(f"{algo:4s} | {ds:10s} | {bl_acc!s:>11s}  | {inf_acc!s:>12s}  | N/A")

    # Quantization summary from sub-directory
    quant_pattern = f"{result_dir}/quantization/case4_all_results_*.json"
    quant_data = load_latest(quant_pattern)
    if quant_data:
        print(f"\nQuantization records: {len(quant_data)}")
        agg = defaultdict(list)
        for r in quant_data:
            algo = r["metadata"]["algorithm"]
            ds = r["metadata"]["dataset"]
            bits = r["bit_width"]
            d = r.get("accuracy_degradation")
            if d is None:
                d = r.get("silhouette_degradation")
            if d is not None:
                agg[(algo, ds, bits)].append(d)
        print(f"\n{'Algo':4s} | {'Dataset':10s} | Bits | MeanDegrad (across modes & seeds)")
        print("-" * 60)
        for (algo, ds, bits), vals in sorted(agg.items()):
            mean_d = sum(vals) / len(vals)
            print(f"{algo:4s} | {ds:10s} | {bits:2d}   | {mean_d:+.4f}")


if __name__ == "__main__":
    analyze_case4()
    print("\n" + "=" * 80)
    analyze_combo_case("case6_sip_sipq", "results/case6_sip_sipq", "CASE 6")
    print("\n" + "=" * 80)
    analyze_combo_case("case7_sipv_sipq", "results/case7_sipv_sipq", "CASE 7")
    print("\n" + "=" * 80)
    analyze_combo_case("case8_sip_sipv_sipq", "results/case8_sip_sipv_sipq", "CASE 8")
