#!/usr/bin/env python3
"""Extract SIP-V verification data from experiment results."""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_json(path):
    with open(os.path.join(BASE, path)) as f:
        return json.load(f)

print("=" * 60)
print("CASE 3 — SIP-V Formal Verification Evaluation")
print("=" * 60)
c3 = load_json("results/case3/case3_all_results_2026-03-07T12-57-30Z.json")
for r in c3:
    mode = r["verification_mode"]
    model = r["model"]
    ds = r["dataset"]
    m = r["metrics"]
    print(f"\nMode: {mode}, Model: {model}, Dataset: {ds}")
    print(f"  Accuracy: {m['accuracy']:.4f}, F1: {m['f1']:.4f}")
    print(f"  Exec time: {m['execution_time_seconds']:.4f}s")
    if m.get("timing_breakdown"):
        for k, v in m["timing_breakdown"].items():
            print(f"    {k}: {v:.4f}s")
    if m.get("data_inference_statistics"):
        stats = m["data_inference_statistics"]
        print(f"  Transformations applied: {len(stats.get('transformations_applied', []))}")

print("\n" + "=" * 60)
print("CASE 5 — SIP + SIP-V")
print("=" * 60)
c5 = load_json("results/case5_sip_sipv/case5_sip_sipv_all_results_2026-03-07T12-57-34Z.json")
if isinstance(c5, dict):
    recs = c5.get("sip_or_sipv", {}).get("records", [])
    for r in recs:
        print(f"\nDataset: {r['dataset']}, Model: {r['algorithm']}, Solver: {r.get('solver','N/A')}")
        bm = r.get("baseline_metrics", {})
        im = r.get("inference_metrics", {})
        print(f"  Baseline Accuracy: {bm.get('accuracy',0):.4f}")
        print(f"  Inference Accuracy: {im.get('accuracy',0):.4f}")
        print(f"  Inference Exec time: {im.get('execution_time_seconds',0):.4f}s")
        if im.get("timing_breakdown"):
            for k, v in im["timing_breakdown"].items():
                print(f"    {k}: {v:.6f}s")
        vs = im.get("verification_summary", {})
        for phase, solvers in vs.items():
            print(f"  Verification Phase: {phase}")
            for solver_name, details in solvers.items():
                print(f"    Solver: {solver_name}")
                print(f"      Status: {details.get('status')}")
                print(f"      Exec time: {details.get('execution_time',0):.6f}s")
                print(f"      Checked: {details.get('constraints_checked', [])}")
                print(f"      Satisfied: {details.get('constraints_satisfied', [])}")
                print(f"      Violated: {details.get('constraints_violated', [])}")
        if im.get("data_inference_statistics"):
            stats = im["data_inference_statistics"]
            print(f"  Transformations applied: {len(stats.get('transformations_applied', []))}")

print("\n" + "=" * 60)
print("CASE 7 — SIP-V + SIP-Q")
print("=" * 60)
c7 = load_json("results/case7_sipv_sipq/case7_sipv_sipq_all_results_2026-03-07T12-57-38Z.json")
if isinstance(c7, dict):
    recs = c7.get("sip_or_sipv", {}).get("records", [])
    for r in recs:
        print(f"\nDataset: {r['dataset']}, Model: {r['algorithm']}, Solver: {r.get('solver','N/A')}")
        bm = r.get("baseline_metrics", {})
        im = r.get("inference_metrics", {})
        print(f"  Baseline Accuracy: {bm.get('accuracy',0):.4f}")
        print(f"  Inference Accuracy: {im.get('accuracy',0):.4f}")
        vs = im.get("verification_summary", {})
        for phase, solvers in vs.items():
            print(f"  Verification Phase: {phase}")
            for solver_name, details in solvers.items():
                print(f"    Solver: {solver_name}, Status: {details.get('status')}, Time: {details.get('execution_time',0):.6f}s")
                print(f"      Checked: {details.get('constraints_checked', [])}")
                print(f"      Satisfied: {details.get('constraints_satisfied', [])}")
                print(f"      Violated: {details.get('constraints_violated', [])}")
    qrecs = c7.get("quantization", {})
    if isinstance(qrecs, dict) and qrecs.get("all_results_file"):
        print(f"  Quantization results file: {qrecs['all_results_file']}")

print("\n" + "=" * 60)
print("CASE 8 — SIP + SIP-V + SIP-Q")
print("=" * 60)
c8 = load_json("results/case8_sip_sipv_sipq/case8_sip_sipv_sipq_all_results_2026-03-07T12-57-42Z.json")
if isinstance(c8, dict):
    recs = c8.get("sip_or_sipv", {}).get("records", [])
    for r in recs:
        print(f"\nDataset: {r['dataset']}, Model: {r['algorithm']}, Solver: {r.get('solver','N/A')}")
        bm = r.get("baseline_metrics", {})
        im = r.get("inference_metrics", {})
        print(f"  Baseline Accuracy: {bm.get('accuracy',0):.4f}")
        print(f"  Inference Accuracy: {im.get('accuracy',0):.4f}")
        vs = im.get("verification_summary", {})
        for phase, solvers in vs.items():
            print(f"  Verification Phase: {phase}")
            for solver_name, details in solvers.items():
                print(f"    Solver: {solver_name}, Status: {details.get('status')}, Time: {details.get('execution_time',0):.6f}s")
                print(f"      Satisfied: {details.get('constraints_satisfied', [])}")
                print(f"      Violated: {details.get('constraints_violated', [])}")

# Also check the latex tables from case3
print("\n" + "=" * 60)
print("CASE 3 — LaTeX Tables")
print("=" * 60)
try:
    with open(os.path.join(BASE, "results/case3/case3_latex_tables_2026-03-07T12-57-30Z.tex")) as f:
        print(f.read())
except Exception as e:
    print(f"Error: {e}")
