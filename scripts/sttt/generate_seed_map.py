"""Generate deterministic seed mapping for STTT matrix."""
from __future__ import annotations

import json
from pathlib import Path

DATASETS = ["wids", "ieee"]
MODELS = ["lr", "dt", "rf", "mlp"]
SEEDS = list(range(1, 31))

OUT = Path("results/sttt/seed_map.json")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DATASETS:
        for m in MODELS:
            for s in SEEDS:
                rows.append(
                    {
                        "dataset": d,
                        "model": m,
                        "seed": s,
                        "run_key": f"{d}:{m}:{s}",
                        "run_id_sip": f"{d}:{m}:{s}:sip",
                        "run_id_sipv_z3": f"{d}:{m}:{s}:sipv_z3",
                        "run_id_sipv_cvc5": f"{d}:{m}:{s}:sipv_cvc5",
                        "run_id_pandera": f"{d}:{m}:{s}:pandera",
                    }
                )
    payload = {
        "schema_version": "sttt.seedmap.v1",
        "total_expected": len(rows),
        "datasets": DATASETS,
        "models": MODELS,
        "seeds": SEEDS,
        "entries": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {OUT} with {len(rows)} entries")


if __name__ == "__main__":
    main()
