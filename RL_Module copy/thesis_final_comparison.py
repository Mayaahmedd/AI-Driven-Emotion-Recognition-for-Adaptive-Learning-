"""
Final thesis comparison: Random vs ERT vs DQN (10 seeds, 1000 eval episodes).

Uses fixed DQN hyperparameters from thesis model selection.
Gain ratio fixed at G4 (10:1) unless overridden.

Usage:
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.thesis_final_comparison --phase all
  python -m RL_Module.thesis_final_comparison --phase analyze
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RL_Module import config
from RL_Module.thesis_gain_sensitivity import (
    ALGORITHMS,
    ALL_METRICS,
    FINAL_DQN_HP,
    GAIN_CONFIGS,
    OUT_DIR as _SENS_DIR,
    analyze_results,
    run_experiment,
)

OUT_DIR = _HERE / "figures" / "thesis_final_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GAIN = "G4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Final thesis tutor comparison")
    parser.add_argument("--phase", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gain-id", default=DEFAULT_GAIN)
    args = parser.parse_args()

    seeds = list(config.SEEDS)
    train_ts = config.TRAINING_TIMESTEPS
    eval_eps = config.EVAL_EPISODES
    if args.quick:
        seeds = [42, 7]
        train_ts = 2_000
        eval_eps = 100

    gain_params = GAIN_CONFIGS[args.gain_id]
    csv_path = OUT_DIR / "thesis_final_per_run.csv"

    if args.phase in ("run", "all"):
        import RL_Module.config as cfg
        from RL_Module.thesis_gain_sensitivity import run_single

        per_run: List[Dict[str, Any]] = []
        done_keys = set()
        if args.resume and csv_path.exists():
            existing = pd.read_csv(csv_path)
            per_run = existing.to_dict("records")
            done_keys = {(r["algorithm"], int(r["seed"])) for r in per_run}

        for seed in seeds:
            for algo in ALGORITHMS:
                if (algo, seed) in done_keys:
                    continue
                row = run_single(algo, seed, args.gain_id, gain_params, train_ts, eval_eps)
                per_run.append(row)
                pd.DataFrame(per_run).to_csv(csv_path, index=False)

    if args.phase in ("analyze", "all"):
        if not csv_path.exists():
            raise SystemExit(f"No results at {csv_path}")
        df = pd.read_csv(csv_path)
        df["gain_id"] = args.gain_id
        report = analyze_results(df)
        # Copy outputs to final comparison dir
        for name in (
            "ranking_table_g4.csv",
            "pairwise_tests_g4.csv",
            "thesis_gain_sensitivity_report.json",
            "THESIS_GAIN_SENSITIVITY_REPORT.md",
            "thesis_gain_overview.png",
        ):
            src = _SENS_DIR / name
            if src.exists():
                dst = OUT_DIR / name.replace("gain", "final")
                dst.write_bytes(src.read_bytes())
        summary = {
            "study": "thesis_final_comparison",
            "gain_id": args.gain_id,
            "ratio": gain_params["ratio_label"],
            "seeds": seeds,
            "eval_episodes": eval_eps,
            "dqn_hyperparameters": FINAL_DQN_HP,
            "conclusion": report["thesis_conclusion"],
        }
        with open(OUT_DIR / "thesis_final_comparison_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Final report: {OUT_DIR / 'thesis_final_comparison_report.json'}")


if __name__ == "__main__":
    main()
