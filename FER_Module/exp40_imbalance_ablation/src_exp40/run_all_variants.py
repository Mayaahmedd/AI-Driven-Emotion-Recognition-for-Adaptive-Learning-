"""
Orchestrator for the exp40 imbalance ablation.

Trains and evaluates variants A, B, C sequentially, then calls the
comparison script to build the final A vs B vs C report. The script
also runs ``make_balanced_csv_exp40.py`` first if the oversampled CSV
is missing, so a fresh Kaggle session can execute everything from a
single command.

Usage
-----
    python run_all_variants.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIGS_DIR = HERE / "configs"

CONFIGS = [
    CONFIGS_DIR / "variant_A_oversampling_only.json",
    CONFIGS_DIR / "variant_B_posweight_only.json",
    CONFIGS_DIR / "variant_C_combined.json",
]

OVERSAMPLED_CSV = "/kaggle/working/FER_Project/processed_exp40/train_multilabel_exp40_oversampled.csv"


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main():
    if not os.path.exists(OVERSAMPLED_CSV):
        print("Oversampled CSV missing; building it now ...")
        run([sys.executable, str(HERE / "make_balanced_csv_exp40.py")])

    for cfg in CONFIGS:
        run([sys.executable, str(HERE / "train_exp40.py"), "--config", str(cfg)])

    for variant in ["A", "B", "C"]:
        run([sys.executable, str(HERE / "evaluate_exp40.py"), "--variant", variant])

    run([sys.executable, str(HERE / "compare_variants.py")])

    print("\nExp40 imbalance ablation finished. See:")
    print("  /kaggle/working/FER_Project/checkpoints/exp40_imbalance_ablation/")


if __name__ == "__main__":
    main()
