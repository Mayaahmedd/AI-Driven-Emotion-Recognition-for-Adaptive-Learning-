"""
ERT action-distribution validation after redesign.

Runs the Expert Rule Tutor for multiple seeds and 500 evaluation episodes,
reports action frequencies, diversity, state visitation, and reachability.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python3 -m RL_Module.ert_action_validation
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RL_Module.agents.rule_based import ExpertRuleBasedAgent
from RL_Module.environment.student_env import StudentEnv
from RL_Module.mdp_definition import ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "ert_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 13, 21, 99, 314]
EVAL_EPS = 500
TARGET_ACTIONS = ("explanation", "simplify_problem", "break", "harder_problem")
TARGET_LO, TARGET_HI = 0.05, 0.10
NO_ACTION_TARGET_HI = 0.20


def _shannon_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    p = np.array([c / total for c in counts.values() if c > 0], dtype=float)
    return float(-np.sum(p * np.log(p)))


def run_seed(seed: int, n_episodes: int) -> Dict[str, Any]:
    env = StudentEnv(population_seed=seed)
    agent = ExpertRuleBasedAgent()
    action_counts: Dict[str, int] = defaultdict(int)
    rule_counts: Dict[str, int] = defaultdict(int)
    state_bins: Dict[str, int] = defaultdict(int)
    total_steps = 0

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent.predict(
                obs,
                info["action_masks"],
                persistent_flag=info.get("persistent_frustration_flag", False),
            )
            name = ID_TO_ACTION[action]
            action_counts[name] += 1
            if agent.last_rule_id:
                rule_counts[agent.last_rule_id] += 1
            s = env._state
            state_bins[
                f"k={s.knowledge:.1f}|e={s.engagement:.1f}|"
                f"f={s.frustration:.2f}|c={s.confusion:.2f}|b={s.boredom:.2f}"
            ] += 1
            total_steps += 1
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
    env.close()

    freqs = {a: action_counts[a] / max(total_steps, 1) for a in ID_TO_ACTION.values()}
    return {
        "seed": seed,
        "total_steps": total_steps,
        "action_counts": dict(action_counts),
        "action_frequencies": {k: round(v, 4) for k, v in freqs.items()},
        "shannon_entropy": round(_shannon_entropy(action_counts), 4),
        "unique_state_bins": len(state_bins),
        "rule_frequencies": {
            k: round(v / max(total_steps, 1), 4)
            for k, v in sorted(rule_counts.items(), key=lambda x: -x[1])
        },
        "no_action_freq": round(freqs.get("no_action", 0.0), 4),
    }


def validate_all(seeds: List[int], n_episodes: int) -> Dict[str, Any]:
    per_seed = [run_seed(s, n_episodes) for s in seeds]

    agg_freq: Dict[str, List[float]] = defaultdict(list)
    for row in per_seed:
        for action, freq in row["action_frequencies"].items():
            agg_freq[action].append(freq)

    summary_freq = {
        a: {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }
        for a, vals in agg_freq.items()
    }

    target_checks = {}
    for action in TARGET_ACTIONS:
        m = summary_freq[action]["mean"]
        target_checks[action] = {
            "mean_freq": m,
            "in_5_10_pct": TARGET_LO <= m <= TARGET_HI,
            "in_5_20_pct": TARGET_LO <= m <= 0.20,
        }

    no_action_mean = summary_freq["no_action"]["mean"]
    all_actions_used = all(
        summary_freq[a]["min"] > 0 for a in ID_TO_ACTION.values()
    )

    report = {
        "seeds": seeds,
        "eval_episodes_per_seed": n_episodes,
        "per_seed": per_seed,
        "aggregate_action_frequencies": summary_freq,
        "target_action_checks": target_checks,
        "no_action_mean": no_action_mean,
        "no_action_below_20pct": no_action_mean <= NO_ACTION_TARGET_HI,
        "all_actions_reachable": all_actions_used,
        "mean_shannon_entropy": round(
            float(np.mean([r["shannon_entropy"] for r in per_seed])), 4
        ),
    }
    return report


def _plot(report: Dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    actions = list(ID_TO_ACTION.values())
    means = [report["aggregate_action_frequencies"][a]["mean"] for a in actions]
    stds = [report["aggregate_action_frequencies"][a]["std"] for a in actions]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors = ["#4C72B0" if a in TARGET_ACTIONS else "#999999" for a in actions]
    ax.bar(actions, means, yerr=stds, capsize=4, color=colors, edgecolor="black")
    ax.axhline(TARGET_LO, color="green", linestyle="--", alpha=0.6, label="5% target")
    ax.axhline(TARGET_HI, color="green", linestyle=":", alpha=0.6, label="10% target")
    ax.axhline(NO_ACTION_TARGET_HI, color="red", linestyle="--", alpha=0.6, label="20% no_action cap")
    ax.set_ylabel("Frequency")
    ax.set_title("ERT Action Distribution (mean across seeds)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    per_seed_df = pd.DataFrame(
        [{**{"seed": r["seed"]}, **r["action_frequencies"]} for r in report["per_seed"]]
    )
    heat = per_seed_df.set_index("seed")[actions]
    im = ax2.imshow(heat.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.5)
    ax2.set_xticks(range(len(actions)))
    ax2.set_xticklabels(actions, rotation=45, ha="right")
    ax2.set_yticks(range(len(heat)))
    ax2.set_yticklabels(heat.index.astype(str))
    ax2.set_title("Per-seed action frequencies")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ert_action_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"ERT validation: seeds={SEEDS}, episodes={EVAL_EPS}")
    report = validate_all(SEEDS, EVAL_EPS)

    freq_rows = []
    for action, stats in report["aggregate_action_frequencies"].items():
        freq_rows.append({"action": action, **stats})
    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(OUT_DIR / "action_frequency_summary.csv", index=False)

    per_seed_df = pd.DataFrame(
        [
            {
                "seed": r["seed"],
                "shannon_entropy": r["shannon_entropy"],
                "no_action_freq": r["no_action_freq"],
                **r["action_frequencies"],
            }
            for r in report["per_seed"]
        ]
    )
    per_seed_df.to_csv(OUT_DIR / "action_frequency_per_seed.csv", index=False)

    with open(OUT_DIR / "ert_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    _plot(report)

    print("\n--- Aggregate action frequencies ---")
    print(freq_df.to_string(index=False))
    print(f"\nno_action mean: {report['no_action_mean']:.3f} (target <= {NO_ACTION_TARGET_HI})")
    print(f"Shannon entropy (mean): {report['mean_shannon_entropy']:.3f}")
    print(f"All actions reachable: {report['all_actions_reachable']}")
    for action, chk in report["target_action_checks"].items():
        status = "OK" if chk["in_5_20_pct"] else "CHECK"
        print(f"  {action}: {chk['mean_freq']:.3f} [{status}]")
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
