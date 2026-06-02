"""
Final thesis DQN vs Rule (ERT) vs Random - per thesis configuration A and B.

Configuration A: Balanced-2 + nearly equal gains + single canonical student
Configuration B: Balanced-2 + nearly equal gains + mixed student population

Both use G4 IRT gain ratio (10:1), thesis DQN hyperparameters, and checkpoints
from final_thesis_comparison when available. Rule / Random: eval only.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python3 -m RL_Module.final_thesis_algorithm_comparison --phase all
  python3 -m RL_Module.final_thesis_algorithm_comparison --phase run --resume
  python3 -m RL_Module.final_thesis_algorithm_comparison --phase analyze
  python3 -m RL_Module.final_thesis_algorithm_comparison --configuration B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import RL_Module.config as cfg
from RL_Module.agents.dqn_agent import DQNAgent, make_env
from RL_Module.agents.random_agent import RandomAgent
from RL_Module.agents.rule_based import RuleBasedAgent
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation.metrics import (
    action_frequency_report,
    confidence_interval,
    evaluate,
    normalized_knowledge_gain,
)
from RL_Module.explainability.explainer import Explainer
from RL_Module.final_thesis_comparison import (
    BASELINE_HP,
    CHECKPOINT_STEPS,
    EVAL_EPS,
    G4_GAIN,
    NEARLY_EQUAL_GAINS,
    SEEDS,
    THESIS_CONFIGS,
    TRAIN_TS,
    VAL_EPS,
    _checkpoint_path_from_selection,
    _patch_action_gains,
    _patch_gain,
    _patch_reward,
    _population_patch,
    _restore_action_gains,
    _restore_gain,
    _restore_reward,
)
from RL_Module.mdp_definition import ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "final_thesis_algorithm_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_ORDER = ["A_stable_canonical", "B_realistic_mixed"]
ALGORITHMS = ["DQN", "Rule", "Random"]

THESIS_HP: Dict[str, Any] = dict(BASELINE_HP)

METRICS = [
    "learning_gain",
    "success_rate",
    "adaptation_accuracy",
    "mean_episode_reward",
    "dropout_rate",
]

STAT_METRICS = ["learning_gain", "success_rate", "adaptation_accuracy"]
ACTION_NAMES = list(ID_TO_ACTION.values())

RESULTS_PATH = OUT_DIR / "algorithm_comparison_results.csv"
LEGACY_DQN_SOURCES = [
    _HERE / "final_thesis_comparison" / "final_comparison_results.csv",
    _HERE / "figures" / "combined_improvement_validation" / "combined_results.csv",
]


@contextmanager
def thesis_mdp(configuration: str) -> Iterator[Dict[str, Any]]:
    spec = THESIS_CONFIGS[configuration]
    gain_snap = _patch_gain()
    reward_snap = _patch_reward(spec["reward"])
    action_snap = _patch_action_gains(spec["action_gains"])
    with _population_patch(spec["population_mode"]):
        try:
            yield spec
        finally:
            _restore_action_gains(action_snap)
            _restore_reward(reward_snap)
            _restore_gain(gain_snap)


def _find_legacy_dqn_checkpoint(seed: int, configuration: str) -> Optional[str]:
    for path in LEGACY_DQN_SOURCES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "configuration" in df.columns:
            sub = df[(df["configuration"] == configuration) & (df["seed"] == seed)]
            if not sub.empty:
                selection = str(sub.iloc[0].get("selection", sub.iloc[0].get("dqn_checkpoint", "")))
                if selection and selection != "nan":
                    ckpt = _checkpoint_path_from_selection(selection)
                    if Path(ckpt).exists():
                        return ckpt
        legacy_key = THESIS_CONFIGS[configuration]["legacy_combined_key"]
        if "configuration" in df.columns:
            sub = df[(df["configuration"] == legacy_key) & (df["seed"] == seed)]
            if not sub.empty:
                ckpt = _checkpoint_path_from_selection(str(sub.iloc[0]["selection"]))
                if Path(ckpt).exists():
                    return ckpt

    tag = f"final_thesis_{configuration}_s{seed}"
    ckpt_dir = cfg.MODELS_DIR / tag
    if ckpt_dir.is_dir():
        zips = sorted(ckpt_dir.glob("dqn_*_steps.zip"))
        if zips:
            return str(zips[-1])

    legacy_tag = f"combined_{THESIS_CONFIGS[configuration]['legacy_combined_key']}_s{seed}"
    ckpt_dir = cfg.MODELS_DIR / legacy_tag
    if ckpt_dir.is_dir():
        zips = sorted(ckpt_dir.glob("dqn_*_steps.zip"))
        if zips:
            return str(zips[-1])
    return None


def quick_eval_lg(
    agent: DQNAgent,
    seed: int,
    configuration: str,
    n_episodes: int = VAL_EPS,
) -> float:
    spec = THESIS_CONFIGS[configuration]
    with thesis_mdp(configuration):
        env = StudentEnv(
            population_seed=seed,
            obs_ablation="full_emotion",
            emotion_dynamics="full",
        )
        cfg.set_all_seeds(seed)
        gains: List[float] = []
        for ep in range(1, n_episodes + 1):
            obs, info = env.reset(seed=seed + ep)
            mask = info["action_masks"]
            start_k = env._state.knowledge
            done = False
            while not done:
                action = agent.predict(obs, mask)
                obs, _, term, trunc, info = env.step(action)
                mask = info["action_masks"]
                done = term or trunc
            gains.append(normalized_knowledge_gain(start_k, env._state.knowledge))
        env.close()
    return float(np.mean(gains))


def train_dqn(seed: int, configuration: str) -> Tuple[str, float]:
    spec = THESIS_CONFIGS[configuration]
    tag = f"final_thesis_{configuration}_s{seed}"
    with thesis_mdp(configuration):
        cfg.set_all_seeds(seed)
        train_env = make_env(
            seed=seed,
            obs_ablation="full_emotion",
            emotion_dynamics="full",
            algo_tag=tag,
        )
        agent = DQNAgent()
        ckpt_dir = cfg.MODELS_DIR / tag
        paths = agent.train_with_checkpoints(
            train_env,
            TRAIN_TS,
            seed,
            hyperparams=THESIS_HP,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_freq=min(10_000, TRAIN_TS),
        )
        train_env.close()

    valid_paths = [p for p in paths if any(f"_{s}_steps" in p for s in CHECKPOINT_STEPS)]
    if not valid_paths:
        valid_paths = paths

    best_path, best_val = agent.select_best_checkpoint(
        valid_paths,
        lambda a, s: quick_eval_lg(a, s, configuration, VAL_EPS),
        seed,
    )
    return best_path, best_val


def _make_agent(algorithm: str, seed: int):
    if algorithm == "DQN":
        return DQNAgent()
    if algorithm == "Rule":
        return RuleBasedAgent(obs_ablation="full_emotion")
    if algorithm == "Random":
        return RandomAgent(seed)
    raise ValueError(algorithm)


def eval_algorithm(
    configuration: str,
    algorithm: str,
    seed: int,
    dqn_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    spec = THESIS_CONFIGS[configuration]
    with thesis_mdp(configuration):
        cfg.set_all_seeds(seed)
        agent = _make_agent(algorithm, seed)
        if algorithm == "DQN":
            if dqn_checkpoint is None or not Path(dqn_checkpoint).exists():
                raise FileNotFoundError(
                    f"DQN checkpoint missing for {configuration} seed {seed}"
                )
            agent.load_checkpoint(dqn_checkpoint)

        env = StudentEnv(
            population_seed=seed,
            obs_ablation="full_emotion",
            emotion_dynamics="full",
        )
        env.set_algorithm_name(algorithm)
        explainer = Explainer(algorithm)
        result = evaluate(
            agent,
            env,
            n_episodes=EVAL_EPS,
            seed=seed,
            algorithm=algorithm,
            explainer=explainer,
        )
        explainer.close()
        env.close()

    freq_report = action_frequency_report(result["episode_results"], print_table=False)
    freqs = freq_report["frequencies"]

    row: Dict[str, Any] = {
        "configuration": configuration,
        "configuration_label": spec["label"],
        "population_mode": spec["population_mode"],
        "algorithm": algorithm,
        "seed": seed,
        "eval_episodes": EVAL_EPS,
        "dqn_checkpoint": dqn_checkpoint or "",
        "gain_ratio": G4_GAIN["ratio_label"],
        **{m: result[m] for m in METRICS},
        "action_entropy": float(-np.sum([v * np.log(v) for v in freqs.values() if v > 0]))
        if freqs
        else 0.0,
        "dominant_action_pct": float(max(freqs.values())) if freqs else 0.0,
    }
    for name in ACTION_NAMES:
        row[f"freq_{name}"] = freqs.get(name, 0.0)
    return row


def _normalize_legacy_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        if "configuration" not in row or pd.isna(row.get("configuration")):
            row["configuration"] = "B_realistic_mixed"
        if "configuration_label" not in row:
            row["configuration_label"] = THESIS_CONFIGS[row["configuration"]]["label"]
        if "population_mode" not in row:
            row["population_mode"] = THESIS_CONFIGS[row["configuration"]]["population_mode"]
        if "gain_ratio" not in row:
            row["gain_ratio"] = G4_GAIN["ratio_label"]
        normalized.append(row)
    return normalized


def run_study(
    configurations: List[str],
    seeds: List[int],
    resume: bool,
    force_train_dqn: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if resume and RESULTS_PATH.exists():
        rows = _normalize_legacy_rows(pd.read_csv(RESULTS_PATH).to_dict("records"))
    done = {(r["configuration"], r["algorithm"], int(r["seed"])) for r in rows}

    dqn_ckpts: Dict[Tuple[str, int], str] = {}
    for r in rows:
        if r["algorithm"] == "DQN" and r.get("dqn_checkpoint"):
            dqn_ckpts[(r["configuration"], int(r["seed"]))] = str(r["dqn_checkpoint"])

    for configuration in configurations:
        spec = THESIS_CONFIGS[configuration]
        print(f"\n=== {spec['label']} ({configuration}) ===")

        for seed in seeds:
            key = (configuration, "DQN", seed)
            if key not in done:
                ckpt = None if force_train_dqn else _find_legacy_dqn_checkpoint(seed, configuration)
                if ckpt is None or force_train_dqn:
                    print(f"  DQN seed={seed}: training ({TRAIN_TS} steps)...")
                    t0 = time.time()
                    ckpt, val_lg = train_dqn(seed, configuration)
                    print(f"    checkpoint={ckpt} val_lg={val_lg:.3f} ({time.time()-t0:.0f}s)")
                else:
                    print(f"  DQN seed={seed}: using checkpoint {ckpt}")
                dqn_ckpts[(configuration, seed)] = ckpt
                row = eval_algorithm(configuration, "DQN", seed, dqn_ckpts[(configuration, seed)])
                rows.append(row)
                pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
                done.add(key)
                print(
                    f"    lg={row['learning_gain']:.3f} succ={row['success_rate']:.3f} "
                    f"adapt={row['adaptation_accuracy']:.3f}"
                )

        for algorithm in ("Rule", "Random"):
            print(f"\n--- {algorithm} ---")
            for seed in seeds:
                key = (configuration, algorithm, seed)
                if key in done:
                    continue
                row = eval_algorithm(configuration, algorithm, seed)
                rows.append(row)
                pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
                done.add(key)
                print(
                    f"  {algorithm} seed={seed}: lg={row['learning_gain']:.3f} "
                    f"succ={row['success_rate']:.3f} adapt={row['adaptation_accuracy']:.3f}"
                )

    return pd.DataFrame(rows)


def cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (configuration, algorithm), sub in df.groupby(["configuration", "algorithm"]):
        spec = THESIS_CONFIGS[configuration]
        lgs = sub["learning_gain"].astype(float).tolist()
        lg_mean, lg_lo, lg_hi = confidence_interval(lgs)
        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "population_mode": spec["population_mode"],
            "algorithm": algorithm,
            "n_seeds": len(sub),
            "learning_gain_mean": round(lg_mean, 4),
            "learning_gain_std": round(float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0, 4),
            "learning_gain_ci_95": f"[{lg_lo:.3f}, {lg_hi:.3f}]",
        }
        for m in METRICS:
            if m == "learning_gain":
                continue
            vals = sub[m].astype(float).tolist()
            mean, lo, hi = confidence_interval(vals)
            row[f"{m}_mean"] = round(mean, 4)
            row[f"{m}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
            row[f"{m}_ci_95"] = f"[{lo:.3f}, {hi:.3f}]"
        if "action_entropy" in sub.columns:
            row["action_entropy_mean"] = round(float(sub["action_entropy"].mean()), 4)
            row["dominant_action_pct_mean"] = round(float(sub["dominant_action_pct"].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_vs_dqn(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    p_vals: List[float] = []
    for configuration in df["configuration"].unique():
        sub = df[df["configuration"] == configuration]
        dqn = sub[sub["algorithm"] == "DQN"]
        for baseline in ("Rule", "Random"):
            base = sub[sub["algorithm"] == baseline]
            for metric in STAT_METRICS:
                v_d = dqn[metric].astype(float).tolist()
                v_b = base[metric].astype(float).tolist()
                if len(v_d) < 2 or len(v_b) < 2:
                    continue
                t_stat, p_val = stats.ttest_ind(v_d, v_b, equal_var=False)
                mean_d, lo_d, hi_d = confidence_interval(v_d)
                mean_b, lo_b, hi_b = confidence_interval(v_b)
                rows.append({
                    "configuration": configuration,
                    "comparison": f"DQN_vs_{baseline}",
                    "metric": metric,
                    "mean_DQN": round(mean_d, 4),
                    "mean_baseline": round(mean_b, 4),
                    "delta_DQN_minus_baseline": round(mean_d - mean_b, 4),
                    "ci_95_DQN": f"[{lo_d:.3f}, {hi_d:.3f}]",
                    "ci_95_baseline": f"[{lo_b:.3f}, {hi_b:.3f}]",
                    "cohens_d": round(cohens_d(v_d, v_b), 4),
                    "t": round(float(t_stat), 4),
                    "p_raw": round(float(p_val), 6),
                })
                p_vals.append(p_val)
    if p_vals:
        order = np.argsort(p_vals)
        holm = [1.0] * len(p_vals)
        for rank, idx in enumerate(order):
            holm[idx] = min(1.0, p_vals[idx] * (len(p_vals) - rank))
        for i, row in enumerate(rows):
            row["p_holm"] = round(holm[i], 6)
    return pd.DataFrame(rows)


def _plot_bars(summary_df: pd.DataFrame, configuration: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    spec = THESIS_CONFIGS[configuration]
    sub = summary_df[summary_df["configuration"] == configuration]
    order = ["DQN", "Rule", "Random"]
    sub = sub.set_index("algorithm").reindex(order).reset_index()
    colors = ["#2A9D8F", "#457B9D", "#E76F51"]
    xs = np.arange(3)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("learning_gain_mean", "learning_gain_std", "Learning Gain"),
        ("success_rate_mean", "success_rate_std", "Success Rate"),
        ("adaptation_accuracy_mean", "adaptation_accuracy_std", "Adaptation Accuracy"),
    ]
    for ax, (mean_c, std_c, title) in zip(axes, panels):
        means = sub[mean_c].tolist()
        stds = sub[std_c].tolist()
        ax.bar(xs, means, yerr=stds, capsize=5, color=colors, edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(order)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    pop_label = "canonical student" if spec["population_mode"] == "single_canonical" else "mixed population"
    fig.suptitle(
        f"{spec['label']}: DQN vs Rule vs Random\n"
        f"(Balanced-2, nearly equal gains, {pop_label}, n={int(sub['n_seeds'].iloc[0])} seeds)",
        fontsize=10,
    )
    fig.tight_layout()
    short = "A" if configuration.startswith("A_") else "B"
    fig.savefig(OUT_DIR / f"algorithm_comparison_{short}.png", dpi=150)
    plt.close(fig)


def _print_summary(summary_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    for configuration in CONFIG_ORDER:
        if configuration not in summary_df["configuration"].values:
            continue
        spec = THESIS_CONFIGS[configuration]
        sub_summary = summary_df[summary_df["configuration"] == configuration]
        sub_stats = stats_df[stats_df["configuration"] == configuration]
        print("\n" + "=" * 72)
        print(f"{spec['label'].upper()}: DQN vs RULE vs RANDOM")
        print("=" * 72)
        for _, row in sub_summary.iterrows():
            print(
                f"\n{row['algorithm']}: "
                f"LG={row['learning_gain_mean']:.4f} {row['learning_gain_ci_95']}, "
                f"succ={row['success_rate_mean']:.3f}, "
                f"adapt={row['adaptation_accuracy_mean']:.3f}, "
                f"reward={row['mean_episode_reward_mean']:.2f}"
            )
        print("\n--- DQN vs baselines (Holm-corrected) ---")
        for _, row in sub_stats.iterrows():
            sig = "yes" if row.get("p_holm", 1) < 0.05 else "no"
            print(
                f"  {row['comparison']} {row['metric']}: "
                f"delta={row['delta_DQN_minus_baseline']:+.4f}, d={row['cohens_d']:.3f}, "
                f"p_holm={row.get('p_holm', row['p_raw']):.4f} (sig={sig})"
            )
        dqn_lg = float(sub_summary[sub_summary["algorithm"] == "DQN"]["learning_gain_mean"].iloc[0])
        rule_lg = float(sub_summary[sub_summary["algorithm"] == "Rule"]["learning_gain_mean"].iloc[0])
        rand_lg = float(sub_summary[sub_summary["algorithm"] == "Random"]["learning_gain_mean"].iloc[0])
        print("\n--- Conclusion ---")
        if dqn_lg > rule_lg and dqn_lg > rand_lg:
            print("  DQN outperforms both Rule and Random on mean learning gain.")
        elif dqn_lg > rand_lg:
            print("  DQN beats Random; compare Rule for adaptation / success tradeoffs.")
        print("=" * 72)


def analyze(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    if df is None:
        if not RESULTS_PATH.exists():
            raise FileNotFoundError(f"No results at {RESULTS_PATH}")
        df = pd.read_csv(RESULTS_PATH)
        df = pd.DataFrame(_normalize_legacy_rows(df.to_dict("records")))

    summary_df = aggregate_summary(df)
    stats_df = pairwise_vs_dqn(df)

    df.to_csv(RESULTS_PATH, index=False)
    summary_df.to_csv(OUT_DIR / "algorithm_comparison_summary.csv", index=False)
    stats_df.to_csv(OUT_DIR / "algorithm_comparison_stats.csv", index=False)

    for configuration in df["configuration"].unique():
        _plot_bars(summary_df, configuration)

    _print_summary(summary_df, stats_df)

    report: Dict[str, Any] = {
        "study": "final_thesis_algorithm_comparison",
        "configurations": {},
        "seeds": sorted(df["seed"].unique().tolist()),
        "train_timesteps_dqn": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "dqn_hyperparameters": THESIS_HP,
        "gain_ratio": G4_GAIN["ratio_label"],
        "action_gains": NEARLY_EQUAL_GAINS,
    }
    for configuration in df["configuration"].unique():
        spec = THESIS_CONFIGS[configuration]
        sub_summary = summary_df[summary_df["configuration"] == configuration]
        sub_stats = stats_df[stats_df["configuration"] == configuration]
        report["configurations"][configuration] = {
            "label": spec["label"],
            "population_mode": spec["population_mode"],
            "reward_weights": spec["reward"],
            "summary": sub_summary.to_dict("records"),
            "statistical_tests": sub_stats.to_dict("records"),
        }

    with open(OUT_DIR / "algorithm_comparison_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final thesis DQN vs Rule vs Random (configs A and B)"
    )
    parser.add_argument("--phase", choices=["run", "analyze", "all"], default="all")
    parser.add_argument(
        "--configuration",
        choices=["A", "B", "both"],
        default="both",
        help="A=canonical student, B=mixed population, both=run both",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-train-dqn", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    global TRAIN_TS, EVAL_EPS, VAL_EPS, SEEDS
    seeds = list(SEEDS)
    configurations = list(CONFIG_ORDER)
    if args.configuration == "A":
        configurations = ["A_stable_canonical"]
    elif args.configuration == "B":
        configurations = ["B_realistic_mixed"]

    if args.quick:
        seeds = [42, 7]
        TRAIN_TS = 5_000
        EVAL_EPS = 50
        VAL_EPS = 20

    if args.phase in ("run", "all"):
        print("=== Final Thesis Algorithm Comparison (run) ===")
        print(f"Configurations: {configurations}")
        print(f"Gain ratio: {G4_GAIN['ratio_label']} | Reward: Balanced-2")
        run_study(configurations, seeds, args.resume, args.force_train_dqn)

    if args.phase in ("analyze", "all"):
        print("\n=== Final Thesis Algorithm Comparison (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
