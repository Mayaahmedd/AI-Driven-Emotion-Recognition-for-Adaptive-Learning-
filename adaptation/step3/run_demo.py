import argparse
import csv
from pathlib import Path

from adaptation.step3.baselines import FixedPolicy, HeuristicPolicy, RandomPolicy
from adaptation.step3.config import DQNConfig, SimConfig
from adaptation.step3.dqn_agents import ExplanationDQNAgent, PracticeDQNAgent
from adaptation.step3.evaluator import Evaluator
from adaptation.step3.simulator import StudentSimulator
from adaptation.step3.trainer import Trainer


def _save_comparison_csv(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "policy",
                "archetype",
                "avg_reward",
                "avg_correct_rate",
                "avg_engagement",
                "avg_frustration",
                "avg_boredom",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.policy,
                    row.archetype,
                    f"{row.avg_reward:.6f}",
                    f"{row.avg_correct_rate:.6f}",
                    f"{row.avg_engagement:.6f}",
                    f"{row.avg_frustration:.6f}",
                    f"{row.avg_boredom:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3 DQN adaptation demo.")
    parser.add_argument("--compare", action="store_true", help="Compare DQN with random/fixed/heuristic baselines.")
    parser.add_argument("--episodes", type=int, default=450, help="Training episodes for DQN.")
    parser.add_argument(
        "--csv-out",
        default="adaptation/step3/results/comparison.csv",
        help="Output CSV path for comparison rows.",
    )
    args = parser.parse_args()

    sim_config = SimConfig()
    dqn_config = DQNConfig()
    train_sim = StudentSimulator(sim_config, seed=21)
    eval_sim = StudentSimulator(sim_config, seed=31)

    explanation_agent = ExplanationDQNAgent(dqn_config)
    practice_agent = PracticeDQNAgent(dqn_config)

    trainer = Trainer(train_sim, explanation_agent, practice_agent)
    train_stats = trainer.train(episodes=args.episodes)

    evaluator = Evaluator(eval_sim, explanation_agent, practice_agent)
    eval_stats = evaluator.evaluate(episodes=120)

    print("=== Step 3 (DQN) Training ===")
    print(f"Mean episode reward: {train_stats.mean_reward:.4f}")
    print("=== Step 3 (DQN) Evaluation ===")
    print(f"Avg reward: {eval_stats.avg_reward:.4f}")
    print(f"Avg practice correct rate: {eval_stats.avg_correct_rate:.4f}")
    print(f"Avg engagement: {eval_stats.avg_engagement:.4f}")
    print(f"Avg frustration: {eval_stats.avg_frustration:.4f}")
    print(f"Avg boredom: {eval_stats.avg_boredom:.4f}")

    if args.compare:
        rows = evaluator.compare_policies(
            baselines=[RandomPolicy(), FixedPolicy(), HeuristicPolicy()],
            episodes_per_archetype=80,
        )
        print("=== Step 3 Comparison (Per Archetype) ===")
        print("policy,archetype,avg_reward,avg_correct,avg_engagement,avg_frustration,avg_boredom")
        for row in rows:
            print(
                f"{row.policy},{row.archetype},{row.avg_reward:.4f},{row.avg_correct_rate:.4f},"
                f"{row.avg_engagement:.4f},{row.avg_frustration:.4f},{row.avg_boredom:.4f}"
            )
        out_path = Path(args.csv_out)
        _save_comparison_csv(rows, out_path)
        print(f"Saved comparison CSV to: {out_path}")


if __name__ == "__main__":
    main()
