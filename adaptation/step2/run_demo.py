import argparse
import csv
from pathlib import Path

from adaptation.step2.agents import ExplanationAgent, PracticeAgent
from adaptation.step2.baselines import FixedPolicy, HeuristicPolicy, RandomPolicy
from adaptation.step2.config import SimConfig
from adaptation.step2.evaluator import Evaluator
from adaptation.step2.simulator import StudentSimulator
from adaptation.step2.trainer import Trainer


def _save_comparison_csv(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["policy", "archetype", "avg_reward", "avg_correct_rate", "avg_engagement", "avg_frustration", "avg_boredom"])
        for row in rows:
            writer.writerow([row.policy, row.archetype, f"{row.avg_reward:.6f}", f"{row.avg_correct_rate:.6f}", f"{row.avg_engagement:.6f}", f"{row.avg_frustration:.6f}", f"{row.avg_boredom:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run step2 emotion-only adaptation demo.")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--csv-out", default="adaptation/snapshots/step2/results/comparison.csv")
    args = parser.parse_args()

    config = SimConfig()
    train_sim = StudentSimulator(config=config, seed=11)
    eval_sim = StudentSimulator(config=config, seed=29)
    explanation_agent = ExplanationAgent()
    practice_agent = PracticeAgent()
    trainer = Trainer(train_sim, explanation_agent, practice_agent)
    train_stats = trainer.train(episodes=400)
    evaluator = Evaluator(eval_sim, explanation_agent, practice_agent)
    eval_stats = evaluator.evaluate(episodes=120)

    print("=== Step 2 Training ===")
    print(f"Mean episode reward: {train_stats.mean_reward:.4f}")
    print("=== Step 2 Evaluation ===")
    print(f"Avg reward: {eval_stats.avg_reward:.4f}")
    print(f"Avg practice correct rate: {eval_stats.avg_correct_rate:.4f}")
    print(f"Avg engagement: {eval_stats.avg_engagement:.4f}")
    print(f"Avg frustration: {eval_stats.avg_frustration:.4f}")
    print(f"Avg boredom: {eval_stats.avg_boredom:.4f}")

    if args.compare:
        rows = evaluator.compare_policies([RandomPolicy(), FixedPolicy(), HeuristicPolicy()], episodes_per_archetype=80)
        print("policy,archetype,avg_reward,avg_correct,avg_engagement,avg_frustration,avg_boredom")
        for row in rows:
            print(f"{row.policy},{row.archetype},{row.avg_reward:.4f},{row.avg_correct_rate:.4f},{row.avg_engagement:.4f},{row.avg_frustration:.4f},{row.avg_boredom:.4f}")
        out = Path(args.csv_out)
        _save_comparison_csv(rows, out)
        print(f"Saved comparison CSV to: {out}")

    if args.trace:
        state = eval_sim.reset()
        print("t,phase,state_key,action,reward,correct,engaged,confused,frustrated,bored")
        for t in range(eval_sim.config.steps_per_episode):
            phase = eval_sim.phase()
            state_key = eval_sim.discretize_emotion(state)
            agent = explanation_agent if phase == "explanation" else practice_agent
            action = agent.choose_action(state_key, explore=False)
            result = eval_sim.step(action)
            s = result.state
            print(f"{t},{phase},{state_key},{action},{result.reward:.4f},{result.correct},{s['engaged']:.4f},{s['confused']:.4f},{s['frustrated']:.4f},{s['bored']:.4f}")
            state = s


if __name__ == "__main__":
    main()
