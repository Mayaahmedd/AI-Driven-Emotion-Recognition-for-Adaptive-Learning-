# DQN Smoke Test Report

**Date:** 2026-05-31
**Scope:** Lightweight verification (1 seed, quick-equivalent budget)

## Configuration

- Seed: 42
- Mode: quick-equivalent (1 seed, 5k train, 50 eval, 20 val)
- Observation: knowledge, engagement, frustration, confusion, boredom (no emotion_id)
- Training timesteps: 5,000
- Eval episodes: 50
- Validation episodes (checkpoint selection): 20
- Checkpoints: [5,000]

## Results

| Check | Status |
|-------|--------|
| Training completed | Yes |
| Evaluation completed | Yes |
| Checkpoint selection completed | Yes |
| Report generation completed | Yes |

## Metrics (seed 42, Tier A)

```json
{
  "best_checkpoint_step": 5000,
  "best_val_learning_gain": 0.4766710186636921,
  "best_learning_gain": 0.5048045113935999,
  "best_final_knowledge": 0.6601764985846755,
  "best_success_rate": 0.22,
  "final_learning_gain": 0.5048045113935999,
  "elapsed_seconds": 4.5,
  "n_checkpoint_eval_rows": 1
}
```

## Errors / Notes

- Analyze not re-run: using pre-existing FINAL_DQN_MODEL_REPORT.md from prior 20-seed study.

## Conclusion

Pipeline **operational**.
