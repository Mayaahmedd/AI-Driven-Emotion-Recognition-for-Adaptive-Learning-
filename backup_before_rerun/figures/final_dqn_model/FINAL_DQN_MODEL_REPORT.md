# Final DQN Thesis Model - Matched 20-Seed Confirmation

## Configuration

**Observation (agent-facing):** knowledge, engagement, frustration, confusion, boredom

**Removed:** emotion_id

**Baseline:** default emotional dynamics, default reward, default transitions

**Tier A emotion stabilization** (runtime `SIMULATOR_PARAMS` patch only):

| Parameter | Default | Tier A |
|-----------|---------|--------|
| LAMBDA_ENGAGEMENT | 0.60 | 0.80 |
| LAMBDA_FRUSTRATION | 0.69 | 0.845 |
| LAMBDA_CONFUSION | 0.47 | 0.735 |
| LAMBDA_BOREDOM | 0.36 | 0.68 |
| EMOTION_NOISE_STD | 0.03 | 0.015 |

Gain ratio: **10:1** | Training: **50,000** steps | Eval: **500** episodes | Checkpoints: [10000, 20000, 30000, 40000, 50000]

Seeds (matched n=20): `[42, 0, 1, 7, 13, 21, 99, 100, 200, 314, 500, 777, 1234, 2024, 5555, 8888, 9999, 10101, 20202, 30303]`

## Table 1 - Aggregate Metrics (Best Checkpoint, n=20)

| model | n_seeds | learning_gain_mean | learning_gain_std | learning_gain_ci | final_knowledge_mean | success_rate_mean | adaptation_accuracy_mean | mean_episode_reward_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tier A (20 seeds) | 20 | 0.5888 | 0.0923 | [0.548, 0.629] | 0.716 | 0.3697 | 0.7058 | 2.9564 |
| Baseline (20 seeds) | 20 | 0.6089 | 0.0887 | [0.570, 0.648] | 0.7297 | 0.4003 | 0.7751 | 2.9041 |

## Table 2 - Pairwise Comparison (Tier A vs Baseline, n=20)

| metric | tier_a_mean | baseline_mean | tier_minus_baseline | tier_a_std | baseline_std | cohens_d | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learning_gain | 0.5888 | 0.6089 | -0.0201 | 0.0923 | 0.0887 | -0.2224 | 1.0 |
| final_knowledge | 0.716 | 0.7297 | -0.0137 | 0.0634 | 0.0611 | -0.2209 | 1.0 |
| success_rate | 0.3697 | 0.4003 | -0.0306 | 0.1528 | 0.1502 | -0.202 | 1.0 |
| adaptation_accuracy | 0.7058 | 0.7751 | -0.0693 | 0.1658 | 0.2138 | -0.3623 | 1.0 |
| mean_episode_reward | 2.9564 | 2.9041 | 0.0524 | 0.398 | 0.3868 | 0.1334 | 0.675413 |

## Checkpoint Selection (Tier A, per seed)

| seed | selected_checkpoint | validation_learning_gain | final_eval_learning_gain | selected_eval_learning_gain |
| --- | --- | --- | --- | --- |
| 42.0 | 40000.0 | 0.6578 | 0.4111 | 0.6834 |
| 0.0 | 40000.0 | 0.4276 | 0.2968 | 0.4503 |
| 1.0 | 30000.0 | 0.4251 | 0.2543 | 0.4214 |
| 7.0 | 50000.0 | 0.6144 | 0.6212 | 0.6212 |
| 13.0 | 40000.0 | 0.6648 | 0.261 | 0.6573 |
| 21.0 | 30000.0 | 0.641 | 0.2218 | 0.6805 |
| 99.0 | 50000.0 | 0.6494 | 0.66 | 0.66 |
| 100.0 | 30000.0 | 0.4393 | 0.32 | 0.4348 |
| 200.0 | 40000.0 | 0.4594 | 0.3668 | 0.4598 |
| 314.0 | 40000.0 | 0.6697 | 0.1996 | 0.6466 |
| 500.0 | 50000.0 | 0.6153 | 0.6203 | 0.6203 |
| 777.0 | 30000.0 | 0.6665 | 0.1518 | 0.6483 |
| 1234.0 | 40000.0 | 0.5028 | 0.356 | 0.5249 |
| 2024.0 | 40000.0 | 0.6301 | 0.6023 | 0.62 |
| 5555.0 | 20000.0 | 0.502 | 0.3202 | 0.4982 |
| 8888.0 | 50000.0 | 0.4957 | 0.5387 | 0.5387 |
| 9999.0 | 50000.0 | 0.6117 | 0.6108 | 0.6108 |
| 10101.0 | 30000.0 | 0.7137 | 0.4233 | 0.7064 |
| 20202.0 | 40000.0 | 0.6239 | 0.458 | 0.6184 |
| 30303.0 | 20000.0 | 0.6607 | 0.4517 | 0.6749 |

## Checkpoint Selection (Baseline, per seed)

| seed | selected_checkpoint | validation_learning_gain | final_eval_learning_gain | selected_eval_learning_gain |
| --- | --- | --- | --- | --- |
| 0.0 | 40000.0 | 0.6776 | 0.3603 | 0.695 |
| 1.0 | 20000.0 | 0.5822 | 0.3125 | 0.5944 |
| 7.0 | 30000.0 | 0.5998 | 0.5416 | 0.6137 |
| 13.0 | 40000.0 | 0.6865 | 0.3253 | 0.6796 |
| 21.0 | 40000.0 | 0.6454 | 0.2817 | 0.6758 |
| 42.0 | 50000.0 | 0.6286 | 0.6588 | 0.6588 |
| 99.0 | 40000.0 | 0.4487 | 0.4204 | 0.4374 |
| 100.0 | 50000.0 | 0.6665 | 0.6741 | 0.6741 |
| 200.0 | 30000.0 | 0.6733 | 0.672 | 0.6738 |
| 314.0 | 20000.0 | 0.4361 | 0.3854 | 0.4104 |
| 500.0 | 40000.0 | 0.6749 | 0.4934 | 0.6819 |
| 777.0 | 30000.0 | 0.5958 | 0.1661 | 0.5815 |
| 1234.0 | 50000.0 | 0.4611 | 0.4913 | 0.4913 |
| 2024.0 | 40000.0 | 0.6457 | 0.24 | 0.6361 |
| 5555.0 | 20000.0 | 0.6953 | 0.3692 | 0.6879 |
| 8888.0 | 50000.0 | 0.5484 | 0.592 | 0.592 |
| 9999.0 | 10000.0 | 0.5125 | 0.2539 | 0.5134 |
| 10101.0 | 10000.0 | 0.6999 | 0.375 | 0.6844 |
| 20202.0 | 30000.0 | 0.5275 | 0.4271 | 0.5267 |
| 30303.0 | 20000.0 | 0.6563 | 0.4989 | 0.6704 |

## Research Questions (Matched 20-Seed Protocol)

1. Does Baseline still outperform Tier A? **True** (baseline - tier A LG = +0.0201)
2. Are differences statistically significant? **False** (learning gain p_holm=1.0000, Cohen's d=-0.222)
3. Does Baseline still have the highest learning gain? **True**
4. Which model has the lowest variance? **`baseline_thesis`** (Tier A std=0.092, Baseline std=0.089)
5. Which model should be frozen as the final thesis model? **`baseline_thesis`**

## Final Thesis Recommendation

**Selected model:** `baseline_thesis`

- Observation: knowledge, engagement, frustration, confusion, boredom
- Emotion configuration: default (mdp_definition literature defaults)
- Checkpoint strategy: best checkpoint selected by validation learning gain (checkpoints [10000, 20000, 30000, 40000, 50000])
- Mean learning gain: **0.609**
- 95% CI: [0.570, 0.648]
- Std: 0.089

Baseline retains higher mean learning gain on matched 20 seeds (0.609 vs Tier A 0.589); with equal or lower variance (std 0.089 vs 0.092); the LG difference is not statistically significant after Holm correction (p=1.000)

**This is the model that should be used for all remaining thesis experiments and thesis reporting.**

## Reference (15-Seed Overlap)

- learning_gain: Tier A=0.5751 vs Baseline=0.6128, d=-0.393, p_holm=0.5831

## Reference Studies

- Baseline seeds 1-15: `figures/emotion_id_ablation/` (D_no_emotion_id)
- Baseline seeds 16-20: `figures/final_dqn_model/baseline_per_run_20.csv`
- Tier A: `figures/final_dqn_model/final_dqn_per_run.csv`
