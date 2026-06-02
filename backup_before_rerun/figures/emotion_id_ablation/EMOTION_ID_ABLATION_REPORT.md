# Emotion-ID Ablation + DQN Stability Report

## Methodology

Two-condition DQN study isolating the discrete `emotion_id` observation channel.

- **Condition A (Full Emotion):** knowledge, engagement, frustration, confusion, boredom, emotion_id
- **Condition D (No Emotion ID):** knowledge, engagement, frustration, confusion, boredom

Unchanged: reward function, emotional dynamics, action space, transitions, 50k training budget.
Gain ratio: **10:1** | Seeds: **15** | Eval: 500 episodes
Checkpoints evaluated: [10000, 20000, 30000, 40000, 50000]

## Results - Final Checkpoint (50k)

| condition_label | n_seeds | LG mean | LG std | LG 95% CI | Adapt acc | Success rate |
| --- | --- | --- | --- | --- | --- | --- |
| A: Full Emotion | 15 | 0.4115 | 0.1849 | [0.318, 0.505] | 0.4546 | 0.1779 |
| D: No Emotion ID | 15 | 0.4261 | 0.1589 | [0.346, 0.507] | 0.4857 | 0.1827 |

## Results - Best Checkpoint Selection

| condition_label | n_seeds | LG mean | LG std | LG 95% CI | Adapt acc | Success rate |
| --- | --- | --- | --- | --- | --- | --- |
| A: Full Emotion | 15 | 0.5801 | 0.1088 | [0.525, 0.635] | 0.7464 | 0.3489 |
| D: No Emotion ID | 15 | 0.6128 | 0.0939 | [0.565, 0.660] | 0.8193 | 0.4096 |

## Pairwise A vs D (Best Checkpoint)

- **learning_gain**: A=0.5801 vs D=0.6128, delta=-0.0326, d=-0.321, p_holm=0.3868
- **final_knowledge**: A=0.7098 vs D=0.7329, delta=-0.0231, d=-0.329, p_holm=0.7519
- **success_rate**: A=0.3489 vs D=0.4096, delta=-0.0607, d=-0.365, p_holm=1.0000
- **adaptation_accuracy**: A=0.7464 vs D=0.8193, delta=-0.0728, d=-0.349, p_holm=1.0000
- **mean_episode_reward**: A=3.0417 vs D=2.9410, delta=+0.1007, d=0.367, p_holm=1.0000

## Seed Robustness (10 vs 15 seeds)

| condition | seed_set | eval_type | n | learning_gain_mean | learning_gain_std | learning_gain_ci |
| --- | --- | --- | --- | --- | --- | --- |
| A_full_emotion | 10_seeds | final | 10 | 0.4529 | 0.1809 | [0.341, 0.565] |
| A_full_emotion | 10_seeds | best | 10 | 0.5781 | 0.1269 | [0.499, 0.657] |
| A_full_emotion | 15_seeds | final | 15 | 0.4115 | 0.1849 | [0.318, 0.505] |
| A_full_emotion | 15_seeds | best | 15 | 0.5801 | 0.1088 | [0.525, 0.635] |
| D_no_emotion_id | 10_seeds | final | 10 | 0.4632 | 0.1583 | [0.365, 0.561] |
| D_no_emotion_id | 10_seeds | best | 10 | 0.6113 | 0.1037 | [0.547, 0.676] |
| D_no_emotion_id | 15_seeds | final | 15 | 0.4261 | 0.1589 | [0.346, 0.507] |
| D_no_emotion_id | 15_seeds | best | 15 | 0.6128 | 0.0939 | [0.565, 0.660] |

## Emotion Volatility Correlations

| condition | eval_type | volatility_metric | pearson_r | pearson_p | spearman_r | spearman_p | n_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_full_emotion | final | confusion_variance | 0.8284 | 0.000136 | 0.4536 | 0.089486 | 15 |
| A_full_emotion | final | boredom_variance | -0.6694 | 0.00634 | -0.6714 | 0.006128 | 15 |
| A_full_emotion | final | emotion_id_switches | 0.1535 | 0.584963 | -0.1821 | 0.515882 | 15 |
| A_full_emotion | best | confusion_variance | 0.8714 | 2.3e-05 | 0.9 | 5e-06 | 15 |
| A_full_emotion | best | boredom_variance | -0.7774 | 0.000647 | -0.9214 | 1e-06 | 15 |
| A_full_emotion | best | emotion_id_switches | 0.2669 | 0.336268 | 0.5286 | 0.042797 | 15 |
| D_no_emotion_id | final | confusion_variance | 0.8067 | 0.000279 | 0.7393 | 0.001636 | 15 |
| D_no_emotion_id | final | boredom_variance | -0.2867 | 0.300193 | -0.6036 | 0.0172 | 15 |
| D_no_emotion_id | final | emotion_id_switches | 0.2724 | 0.32607 | 0.2043 | 0.465162 | 15 |
| D_no_emotion_id | best | confusion_variance | 0.9348 | 0.0 | 0.8286 | 0.000135 | 15 |
| D_no_emotion_id | best | boredom_variance | -0.8402 | 8.8e-05 | -0.925 | 1e-06 | 15 |
| D_no_emotion_id | best | emotion_id_switches | 0.4821 | 0.068777 | 0.6893 | 0.004474 | 15 |

## Research Questions

1. Does emotion_id improve learning gain? **False** (delta=-0.0327, p_holm=0.3868)
2. Does emotion_id improve adaptation accuracy? **False** (p_holm=1.0000)
3. Does removing emotion_id reduce variance? **True** (std A=0.109, std D=0.094)
4. Does checkpoint selection improve stability? **True** (A std reduction 41.2%, D 40.9%)
5. Does emotional volatility explain poor seeds? **True**
   - Confusion: significant positive (rho=0.900, p=0.0000)
   - Boredom: significant negative (rho=-0.921, p=0.0000)
   - Emotion ID switches: significant positive (rho=0.529, p=0.0428)

## Final Recommendation

**Thesis model:** D: No Emotion ID
- Checkpoint strategy: **best** checkpoint
- Learning gain: **0.613** (95% CI [0.565, 0.660], std=0.094)
- Adaptation accuracy: **0.819** (95% CI [0.706, 0.933])

emotion_id does not significantly improve learning gain (delta=-0.0327, p_holm=0.387); no_emotion_id reduces seed variance (std 0.094 vs 0.109); checkpoint selection reduces variance vs final-50k model; Final thesis model: D: No Emotion ID with best checkpoint selection

## Discussion

The emotion update audit identified confusion and boredom as the most volatile
continuous channels (low AR persistence, frequent >0.2 step changes). The discrete
`emotion_id` is derived from these channels and may add redundant or noisy information
for value-based learning. This study tests whether removing it improves DQN stability
without altering simulator dynamics.

Reference: emotion update audit (`figures/emotion_update_audit/EMOTION_UPDATE_AUDIT.md`).
