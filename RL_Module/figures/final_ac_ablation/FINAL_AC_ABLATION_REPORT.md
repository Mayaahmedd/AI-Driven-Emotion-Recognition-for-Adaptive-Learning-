# Final Thesis Ablation: Emotion-Aware vs Knowledge-Only DQN

## Experimental Design

### Condition A - Emotion-Aware DQN (Frozen Thesis Model)

- **Observation:** knowledge, engagement, frustration, confusion, boredom
- **Excluded:** emotion_id
- **Environment:** default emotional dynamics, default reward, default transitions

### Condition C - Knowledge-Only DQN (Strict No-Emotion)

- **Observation:** knowledge only
- **Environment:** strict no-emotion (transitions, reward, masks, dropout disabled)
- **Validation:** emotion-leakage-free implementation (see EMOTION_LEAKAGE_AUDIT.md)

### DQN Configuration

- Learning rate: 0.001
- Batch size: 64
- Buffer size: 100,000
- Network: [64, 64]
- Training budget: 50,000 steps
- Checkpoints: [10000, 20000, 30000, 40000, 50000]
- Seeds (n=20): `[42, 0, 1, 7, 13, 21, 99, 100, 200, 314, 500, 777, 1234, 2024, 5555, 8888, 9999, 10101, 20202, 30303]`

## Table 1 - Aggregate Metrics (Best Checkpoint, n=20)

| condition | n_seeds | learning_gain_mean | learning_gain_std | learning_gain_ci | final_knowledge_mean | success_rate_mean | adaptation_accuracy_mean | mean_episode_reward_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: Emotion-Aware DQN | 20 | 0.6089 | 0.0887 | [0.570, 0.648] | 0.7297 | 0.4003 | 0.7751 | 2.9041 |
| C: Knowledge-Only DQN (Strict No-Emotion) | 20 | 0.69 | 0.0131 | [0.684, 0.696] | 0.7856 | 0.5461 | 0.9536 | 0.6279 |

## Table 2 - Pairwise Comparison (A vs C, Welch t-test + Holm)

| metric | A_mean | C_mean | delta_A_minus_B | std_A | std_B | cohens_d | t | p_raw | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| learning_gain | 0.6089 | 0.69 | -0.0811 | 0.0887 | 0.0131 | -1.2791 | -4.0448 | 0.000643 | 0.001285 |
| final_knowledge | 0.7297 | 0.7856 | -0.0559 | 0.0611 | 0.0094 | -1.2788 | -4.044 | 0.00064 | 0.001919 |
| success_rate | 0.4003 | 0.5461 | -0.1458 | 0.1502 | 0.0241 | -1.3557 | -4.287 | 0.000361 | 0.001442 |
| adaptation_accuracy | 0.7751 | 0.9536 | -0.1785 | 0.2138 | 0.0772 | -1.1108 | -3.5128 | 0.001795 | 0.001795 |
| mean_episode_reward | 2.9041 | 0.6279 | 2.2761 | 0.3868 | 0.0243 | 8.3054 | 26.264 | 0.0 | 0.0 |

## Checkpoint Selection (Condition A)

| seed | selected_checkpoint | validation_learning_gain | final_eval_learning_gain | selected_eval_learning_gain | condition |
| --- | --- | --- | --- | --- | --- |
| 0 | 40000 | 0.6776 | 0.3603 | 0.695 | A |
| 1 | 20000 | 0.5822 | 0.3125 | 0.5944 | A |
| 7 | 30000 | 0.5998 | 0.5416 | 0.6137 | A |
| 13 | 40000 | 0.6865 | 0.3253 | 0.6796 | A |
| 21 | 40000 | 0.6454 | 0.2817 | 0.6758 | A |
| 42 | 50000 | 0.6286 | 0.6588 | 0.6588 | A |
| 99 | 40000 | 0.4487 | 0.4204 | 0.4374 | A |
| 100 | 50000 | 0.6665 | 0.6741 | 0.6741 | A |
| 200 | 30000 | 0.6733 | 0.672 | 0.6738 | A |
| 314 | 20000 | 0.4361 | 0.3854 | 0.4104 | A |
| 500 | 40000 | 0.6749 | 0.4934 | 0.6819 | A |
| 777 | 30000 | 0.5958 | 0.1661 | 0.5815 | A |
| 1234 | 50000 | 0.4611 | 0.4913 | 0.4913 | A |
| 2024 | 40000 | 0.6457 | 0.24 | 0.6361 | A |
| 5555 | 20000 | 0.6953 | 0.3692 | 0.6879 | A |
| 8888 | 50000 | 0.5484 | 0.592 | 0.592 | A |
| 9999 | 10000 | 0.5125 | 0.2539 | 0.5134 | A |
| 10101 | 10000 | 0.6999 | 0.375 | 0.6844 | A |
| 20202 | 30000 | 0.5275 | 0.4271 | 0.5267 | A |
| 30303 | 20000 | 0.6563 | 0.4989 | 0.6704 | A |

## Checkpoint Selection (Condition C)

| seed | selected_checkpoint | validation_learning_gain | final_eval_learning_gain | selected_eval_learning_gain | condition |
| --- | --- | --- | --- | --- | --- |
| 42 | 30000 | 0.6685 | 0.7 | 0.7011 | C |
| 0 | 10000 | 0.6791 | 0.5677 | 0.693 | C |
| 1 | 50000 | 0.6794 | 0.681 | 0.681 | C |
| 7 | 20000 | 0.6857 | 0.6341 | 0.6836 | C |
| 13 | 10000 | 0.7107 | 0.3901 | 0.6899 | C |
| 21 | 10000 | 0.6818 | 0.551 | 0.7152 | C |
| 99 | 20000 | 0.7027 | 0.3214 | 0.698 | C |
| 100 | 40000 | 0.6937 | 0.6876 | 0.711 | C |
| 200 | 10000 | 0.6835 | 0.6792 | 0.6795 | C |
| 314 | 30000 | 0.6679 | 0.5351 | 0.6764 | C |
| 500 | 30000 | 0.6788 | 0.3381 | 0.6901 | C |
| 777 | 30000 | 0.7231 | 0.6524 | 0.6963 | C |
| 1234 | 20000 | 0.6574 | 0.3393 | 0.6537 | C |
| 2024 | 10000 | 0.7007 | 0.5581 | 0.697 | C |
| 5555 | 30000 | 0.6686 | 0.6412 | 0.6799 | C |
| 8888 | 40000 | 0.6922 | 0.1772 | 0.6884 | C |
| 9999 | 10000 | 0.6778 | 0.6767 | 0.691 | C |
| 10101 | 30000 | 0.7267 | 0.0202 | 0.6986 | C |
| 20202 | 50000 | 0.6835 | 0.6881 | 0.6881 | C |
| 30303 | 20000 | 0.6902 | 0.6025 | 0.6885 | C |

## Research Questions

1. Does emotion-aware tutoring outperform knowledge-only? **False** (delta LG = -0.0811)
2. Effect size (Cohen's d): **-1.2791**
3. Statistically significant (Holm)? **True** (p = 0.0013)
4. Performance attributable to emotion: **-11.75%** relative LG gain
5. Emotional component justified? **False**

## Final Thesis Conclusion

**Knowledge Only matches or exceeds Emotion + Knowledge**

Emotion-aware tutoring (A) mean LG=0.6089 vs knowledge-only (C) LG=0.6900; delta=-0.0811; Cohen's d=-1.279; Holm p=0.0013 (significant); relative improvement=-11.8%

| Metric | A (Emotion-Aware) | C (Knowledge-Only) |
| --- | --- | --- |
| Learning Gain | 0.6089 [0.570, 0.648] | 0.6900 [0.684, 0.696] |

## Limitations

- Condition C uses knowledge-only reward (wk * delta_k); mean reward is not directly comparable.
- Adaptation accuracy in C uses pinned neutral emotion_id (diagnostic metric).
- Results are simulator-specific; 20-seed protocol matches frozen thesis model.
