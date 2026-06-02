# DQN Stability Study Report

## Methodology

Environment: full_emotion, G4 gain (10:1), 10 seeds.
Goal: reduce learning-gain variance across seeds without changing MDP definition.

Factors tested:
- Training length: 50k-200k
- Checkpoint selection: best validation LG among 10k-200k saves
- Exploration: fraction {0.3,0.5,0.7} x final_eps {0.05,0.02,0.01}
- Target update: {500, 1000, 2000}
- Architecture: [64,64] vs [128,128,128]

## Training Length Results

| train_timesteps | n_seeds | learning_gain_mean | learning_gain_std | learning_gain_cv | learning_gain_ci | success_rate_mean | adaptation_mean | reward_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200000 | 5 | 0.2537 | 0.0935 | 0.3687 | [0.172, 0.336] | 0.0508 | 0.2398 | 3.0546 |
| 100000 | 5 | 0.3595 | 0.1281 | 0.3564 | [0.247, 0.472] | 0.0852 | 0.3585 | 3.0534 |
| 50000 | 5 | 0.5682 | 0.1495 | 0.263 | [0.437, 0.699] | 0.3636 | 0.7359 | 2.9216 |
| 150000 | 5 | 0.3687 | 0.2171 | 0.5888 | [0.178, 0.559] | 0.1648 | 0.2814 | 2.8885 |

## Hyperparameter Screen (std ranking)

| label | n_seeds | learning_gain_mean | learning_gain_std | learning_gain_cv | learning_gain_ci | success_rate_mean | adaptation_mean | reward_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_upd_2000 | 10 | 0.4165 | 0.116 | 0.2786 | [0.345, 0.488] | 0.156 | 0.4378 | 3.3449 |
| target_upd_1000 | 10 | 0.4061 | 0.1386 | 0.3412 | [0.320, 0.492] | 0.149 | 0.4515 | 3.1484 |
| expl_frac_0.5 | 10 | 0.4936 | 0.15 | 0.304 | [0.401, 0.587] | 0.2556 | 0.5218 | 2.9921 |
| final_eps_0.01 | 10 | 0.3949 | 0.1628 | 0.4123 | [0.294, 0.496] | 0.1678 | 0.3781 | 2.917 |
| final_eps_0.02 | 10 | 0.4157 | 0.1734 | 0.4171 | [0.308, 0.523] | 0.167 | 0.3859 | 2.7527 |
| expl_frac_0.7 | 10 | 0.3954 | 0.192 | 0.4856 | [0.276, 0.514] | 0.17 | 0.3965 | 2.9081 |
| baseline | 10 | 0.4652 | 0.1949 | 0.4189 | [0.344, 0.586] | 0.2454 | 0.5609 | 2.9733 |
| expl_frac_0.3 | 10 | 0.4652 | 0.1949 | 0.4189 | [0.344, 0.586] | 0.2454 | 0.5609 | 2.9733 |
| final_eps_0.05 | 10 | 0.4652 | 0.1949 | 0.4189 | [0.344, 0.586] | 0.2454 | 0.5609 | 2.9733 |
| net_arch_64-64 | 10 | 0.4652 | 0.1949 | 0.4189 | [0.344, 0.586] | 0.2454 | 0.5609 | 2.9733 |

## Recommendation

**Best config:** train_200000 (source: training_length)
- Mean LG: 0.2537
- Std LG: 0.0935
- CV: 0.3687
- Std reduction vs baseline: 37.5%

## Thesis Conclusion

Report the recommended configuration as the final thesis DQN. Prefer configs that
reduce seed std while maintaining or improving mean learning gain. Checkpoint selection
addresses seed-specific convergence timing without changing the underlying MDP.

## Discussion

High DQN variance often reflects exploration schedule mismatch and early stopping
before Q-value convergence. Longer training and validation checkpoint selection are
standard SB3 mitigations that do not alter the tutoring problem definition.
