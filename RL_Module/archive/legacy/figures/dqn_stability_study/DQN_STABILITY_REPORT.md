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
| 50000 | 5 | 0.4502 | 0.1368 | 0.3039 | [0.330, 0.570] | 0.1748 | 0.562 | 3.3081 |

## Hyperparameter Screen (std ranking)

| label | n_seeds | learning_gain_mean | learning_gain_std | learning_gain_cv | learning_gain_ci | success_rate_mean | adaptation_mean | reward_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_eps_0.02 | 10 | 0.4325 | 0.1027 | 0.2374 | [0.369, 0.496] | 0.1616 | 0.5137 | 3.1958 |
| target_upd_1000 | 10 | 0.4346 | 0.109 | 0.2509 | [0.367, 0.502] | 0.1558 | 0.5186 | 3.2978 |
| baseline | 10 | 0.4292 | 0.1098 | 0.2557 | [0.361, 0.497] | 0.1556 | 0.5093 | 3.2846 |
| expl_frac_0.3 | 10 | 0.4292 | 0.1098 | 0.2557 | [0.361, 0.497] | 0.1556 | 0.5093 | 3.2846 |
| final_eps_0.05 | 10 | 0.4292 | 0.1098 | 0.2557 | [0.361, 0.497] | 0.1556 | 0.5093 | 3.2846 |
| net_arch_64-64 | 10 | 0.4292 | 0.1098 | 0.2557 | [0.361, 0.497] | 0.1556 | 0.5093 | 3.2846 |
| target_upd_2000 | 10 | 0.4292 | 0.1098 | 0.2557 | [0.361, 0.497] | 0.1556 | 0.5093 | 3.2846 |
| net_arch_128-128-128 | 10 | 0.4141 | 0.1249 | 0.3015 | [0.337, 0.491] | 0.1554 | 0.4945 | 3.2145 |
| expl_frac_0.5 | 10 | 0.4391 | 0.1256 | 0.2861 | [0.361, 0.517] | 0.182 | 0.5397 | 3.3394 |
| expl_frac_0.7 | 10 | 0.4334 | 0.1258 | 0.2903 | [0.355, 0.511] | 0.166 | 0.503 | 3.2578 |

## Recommendation

**Best config:** checkpoint_sel (source: checkpoint)
- Mean LG: 0.5409020500557655
- Std LG: 0.10096464402637806
- CV: 0.18665975478549005
- Std reduction vs baseline: 26.2%

## Thesis Conclusion

Report the recommended configuration as the final thesis DQN. Prefer configs that
reduce seed std while maintaining or improving mean learning gain. Checkpoint selection
addresses seed-specific convergence timing without changing the underlying MDP.

## Discussion

High DQN variance often reflects exploration schedule mismatch and early stopping
before Q-value convergence. Longer training and validation checkpoint selection are
standard SB3 mitigations that do not alter the tutoring problem definition.
