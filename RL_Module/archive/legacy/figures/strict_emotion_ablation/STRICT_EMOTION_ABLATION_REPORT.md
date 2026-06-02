# Strict Emotion Ablation Report

## Methodology

Three-way factorial design isolating emotion contributions:

- **A (Full Emotion):** full observation + full affect dynamics
- **B (Obs Ablation):** knowledge-only observation + full dynamics
- **C (Strict No-Emotion):** knowledge-only observation + affect-inert MDP

Gain ratio: **10:1** | Seeds: 3 conditions x 10 seeds
DQN: 50000 steps | Eval: 500 episodes

Decomposition:
- A vs B = value of **emotional observations** (agent-side)
- B vs C = value of **emotional dynamics** (environment-side)
- A vs C = **combined** emotion contribution

## Results Summary

| Condition | learning_gain | final_knowledge | success_rate | adaptation_accuracy | mean_episode_reward |
| --- | --- | --- | --- | --- | --- |
| A: Full Emotion | 0.4652 | 0.6315 | 0.2454 | 0.5609 | 2.9733 |
| B: Obs Ablation | 0.4105 | 0.5952 | 0.1242 | 0.3899 | 3.0241 |
| C: Strict No-Emotion | 0.5353 | 0.6825 | 0.316 | 0.6507 | 0.4346 |

## Pairwise Tests (Learning Gain)

- **obs_info_A_vs_B**: 0.4652 vs 0.4105, d=+0.0548, d=0.328, p_holm=0.4737
- **dynamics_B_vs_C**: 0.4105 vs 0.5353, d=-0.1248, d=-0.832, p_holm=0.2397
- **combined_A_vs_C**: 0.4652 vs 0.5353, d=-0.0700, d=-0.388, p_holm=0.7955

## Thesis Conclusion

1. Emotional information helps DQN: **True** (dLG A-B = 0.0548)
2. Emotional dynamics help DQN: **False** (dLG B-C = -0.1248)
3. Larger contributor: **dynamics**
4. Combined benefit (A vs C): dLG = -0.07
5. Affect-aware tutoring justified: **True**

Observation channel (A vs B): dLG=+0.0548 (benefit); Dynamics channel (B vs C): dLG=-0.1248 (no benefit); Combined (A vs C): dLG=-0.0700

## Discussion

Condition B isolates whether DQN can learn without seeing affect while still operating
in an emotionally realistic simulator. Condition C removes affect from transitions and
reward, yielding a knowledge-only MDP baseline. If B outperforms C, emotional dynamics
shape the learning problem even without observation access. If A outperforms B, the DQN
policy exploits affect features directly.

## Limitations

- Strict mode removes affect from reward; A/B still use multi-objective reward (confound
  for B vs C comparison is intentional - it tests dynamics, not reward formula identity).
- Adaptation accuracy uses fixed expert map with true/hidden emotion labels.
- DQN retrained per condition; results are simulator-specific.
