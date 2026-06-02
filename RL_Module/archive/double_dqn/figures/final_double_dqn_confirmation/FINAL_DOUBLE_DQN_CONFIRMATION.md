# Final Double DQN Confirmation (20 Seeds)

## Protocol

- **Standard DQN:** frozen thesis baseline (`baseline_thesis`, no retrain)
- **Double DQN:** same MDP, hyperparameters, checkpoint selection
- **Seeds (n=20):** `[0, 1, 7, 13, 21, 42, 99, 100, 200, 314, 500, 777, 1234, 2024, 5555, 8888, 9999, 10101, 20202, 30303]`
- **Training:** 50,000 steps | **Eval:** 500 episodes
- **Hyperparameters:** `{'learning_rate': 0.001, 'batch_size': 64, 'buffer_size': 100000, 'target_update_interval': 500, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05, 'net_arch': [64, 64]}`
- **Observation:** `no_emotion_id` (knowledge + 4 affect dims, no emotion_id)

## Focus Metrics (Best Checkpoint)

| System | Learning Gain | CV | Success Rate | Adaptation Accuracy |
| --- | --- | --- | --- | --- |
| Standard DQN | 0.6089 +/- 0.0887 [0.570, 0.648] | 0.1456 | 0.4003 +/- 0.1502 | 0.7751 +/- 0.2138 |
| Double DQN | 0.5539 +/- 0.0653 [0.525, 0.583] | 0.1178 | 0.3021 +/- 0.1039 | 0.7141 +/- 0.1617 |

## Pairwise Tests (Double DQN vs Standard DQN)

| Metric | Delta (DDQN-DQN) | % Change | CV DQN | CV DDQN | p-value | Significant |
| --- | --- | --- | --- | --- | --- | --- |
| learning_gain | -0.0550 | -9.0% | 0.1456 | 0.1178 | 0.0320 | yes |
| success_rate | -0.0982 | -24.5% | - | - | 0.0218 | yes |
| adaptation_accuracy | -0.0610 | -7.9% | - | - | 0.3156 | no |

## Verdict

- Double DQN higher learning gain: **False** (-9.0%)
- Double DQN lower CV: **True**
- Double DQN higher success rate: **False**
- Double DQN higher adaptation accuracy: **False**
- Focus-metric wins (DDQN): **1/4**

**Recommendation:** Retain standard DQN as the final thesis system.
