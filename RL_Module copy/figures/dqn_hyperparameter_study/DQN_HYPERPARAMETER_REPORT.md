# DQN Hyperparameter Optimization Report
## 1. Baseline (5 seeds)
| Metric | Mean | 95% CI |
|--------|------|--------|
| Mean reward | 2.7580 | [2.2344, 3.2816] |
| Success rate | 0.2382 | [0.0438, 0.4326] |
| Learning gain | 0.4462 | [0.2561, 0.6363] |
| Final knowledge | 0.6149 | [0.4843, 0.7455] |
| Adaptation accuracy | 0.3536 | [0.1802, 0.5269] |

## 2-3. Grid search leaders (screening seed)

### Best reward
- config_id: `e6f464edf368`
- reward=3.655, mastery=0.623, adapt=0.551

### Best mastery
- config_id: `dbda7f1166a6`
- reward=1.876, mastery=0.785, adapt=0.456

### Best balanced
- config_id: `3d833752cc9e`
- reward=2.904, mastery=0.770, adapt=0.996

## 5. Robustness (top 5)
```
   config_id  learning_rate  batch_size  buffer_size  target_update_interval  exploration_fraction  exploration_final_eps  mean_episode_reward_mean  mean_episode_reward_std                                        mean_episode_reward_ci_95  reward_vs_baseline_p  reward_vs_baseline_delta  success_rate_mean  success_rate_std                                                   success_rate_ci_95  learning_gain_mean  learning_gain_std                                               learning_gain_ci_95  final_knowledge_mean  final_knowledge_std                                            final_knowledge_ci_95  adaptation_accuracy_mean  adaptation_accuracy_std                                         adaptation_accuracy_ci_95  stable_improvement
3d833752cc9e         0.0003          64        50000                    1000                   0.2                   0.05                  3.174575                 0.298548 [np.float64(2.9128857683072966), np.float64(3.4362635494273728)]              0.259035                  0.416565             0.1690          0.191472 [np.float64(0.0011674534543433535), np.float64(0.33683254654565664)]            0.437420           0.154452 [np.float64(0.30203610031535877), np.float64(0.5728029233242784)]              0.610266             0.107115 [np.float64(0.5163761418332671), np.float64(0.7041567611453099)]                  0.549741                 0.273257 [np.float64(0.31022118448264296), np.float64(0.7892614055006726)]               False
467106ecc1c7         0.0003          64       100000                    1000                   0.2                   0.05                  3.174575                 0.298548 [np.float64(2.9128857683072966), np.float64(3.4362635494273728)]              0.259035                  0.416565             0.1690          0.191472 [np.float64(0.0011674534543433535), np.float64(0.33683254654565664)]            0.437420           0.154452 [np.float64(0.30203610031535877), np.float64(0.5728029233242784)]              0.610266             0.107115 [np.float64(0.5163761418332671), np.float64(0.7041567611453099)]                  0.549741                 0.273257 [np.float64(0.31022118448264296), np.float64(0.7892614055006726)]               False
4c0cc82b2166         0.0010          64       100000                     500                   0.3                   0.05                  2.929507                 0.299393   [np.float64(2.667077602848936), np.float64(3.191936866046402)]              0.642872                  0.171498             0.3616          0.233314     [np.float64(0.15709149236278708), np.float64(0.566108507637213)]            0.567328           0.150902 [np.float64(0.43505653475559214), np.float64(0.6995989016546758)]              0.698209             0.106006 [np.float64(0.6052904673550781), np.float64(0.7911274229638865)]                  0.734436                 0.300654  [np.float64(0.4709006105232792), np.float64(0.9979710670953552)]               False
501d9476d887         0.0005          32        50000                     500                   0.3                   0.01                  3.128295                 0.226478  [np.float64(2.929777745137717), np.float64(3.3268115769512057)]              0.259972                  0.370285             0.3146          0.167381   [np.float64(0.16788434433912652), np.float64(0.46131565566087346)]            0.549559           0.103875  [np.float64(0.4585084343098747), np.float64(0.6406097025483068)]              0.686098             0.073116 [np.float64(0.6220093552919957), np.float64(0.7501870264088656)]                  0.654123                 0.284934  [np.float64(0.4043669797599734), np.float64(0.9038788638188149)]               False
b1a2d48ae992         0.0010          64        50000                     500                   0.3                   0.05                  2.929507                 0.299393   [np.float64(2.667077602848936), np.float64(3.191936866046402)]              0.642872                  0.171498             0.3616          0.233314     [np.float64(0.15709149236278708), np.float64(0.566108507637213)]            0.567328           0.150902 [np.float64(0.43505653475559214), np.float64(0.6995989016546758)]              0.698209             0.106006 [np.float64(0.6052904673550781), np.float64(0.7911274229638865)]                  0.734436                 0.300654  [np.float64(0.4709006105232792), np.float64(0.9979710670953552)]               False
```


## 4. Educational validity
- **best_reward**: better_personalization_and_challenge, meaningful=True
- **best_mastery**: better_personalization_and_challenge, meaningful=True
- **best_balanced**: better_personalization_and_challenge, meaningful=False

## 6. Recommendation
```json
{
  "learning_rate": 0.001,
  "batch_size": 64,
  "buffer_size": 100000,
  "target_update_interval": 500,
  "exploration_fraction": 0.3,
  "exploration_final_eps": 0.05
}
```
