# Thesis Gain Sensitivity Report (Redesigned ERT)

ERT version: redesigned_t0_t6

## 1. Ranking Table (G4 = 10:1 default)

| Tutor | Learning Gain | Final Knowledge | Success Rate | Dropout | Adaptation | Reward | Wellbeing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 0.3825 | 0.5748 | 0.0817 | 0.0 | 0.288 | 2.7049 | 0.3696 |
| ERT | 0.4108 | 0.5943 | 0.1157 | 0.0 | 0.3732 | 3.1242 | 0.4018 |
| DQN | 0.4362 | 0.6123 | 0.1997 | 0.0 | 0.5455 | 2.9853 | 0.3887 |

## 2. Main Comparison (G4)

| Tutor | Learning Gain | Final Knowledge | Success Rate | Dropout | Adaptation Accuracy | Reward | Wellbeing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 0.3825 | 0.5748 | 0.0817 | 0.0 | 0.288 | 2.7049 | 0.3696 |
| ERT | 0.4108 | 0.5943 | 0.1157 | 0.0 | 0.3732 | 3.1242 | 0.4018 |
| DQN | 0.4362 | 0.6123 | 0.1997 | 0.0 | 0.5455 | 2.9853 | 0.3887 |

## 3. Sensitivity Table (Learning Gain by Ratio)

| Ratio | gain_id | Random_LG | ERT_LG | DQN_LG | Winner |
| --- | --- | --- | --- | --- | --- |
| 1:1 | G0 | 0.5238 | 0.5293 | 0.4311 | ERT |
| 2:1 | G1 | 0.4465 | 0.4648 | 0.543 | DQN |
| 3.3:1 | G2 | 0.4159 | 0.4382 | 0.3716 | ERT |
| 5:1 | G3 | 0.3987 | 0.4241 | 0.4923 | DQN |
| 10:1 | G4 | 0.3825 | 0.4108 | 0.4362 | DQN |
| 20:1 | G5 | 0.3741 | 0.4034 | 0.4955 | DQN |

## 4. Ranking Stability

- Strict Random < ERT < DQN: 2.8% (1/36)
- ERT > Random: 97.2%
- DQN > ERT: 47.2%

## 5. Pairwise Tests at G4 (Holm-corrected)

- **ERT vs Random** [learning_gain]: 0.4108 vs 0.3825, d=2.841, p_holm=0.0120
- **ERT vs Random** [final_knowledge]: 0.5943 vs 0.5748, d=3.007, p_holm=0.0086
- **ERT vs Random** [success_rate]: 0.1157 vs 0.0817, d=3.883, p_holm=0.0013
- **ERT vs Random** [dropout_rate]: 0.0000 vs 0.0000, d=0.000, p_holm=1.0000
- **ERT vs Random** [adaptation_accuracy]: 0.3732 vs 0.2880, d=24.153, p_holm=0.0000
- **ERT vs Random** [emotional_wellbeing]: 0.4018 vs 0.3696, d=12.194, p_holm=0.0000
- **ERT vs Random** [mean_episode_reward]: 3.1242 vs 2.7049, d=13.099, p_holm=0.0000
- **DQN vs ERT** [learning_gain]: 0.4362 vs 0.4108, d=0.180, p_holm=1.0000
- **DQN vs ERT** [final_knowledge]: 0.6123 vs 0.5943, d=0.186, p_holm=1.0000
- **DQN vs ERT** [success_rate]: 0.1997 vs 0.1157, d=0.520, p_holm=1.0000
- **DQN vs ERT** [dropout_rate]: 0.0000 vs 0.0000, d=0.000, p_holm=1.0000
- **DQN vs ERT** [adaptation_accuracy]: 0.5455 vs 0.3732, d=0.758, p_holm=1.0000
- **DQN vs ERT** [emotional_wellbeing]: 0.3887 vs 0.4018, d=-0.883, p_holm=1.0000
- **DQN vs ERT** [mean_episode_reward]: 2.9853 vs 3.1242, d=-1.318, p_holm=0.9712
- **DQN vs Random** [learning_gain]: 0.4362 vs 0.3825, d=0.381, p_holm=1.0000
- **DQN vs Random** [final_knowledge]: 0.6123 vs 0.5748, d=0.388, p_holm=1.0000
- **DQN vs Random** [success_rate]: 0.1997 vs 0.0817, d=0.730, p_holm=1.0000
- **DQN vs Random** [dropout_rate]: 0.0000 vs 0.0000, d=0.000, p_holm=1.0000
- **DQN vs Random** [adaptation_accuracy]: 0.5455 vs 0.2880, d=1.133, p_holm=1.0000
- **DQN vs Random** [emotional_wellbeing]: 0.3887 vs 0.3696, d=1.285, p_holm=0.9692
- **DQN vs Random** [mean_episode_reward]: 2.9853 vs 2.7049, d=2.592, p_holm=0.0694

## 6. Thesis Conclusion

1. **Best educational tutor (G4):** DQN
2. **Safest emotionally (lowest dropout):** Random
3. **DQN significantly outperforms ERT (primary):** False
4. **ERT competitive (beats Random ?80%):** True
5. **10:1 ratio winner (learning gain):** DQN
6. **Conclusions robust across ratios:** True
