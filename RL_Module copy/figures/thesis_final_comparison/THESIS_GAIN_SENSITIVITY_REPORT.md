# Thesis Gain Sensitivity Report

## 1. Ranking Table (G4 = 10:1 default)

| Tutor | Learning Gain | Final Knowledge | Success Rate | Dropout | Adaptation | Reward | Wellbeing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 0.3855 | 0.5726 | 0.0729 | 0.0 | 0.2872 | 2.6977 | 0.3687 |
| ERT | 0.2514 | 0.4831 | 0.0275 | 0.0 | 0.2548 | 3.0454 | 0.409 |
| DQN | 0.4652 | 0.6277 | 0.2418 | 0.0 | 0.5601 | 2.9721 | 0.3855 |

## 2. Sensitivity Table (Learning Gain by Ratio)

| Ratio | gain_id | Random_LG | ERT_LG | DQN_LG | Winner |
| --- | --- | --- | --- | --- | --- |
| 1:1 | G0 | nan | nan | nan | None |
| 2:1 | G1 | nan | nan | nan | None |
| 3.3:1 | G2 | nan | nan | nan | None |
| 5:1 | G3 | nan | nan | nan | None |
| 10:1 | G4 | 0.3855 | 0.2514 | 0.4652 | DQN |
| 20:1 | G5 | nan | nan | nan | None |

## 3. Ranking Stability

- Strict Random < ERT < DQN: 20.0% (2/10)
- ERT > Random: 0.0%
- DQN > ERT: 80.0%

## 4. Pairwise Tests at G4 (Holm-corrected)

- **ERT vs Random** [learning_gain]: 0.2514 vs 0.3855, d=-20.649, p_holm=0.0000
- **ERT vs Random** [final_knowledge]: 0.4831 vs 0.5726, d=-19.589, p_holm=0.0000
- **ERT vs Random** [success_rate]: 0.0275 vs 0.0729, d=-6.729, p_holm=0.0000
- **ERT vs Random** [dropout_rate]: 0.0000 vs 0.0000, d=0.000, p_holm=1.0000
- **ERT vs Random** [adaptation_accuracy]: 0.2548 vs 0.2872, d=-10.556, p_holm=0.0000
- **ERT vs Random** [emotional_wellbeing]: 0.4090 vs 0.3687, d=25.081, p_holm=0.0000
- **ERT vs Random** [mean_episode_reward]: 3.0454 vs 2.6977, d=17.027, p_holm=0.0000
- **DQN vs ERT** [learning_gain]: 0.4652 vs 0.2514, d=1.560, p_holm=0.0956
- **DQN vs ERT** [final_knowledge]: 0.6277 vs 0.4831, d=1.519, p_holm=0.0947
- **DQN vs ERT** [success_rate]: 0.2418 vs 0.0275, d=1.296, p_holm=0.1761
- **DQN vs ERT** [dropout_rate]: 0.0000 vs 0.0000, d=0.000, p_holm=1.0000
- **DQN vs ERT** [adaptation_accuracy]: 0.5601 vs 0.2548, d=1.323, p_holm=0.1760
- **DQN vs ERT** [emotional_wellbeing]: 0.3855 vs 0.4090, d=-1.536, p_holm=0.0956
- **DQN vs ERT** [mean_episode_reward]: 2.9721 vs 3.0454, d=-0.447, p_holm=1.0000
- **DQN vs Random** [learning_gain]: 0.4652 vs 0.3855, d=0.582, p_holm=1.0000
- **DQN vs Random** [final_knowledge]: 0.6277 vs 0.5726, d=0.579, p_holm=1.0000
- **DQN vs Random** [success_rate]: 0.2418 vs 0.0729, d=1.022, p_holm=0.3372
- **DQN vs Random** [dropout_rate]: 0.0000 vs 0.0000, d=0.000, p_holm=1.0000
- **DQN vs Random** [adaptation_accuracy]: 0.5601 vs 0.2872, d=1.182, p_holm=0.2407
- **DQN vs Random** [emotional_wellbeing]: 0.3855 vs 0.3687, d=1.093, p_holm=0.2943
- **DQN vs Random** [mean_episode_reward]: 2.9721 vs 2.6977, d=1.667, p_holm=0.0679

## 5. Thesis Conclusion

1. **Best educational tutor (G4):** DQN
2. **Safest emotionally (lowest dropout):** Random
3. **DQN significantly outperforms ERT (primary):** False
4. **ERT competitive (beats Random ?80%):** False
5. **10:1 ratio winner (learning gain):** DQN
6. **Conclusions robust across ratios:** False
