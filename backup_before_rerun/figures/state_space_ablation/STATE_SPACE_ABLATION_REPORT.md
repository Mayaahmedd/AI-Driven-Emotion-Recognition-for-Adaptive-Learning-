# State-Space Ablation Report

## Observation Configurations

- **A:** 6-dim original [knowledge, engagement, frustration, confusion, boredom, emotion_id]
- **B:** 7-dim A + normalized consecutive_flag_steps
- **C:** 7-dim B with emotion_id masked (ablation only if redundant)

Gain ratio: **10:1** | Seeds: **10** | Eval: 500 episodes

## Results (Best Checkpoint)

| condition_label | n_seeds | learning_gain_mean | final_knowledge_mean | success_rate_mean | adaptation_accuracy_mean | mean_episode_reward_mean | dropout_rate_mean | action_diversity_mean | learning_gain_std | final_knowledge_std | success_rate_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: Original 6-dim | 10 | 0.5781 | 0.7095 | 0.3576 | 0.7505 | 3.0196 | 0.0 | 0.4617 | 0.1269 | 0.088 | 0.2008 |
| B: + Persistent Steps | 10 | 0.5503 | 0.6902 | 0.3326 | 0.6218 | 2.7505 | 0.0 | 0.4875 | 0.1445 | 0.0999 | 0.2104 |
| C: Persistent, No Emotion ID | 10 | 0.5571 | 0.6952 | 0.3226 | 0.6404 | 2.9708 | 0.0 | 0.5199 | 0.1013 | 0.0704 | 0.1659 |

## Pairwise A vs B (persistent channel)

- **learning_gain**: A=0.5781 vs B=0.5503, delta=+0.0278, p_holm=1.0000
- **final_knowledge**: A=0.7095 vs B=0.6902, delta=+0.0193, p_holm=1.0000
- **success_rate**: A=0.3576 vs B=0.3326, delta=+0.0250, p_holm=1.0000
- **adaptation_accuracy**: A=0.7505 vs B=0.6218, delta=+0.1287, p_holm=1.0000
- **mean_episode_reward**: A=3.0196 vs B=2.7505, delta=+0.2691, p_holm=1.0000
- **dropout_rate**: A=0.0000 vs B=0.0000, delta=+0.0000, p_holm=1.0000
- **action_diversity**: A=0.4617 vs B=0.4875, delta=-0.0259, p_holm=1.0000

## Pairwise B vs C (emotion_id redundancy)

- **learning_gain**: B=0.5503 vs C=0.5571, delta=-0.0068, p_holm=1.0000
- **final_knowledge**: B=0.6902 vs C=0.6952, delta=-0.0049, p_holm=1.0000
- **success_rate**: B=0.3326 vs C=0.3226, delta=+0.0100, p_holm=1.0000
- **adaptation_accuracy**: B=0.6218 vs C=0.6404, delta=-0.0186, p_holm=1.0000
- **mean_episode_reward**: B=2.7505 vs C=2.9708, delta=-0.2203, p_holm=1.0000
- **dropout_rate**: B=0.0000 vs C=0.0000, delta=+0.0000, p_holm=1.0000
- **action_diversity**: B=0.4875 vs C=0.5199, delta=-0.0324, p_holm=1.0000

## Recommendation

**Winner:** A: Original 6-dim
- LG: 0.578 (CI [0.499, 0.657], std=0.127)
- Persistent obs helps LG: **False**
- emotion_id redundant (B?C): **True**

persistent channel delta vs original=+0.0278 (not significant); emotion_id is redundant when persistent steps are present (B?C); Recommended: A: Original 6-dim

## Thesis Rationale

The agent must observe state variables that (1) affect transition dynamics or reward, (2) are not fully inferable from other observed channels, and (3) enable anticipatory action before irreversible termination. `consecutive_flag_steps` satisfies (1)-(3): it drives emergency masking, reward penalties, and dropout termination, yet was previously hidden from the Q-network. Normalizing by the dropout horizon gives the agent a time-to-crisis signal that raw frustration alone cannot provide once the persistent flag is latched.
