# Emotion Ablation Study Report

Gain ratio: **10:1** (G4 - thesis default)
Seeds: [42, 7, 13, 21, 99, 314]
Train timesteps (DQN): 50000
Eval episodes: 500

## Methodology (template)

We conducted a controlled ablation on **agent-facing observations** only. The simulator
continued to evolve full internal affect (engagement, frustration, confusion, boredom,
emotion_id); rewards, actions, and transition dynamics were unchanged.

**Condition A (full_emotion):** observation vector
`[knowledge, engagement, frustration, confusion, boredom, emotion_id]`.

**Condition B (knowledge_only):** indices 1-5 zeroed; agents see `[knowledge, 0, 0, 0, 0, 0]`.

**Optional single-channel ablations:** one affect dimension zeroed at a time to rank
contributions (DQN retrained per condition; ERT rules tied to that channel disabled).

Three tutors were compared: **Random** (observation-agnostic), **ERT** (rule tiers
masked to match each condition), **DQN** (retrained per condition with identical
hyperparameters). Evaluation used 500 episodes per seed across seeds
[42, 7, 13, 21, 99, 314] with gain ratio **10:1** (G4).

**Adaptation accuracy** is computed post hoc against `BEST_ACTION_MAP` using the
*true* simulator emotion_id (not the ablated observation), isolating pedagogical
matching from observation availability.


## Results Summary

| Condition | Tutor | Learning Gain | Final Knowledge | Success Rate | Adaptation | Wellbeing | Reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full Emotion (A) | Random | 0.3825 | 0.5748 | 0.0817 | 0.288 | 0.3696 | 2.7049 |
| Full Emotion (A) | ERT | 0.4108 | 0.5943 | 0.1157 | 0.3732 | 0.4018 | 3.1242 |
| Full Emotion (A) | DQN | 0.4362 | 0.6123 | 0.1997 | 0.5455 | 0.3887 | 2.9853 |
| Knowledge Only (B) | Random | 0.3825 | 0.5748 | 0.0817 | 0.288 | 0.3696 | 2.7049 |
| Knowledge Only (B) | ERT | 0.444 | 0.6169 | 0.1497 | 0.5234 | 0.4082 | 3.2436 |
| Knowledge Only (B) | DQN | 0.3908 | 0.5812 | 0.1233 | 0.3183 | 0.377 | 2.8138 |

## Primary Ablation: Full vs Knowledge Only

- **Random** [learning_gain]: full=0.3825 vs knowledge_only=0.3825, delta=+0.0000, d=0.000, p_holm=1.0000
- **Random** [final_knowledge]: full=0.5748 vs knowledge_only=0.5748, delta=+0.0000, d=0.000, p_holm=1.0000
- **Random** [success_rate]: full=0.0817 vs knowledge_only=0.0817, delta=+0.0000, d=0.000, p_holm=1.0000
- **Random** [adaptation_accuracy]: full=0.2880 vs knowledge_only=0.2880, delta=+0.0000, d=0.000, p_holm=1.0000
- **Random** [emotional_wellbeing]: full=0.3696 vs knowledge_only=0.3696, delta=+0.0000, d=0.000, p_holm=1.0000
- **Random** [mean_episode_reward]: full=2.7049 vs knowledge_only=2.7049, delta=+0.0000, d=0.000, p_holm=1.0000
- **Random** [dropout_rate]: full=0.0000 vs knowledge_only=0.0000, delta=+0.0000, d=0.000, p_holm=1.0000
- **ERT** [learning_gain]: full=0.4108 vs knowledge_only=0.4440, delta=-0.0332, d=-3.121, p_holm=0.0009
- **ERT** [final_knowledge]: full=0.5943 vs knowledge_only=0.6169, delta=-0.0226, d=-3.184, p_holm=0.0010
- **ERT** [success_rate]: full=0.1157 vs knowledge_only=0.1497, delta=-0.0340, d=-3.924, p_holm=0.0003
- **ERT** [adaptation_accuracy]: full=0.3732 vs knowledge_only=0.5234, delta=-0.1501, d=-50.558, p_holm=0.0000
- **ERT** [emotional_wellbeing]: full=0.4018 vs knowledge_only=0.4082, delta=-0.0064, d=-2.956, p_holm=0.0010
- **ERT** [mean_episode_reward]: full=3.1242 vs knowledge_only=3.2436, delta=-0.1194, d=-5.873, p_holm=0.0000
- **ERT** [dropout_rate]: full=0.0000 vs knowledge_only=0.0000, delta=+0.0000, d=0.000, p_holm=1.0000
- **DQN** [learning_gain]: full=0.4362 vs knowledge_only=0.3908, delta=+0.0455, d=0.255, p_holm=1.0000
- **DQN** [final_knowledge]: full=0.6123 vs knowledge_only=0.5812, delta=+0.0312, d=0.254, p_holm=1.0000
- **DQN** [success_rate]: full=0.1997 vs knowledge_only=0.1233, delta=+0.0763, d=0.351, p_holm=1.0000
- **DQN** [adaptation_accuracy]: full=0.5455 vs knowledge_only=0.3183, delta=+0.2272, d=0.869, p_holm=1.0000
- **DQN** [emotional_wellbeing]: full=0.3887 vs knowledge_only=0.3770, delta=+0.0117, d=0.323, p_holm=1.0000
- **DQN** [mean_episode_reward]: full=2.9853 vs knowledge_only=2.8138, delta=+0.1715, d=0.392, p_holm=1.0000
- **DQN** [dropout_rate]: full=0.0000 vs knowledge_only=0.0000, delta=+0.0000, d=0.000, p_holm=1.0000

## Research Questions

1. **Does emotion information improve learning gain?** Random d=+0.0000; ERT d=-0.0332*; DQN d=+0.0455
2. **Does emotion information improve adaptation accuracy?** Random d=+0.0000; ERT d=-0.1501*; DQN d=+0.2272
3. **Does emotion information improve success rate?** Random d=+0.0000; ERT d=-0.0340*; DQN d=+0.0763
4. **Which emotions contribute most?** See per-channel table / contribution CSV.
5. **Is affect-aware tutoring justified?** DQN learning gain +0.0455 with emotions

## Discussion (template)

Interpret full-vs-knowledge-only deltas jointly across tutors. A positive DQN delta on
learning gain supports that learned policies exploit affect channels; ERT deltas isolate
the value of hand-crafted affect rules. Random should show negligible deltas, validating
the ablation plumbing.

If adaptation accuracy drops sharply under knowledge_only while learning gain is stable,
agents may still learn via reward shaping without explicit emotion features - report both
metrics. Per-channel ablations rank which continuous signals DQN uses most; emotion_id-only
vs affect-only conditions separate discrete FER labels from dimensional affect.

Report effect sizes (Cohen's d) alongside p-values; Holm correction controls family-wise
error within each tutor's metric battery.


## Limitations (template)

- Ablation masks observations at the environment interface; it does not remove affect from
  latent simulator state or rewards (multi-objective reward still penalizes frustration).
- ERT knowledge_only retains only knowledge-tier rules (T4 + fallback); this is a strong
  baseline, not a naive random policy.
- DQN requires retraining per condition; sample efficiency differs from ERT/Random.
- Adaptation accuracy uses a fixed expert map - high accuracy does not guarantee learning
  gain if the map is misaligned with reward optima.
- Results are simulator-specific; human tutoring studies needed for external validity.
- Single-channel ablations are not fully orthogonal (affect dimensions co-evolve in transitions).

