# State-Space Audit: Educational Tutoring DQN

## Executive Summary

This audit examined whether `emotion_id` is redundant, whether persistent-frustration
information should enter the observation vector, and which representation is
scientifically defensible for the thesis DQN.

**Key finding:** Persistent frustration (`consecutive_flag_steps`) drives action
masking, reward penalties, and episode termination but was **not observable** to the
agent. This is a partial observability defect. The fix adds a normalized urgency
channel at observation index 6.

**Prior evidence on `emotion_id`:** The existing 15-seed emotion-ID ablation
(`figures/emotion_id_ablation/`) shows `no_emotion_id` outperforms `full_emotion`
on learning gain (0.613 vs 0.580, best checkpoint) with lower seed variance
(0.094 vs 0.109). `emotion_id` does not significantly improve LG (p_holm=0.39).

---

## 1. How `emotion_id` Is Computed

| Mode | Source |
|------|--------|
| Simulation (default) | Threshold rules in `fer_adapter.emotion_from_state()` |
| Live FER | External discrete label via `EMOTION_TO_ID` |
| strict_off ablation | Pinned to 3 (engaged) |

Simulation derivation (priority order):

```
frustration > 0.6  ? frustrated (2)
confusion   > 0.5  ? confused   (0)
boredom     > 0.5  ? bored      (1)
else               ? engaged    (3)
```

Set **after** each transition in `StudentEnv.step()`, not by the student model.

---

## 2. Is `emotion_id` Redundant?

### Information-theoretic analysis

| Property | Continuous vars alone | emotion_id adds |
|----------|----------------------|-----------------|
| Frustrated state | frustration > 0.6 | Same threshold, but discretized |
| Priority ordering | Not recoverable | Collapses overlapping high affect |
| FER deployment | N/A | Independent discrete label |

In **simulation mode**, `emotion_id` is largely derivable from continuous affect but
uses **priority rules** that are not invertible (e.g., high frustration + high
confusion ? always "frustrated"). This is a lossy abstraction, not pure redundancy.

In **live FER mode**, `emotion_id` carries independent information.

### Experimental evidence (prior study)

From `EMOTION_ID_ABLATION_REPORT.md` (15 seeds, 50k train, best checkpoint):

| Condition | LG mean | LG std | Adaptation |
|-----------|---------|--------|------------|
| A: full_emotion | 0.580 | 0.109 | 0.746 |
| D: no_emotion_id | 0.613 | 0.094 | 0.819 |

**Conclusion:** Do not automatically remove `emotion_id`. In simulation, it is
*partially* redundant and empirically **harmful** to DQN (adds noise/variance).
Removal is justified for the thesis model but should be validated jointly with the
new persistent channel via the state-space ablation.

---

## 3. Persistent Frustration: Hidden State Defect

### Environment logic (`student_env.py`)

```
if frustration > 0.7 for 3 consecutive steps ? persistent_flag = True
if frustration < 0.5 ? flag cleared, consecutive_flag_steps = 0
while flag active ? consecutive_flag_steps += 1
if consecutive_flag_steps >= 5 ? dropout termination
```

### Effects without agent observation

| Mechanism | Uses flag/steps | Agent could see? (before fix) |
|-----------|-----------------|-------------------------------|
| Emergency action mask | flag | No |
| Reward penalty | flag | No |
| Dropout termination | consecutive_flag_steps | No |

This violates the MDP assumption that the agent observes sufficient state to predict
future rewards and termination.

### Design choice: `consecutive_flag_steps` over binary flag

A binary flag only signals "emergency mode on/off." Normalized consecutive steps
encode **urgency** (0.0 ? 0.2 ? ù ? 1.0 at dropout), enabling anticipatory
de-escalation before episode failure.

Normalization: `min(steps / PERSISTENT_FLAG_DROPOUT_STEPS, 1.0)`

---

## 4. Implementation

### Observation layouts

**v6 (legacy, backward compatible ù default):**
```
[knowledge, engagement, frustration, confusion, boredom, emotion_id]
```

**v7 (extended):**
```
[knowledge, engagement, frustration, confusion, boredom, emotion_id,
 normalized_consecutive_flag_steps]
```

### API

```python
# Legacy 6-dim (existing models)
env = StudentEnv(obs_ablation="full_emotion")

# 7-dim with persistent channel
env = StudentEnv(include_persistent_obs=True)

# Ablation presets
env = StudentEnv(obs_space_variant="B_v7_with_persistent")
env = StudentEnv(obs_space_variant="C_v7_persistent_no_eid")
```

### Model compatibility

```python
agent.load("models/old_6dim.zip")
agent.verify_obs_compat(env)  # raises if dim mismatch
```

Saved 6-dim models require `include_persistent_obs=False`. New 7-dim models require
`include_persistent_obs=True`.

---

## 5. Migration Notes

| Change | Impact |
|--------|--------|
| Default obs dim stays 6 | Existing training scripts unchanged |
| `include_persistent_obs=True` | New 7-dim models; retrain required |
| `StudentState.from_vec()` | Accepts 6- or 7-dim vectors (uses first 6) |
| `build_observation()` | Central obs builder in `mdp_definition.py` |
| CSV logs | Added `consecutive_flag_steps` field |
| Viewer | Compatible (reads CSV columns; new field optional) |

---

## 6. Ablation Protocol

Three conditions (`state_space_ablation_study.py`):

| ID | Variant | Dimensions |
|----|---------|------------|
| A | Original | 6 |
| B | + persistent steps | 7 |
| C | Persistent, no emotion_id | 7 (index 5 masked) |

Metrics: learning gain, adaptation accuracy, success rate, reward, dropout rate,
action diversity (Shannon entropy), seed variance.

Run:
```bash
export PYTHONPATH="$(pwd)"
python3 -m RL_Module.state_space_ablation_study --phase all
python3 -m RL_Module.state_space_ablation_study --phase all --quick  # smoke test
```

Results: `figures/state_space_ablation/STATE_SPACE_ABLATION_REPORT.md`

---

## 7. Thesis-Ready Recommendation Framework

The final observation space should satisfy:

1. **Sufficiency:** All variables affecting reward, termination, or masking are observed (or deliberately ablated with justification).
2. **Minimality:** No channel that empirically degrades LG or increases variance without compensating benefit.
3. **Anticipatory capacity:** Urgency signals (time-to-dropout) included when termination depends on them.

**Recommended decision tree:**

```
IF live FER deployment:
    USE v7 with emotion_id (independent discrete channel)
ELIF simulation (thesis):
    USE v7 WITHOUT emotion_id (Condition C)
    JUSTIFICATION: prior ablation + joint state-space ablation
ELSE:
    USE v6 for backward compatibility with saved checkpoints
```

See `figures/state_space_ablation/STATE_SPACE_ABLATION_REPORT.md` for full results.

### 10-seed results (50k train, 500 eval, best checkpoint)

| Condition | LG mean | LG std | Adaptation | Success | Reward | Diversity |
|-----------|---------|--------|------------|---------|--------|-----------|
| A: Original 6-dim | 0.578 | 0.127 | 0.751 | 0.358 | 3.02 | 0.462 |
| B: + Persistent | 0.550 | 0.145 | 0.622 | 0.333 | 2.75 | 0.488 |
| C: Persistent, no eid | 0.557 | **0.101** | 0.640 | 0.323 | 2.97 | 0.520 |

Pairwise (none significant after Holm correction):
- A vs B LG delta = +0.028 (p_holm=1.0)
- B vs C LG delta = ?0.007 (p_holm=1.0) ? **emotion_id redundant when persistent present**

Dropout rate = 0.0 for all conditions (trained agents avoid persistent-frustration termination).

### Synthesized recommendation

| Criterion | Winner | Evidence |
|-----------|--------|----------|
| Learning gain (this study) | A (marginal) | +0.028 vs B, not significant |
| Learning gain (prior 15-seed) | no_emotion_id | 0.613 vs 0.580, lower variance |
| Seed stability | C | LG std 0.101 vs 0.127 (A) |
| MDP correctness | B or C | Persistent channel fixes partial observability |
| emotion_id value | Remove (sim) | B?C; prior ablation favors removal |

**Final thesis observation space:**

```
[knowledge, engagement, frustration, confusion, boredom]           # v6, no emotion_id
```

with **optional** 7th channel for deployment:

```
[..., normalized_consecutive_flag_steps]                          # v7 extension
```

Use v6 `no_emotion_id` for the thesis DQN (strongest empirical evidence, 15 seeds).
Enable `include_persistent_obs=True` when dropout is non-negligible or for live
deployment where the agent must anticipate crisis termination. Condition C
(v7 + no emotion_id) is the principled upgrade if expanding to 7 dimensions.
