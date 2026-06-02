# Emotion Update Equations — Diagnostic Audit

**Date:** 2026-05-31  
**Scope:** `student_model.py` AR(1) affect dynamics (engagement, frustration, confusion, boredom)  
**Code modified:** None (read-only audit + Monte Carlo simulation)

---

## 1. Equations Under Audit

### Core update (all four emotions)

```
e_{t+1} = clip01( ? · e_t + (1 ? ?) · target + N(0, EMOTION_NOISE_STD) )
```

| Emotion     | ? (persistence) | `(1 ? ?)` (target pull) | `EMOTION_NOISE_STD` |
|-------------|-----------------|-------------------------|---------------------|
| Frustration | 0.69            | 0.31                    | 0.03                |
| Engagement  | 0.60            | 0.40                    | 0.03                |
| Confusion   | 0.47            | 0.53                    | 0.03                |
| Boredom     | 0.36            | 0.64                    | 0.03                |

**Source:** `mdp_definition.py` defaults, read at runtime via `config.get_simulator_param()`.

### Target construction

1. **Mismatch baseline** (`_mismatch_effects`): `m = d ? k` selects flow / too-hard / too-easy target bands.
2. **Action overlay** (`ACTION_EFFECT_SPECS`): each action sets fixed targets or defers to mismatch (`None` / `"mismatch"`).
3. **Engagement scaling** (`_scale_engagement_target`):  
   `target_eng = clip01(target × (0.6 + 0.4 × engagement_recovery))`

Targets are in `[0, 1]`. Clipping after the update can amplify apparent step size near bounds (0 or 1).

---

## 2. Empirical Step-Change Statistics

**Method:** 500 students × 200 steps × 5 seeds = **500,000 transitions**, random actions, difficulty drift matching `student_env` (±0.1 on simplify/harder).

### 2.1 Average |?e| per step

| Emotion     | Mean \|?\| | Std of \|?\| |
|-------------|-----------|--------------|
| Engagement  | 0.033     | 0.028        |
| Frustration | 0.030     | 0.024        |
| Confusion   | 0.052     | 0.048        |
| Boredom     | 0.048     | 0.058        |

**Interpretation:** Under typical random-policy rollouts, mean step changes are modest (3–5%). Confusion and boredom move fastest on average because they have the lowest ? (highest target pull).

### 2.2 Maximum |?e| per step (Monte Carlo)

| Emotion     | Max \|?\| observed | % steps with \|?\| > 0.2 |
|-------------|-------------------|--------------------------|
| Engagement  | 0.310             | 0.06%                    |
| Frustration | 0.239             | 0.01%                    |
| Confusion   | 0.393             | 2.45%                    |
| Boredom     | 0.491             | 3.96%                    |

### 2.3 Trajectory standard deviation (per 200-step episode)

| Emotion     | Mean ?(trajectory) | Median ?(trajectory) |
|-------------|--------------------|----------------------|
| Engagement  | 0.044              | 0.043                |
| Frustration | 0.048              | 0.048                |
| Confusion   | 0.058              | 0.054                |
| Boredom     | 0.083              | 0.055                |

Boredom shows the widest episode-level spread (long right tail from low-? dynamics).

---

## 3. Analytical Bounds (Deterministic, Pre-Clip)

Worst-case deterministic step (current = 0, target = 1 or vice versa):

| Emotion     | Max deterministic \|?\| = `(1 ? ?)` | + 3? noise (0.03) | Exceeds 0.2? |
|-------------|---------------------------------------|-------------------|--------------|
| Frustration | 0.31                                  | 0.40              | **Yes**      |
| Engagement  | 0.40                                  | 0.49              | **Yes**      |
| Confusion   | 0.53                                  | 0.62              | **Yes**      |
| Boredom     | 0.64                                  | 0.73              | **Yes**      |

**All four emotions can change by more than 0.2 in a single step** when current and target are far apart. Boredom is structurally the most volatile (? = 0.36).

### Extreme-scenario Monte Carlo (grid over k, d, actions, corner affect states)

| Emotion     | Max \|?\| | Representative worst case |
|-------------|-----------|---------------------------|
| Engagement  | 0.401     | `no_action`, e: 0?0.40 (mismatch target from flow zone) |
| Frustration | 0.391     | `break`, f: 1.0?0.61 (target 0.1, high mismatch) |
| Confusion   | 0.614     | `explanation`, c: 1.0?0.39 (target 0.05, task too hard) |
| Boredom     | 0.691     | `break`, b: 1.0?0.31 (mismatch boredom target 0.1) |

These are rare but **physically reachable** under the current equations.

---

## 4. Per-Action Volatility Hotspots

Actions that most often produce \|?\| > 0.2 (random-policy rollouts):

| Action            | Dominant volatile dimension | Max \|?\| | % steps > 0.2 |
|-------------------|----------------------------|-----------|---------------|
| `harder_problem`  | Boredom                    | 0.491     | 9.8%          |
| `encouragement`   | Boredom                    | 0.482     | 7.7%          |
| `explanation`     | Confusion                  | 0.393     | 3.0%          |
| `no_action`       | Engagement                 | 0.310     | 0.2%          |
| `hint` / `scaffold` | Confusion / Boredom (mismatch pass-through) | ~0.44 | 1–2.4% |

**Pattern:** Actions with `None` targets defer to mismatch dynamics. When knowledge/difficulty shift mismatch bands between steps, boredom and confusion can jump sharply. `explanation` (confusion target = 0.05) creates large downward pulls from high confusion states.

---

## 5. Answer to Audit Questions

| # | Question | Finding |
|---|----------|---------|
| 1 | Average change per step | 0.030–0.052 (see §2.1); confusion highest |
| 2 | Maximum change per step | 0.239–0.491 typical MC; up to 0.691 in corner cases |
| 3 | Std dev of trajectories | 0.044–0.083 mean episode ? (see §2.3) |
| 4 | Can any emotion change > 0.2 in one step? | **Yes — all four**, analytically guaranteed when \|target ? current\| is large; observed in 0.01–4% of random-policy steps (confusion/boredom most frequent) |
| 5 | Lower-volatility recommendation | See §6 |

---

## 6. Recommended Lower-Volatility Coefficients

Three tiers, from least to most invasive. **No code changes applied yet.**

### Tier A — Moderate (recommended starting point)

Halve the target-pull rate and reduce noise. Preserves relative ordering of emotion persistence (boredom still fastest, frustration slowest).

| Parameter              | Current | Proposed |
|------------------------|---------|----------|
| `LAMBDA_ENGAGEMENT`    | 0.60    | **0.80** |
| `LAMBDA_FRUSTRATION`   | 0.69    | **0.845** |
| `LAMBDA_CONFUSION`     | 0.47    | **0.735** |
| `LAMBDA_BOREDOM`       | 0.36    | **0.68** |
| `EMOTION_NOISE_STD`    | 0.03    | **0.015** |

**Effect:** Max deterministic swing drops to ~0.15–0.32 (down from 0.31–0.64). Expected to cut >0.2 steps by roughly 60–80% in random rollouts.

### Tier B — Cap at ~0.20 max deterministic swing

Scale `(1 ? ?)` uniformly so the worst emotion (boredom) has max deterministic ? ? 0.20:

| Parameter              | Current | Proposed |
|------------------------|---------|----------|
| `LAMBDA_ENGAGEMENT`    | 0.60    | **0.80** |
| `LAMBDA_FRUSTRATION`   | 0.69    | **0.80** |
| `LAMBDA_CONFUSION`     | 0.47    | **0.80** |
| `LAMBDA_BOREDOM`       | 0.36    | **0.80** |
| `EMOTION_NOISE_STD`    | 0.03    | **0.01** |

**Effect:** Uniform persistence; all emotions share ? = 0.80. With noise = 0.01, 3? tail adds ~0.03 ? practical ceiling ? 0.23.

### Tier C — Conservative (cap ~0.15 max deterministic swing)

| Parameter              | Current | Proposed |
|------------------------|---------|----------|
| `LAMBDA_*` (all four)  | 0.36–0.69 | **0.85–0.90** (uniform or scaled) |
| `EMOTION_NOISE_STD`    | 0.03    | **0.01** |

**Effect:** Emotions become slow-moving; may under-represent acute affect shifts documented in D'Mello & Graesser (2012) and Pekrun CVT literature. Use only if RL training instability from affect volatility is confirmed.

### Additional levers (not ?/noise)

If Tier A is insufficient without over-dampening:

1. **Soften mismatch targets** — reduce band extremes in `_mismatch_effects` (e.g. 0.7 ? 0.55 for high-frustration target).
2. **Cap per-step target distance** — `target = current + clip(target_raw ? current, ??_max, +?_max)` with ?_max = 0.15.
3. **Widen flow band** — increase `MISMATCH_HIGH` / decrease `MISMATCH_LOW` magnitude to reduce band-switching jumps.

---

## 7. Volatility Ranking (Current System)

```
Boredom (?=0.36) > Confusion (?=0.47) > Engagement (?=0.60) > Frustration (?=0.69)
```

Primary risk: **mismatch-driven targets** combined with **low ?** for confusion/boredom, plus **Gaussian noise** and **clip01** boundary effects.

---

## 8. Artifacts

| File | Description |
|------|-------------|
| `emotion_update_audit_report.json` | Full numerical results, per-action breakdown, worst-case scenarios |
| `EMOTION_UPDATE_AUDIT.md` | This report |

**Simulation config:** 500 students, 200 steps, 5 seeds, random 8-action policy, population sampled per `population.generate_population()`.
