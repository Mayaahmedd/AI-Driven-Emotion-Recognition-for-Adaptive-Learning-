# ADR-003 — PPO Concept-Level Credit Assignment

**Status:** Accepted
**Owner:** RL Architect
**Date:** 2026-05-11

---

## 1. Context

PPO is used as the **high-level curriculum manager**: it selects an *instruction type* (worked example / problem / explanation / review) and a *concept/skill* to teach next. PPO decisions fire at **concept boundaries** — when the current concept's mastery exceeds threshold $\tau_m$, or after a hard cap of $N_{\max}$ items on the same concept.

Between two consecutive PPO decisions, the **DQN** runs many meso-step decisions (content variant, pace, UI, emotional support) and the **bandit** fires zero or more micro-interventions. The PPO controller therefore operates on a temporally coarser timeline than the environment ticks.

The question is: **what does PPO see as one transition, and which rewards should it credit to its decision?**

The naive solution — let PPO use raw env-step transitions like DQN — is wrong, because the macro action is held constant across many env steps, so the apparent state-action transition at every env step misrepresents what PPO actually controls.

## 2. Decision

### 2.1 PPO operates on a *macro-transition*

We define a macro-transition as

$$
\tau^{\text{macro}}_k = (s_{t_k}, a^{\text{macro}}_k, R_k, s_{t_{k+1}}, d_k),
$$

where $t_k$ and $t_{k+1}$ are the env-step indices of two consecutive concept boundaries, $a^{\text{macro}}_k$ is the macro action chosen at $t_k$, and $R_k$ is the **discounted sum of meso-step rewards inside the segment**:

$$
R_k = \sum_{t = t_k}^{t_{k+1}-1} \gamma^{(t - t_k)} \cdot r_t.
$$

The discount factor inside the segment is the same $\gamma$ used by DQN; this keeps the time-value of reward consistent across controllers.

### 2.2 Macro-value bootstrap

We **do not** train PPO's critic on the segment-return alone. We use the standard PPO advantage based on macro-step bootstrapping:

$$
V^{\text{macro}}(s_{t_k}) \approx R_k + \gamma^{(t_{k+1} - t_k)} \cdot V^{\text{macro}}(s_{t_{k+1}}).
$$

The exponent $(t_{k+1} - t_k)$ on the bootstrap discount is critical — a segment that took 12 env steps depreciates more than a segment that took 4 env steps. This makes the macro-value comparable across segments of different length.

### 2.3 GAE on macro-transitions

Generalized Advantage Estimation (Schulman et al., 2016) is applied to the macro-transition sequence with parameters $\gamma^{\text{macro}}$ and $\lambda^{\text{macro}}$, where

$$
\gamma^{\text{macro}}_k = \gamma^{(t_{k+1} - t_k)}.
$$

The macro-TD error is

$$
\delta^{\text{macro}}_k = R_k + \gamma^{\text{macro}}_k \cdot V(s_{t_{k+1}}) - V(s_{t_k}),
$$

and the macro advantage is

$$
\hat A^{\text{macro}}_k = \sum_{l \geq 0} (\gamma^{\text{macro}}_{k+l} \lambda^{\text{macro}})^l \cdot \delta^{\text{macro}}_{k+l}.
$$

Advantages are normalized **per batch** (subtract batch mean, divide by batch std + $10^{-8}$) before being used in the clipped surrogate.

### 2.4 Clipped surrogate

Standard PPO clip:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}\Big[\min\big(\rho_k(\theta) \hat A^{\text{macro}}_k,\; \mathrm{clip}(\rho_k(\theta), 1-\epsilon, 1+\epsilon) \hat A^{\text{macro}}_k\big)\Big],
$$

with $\rho_k(\theta) = \pi_\theta(a^{\text{macro}}_k \mid s_{t_k}) / \pi_{\theta_{\text{old}}}(a^{\text{macro}}_k \mid s_{t_k})$.

Default $\epsilon = 0.2$. Entropy bonus is added with a small coefficient $c_e$ (default $0.01$) annealed linearly to $0$ over training.

### 2.5 Reward decomposition is preserved

The segment return $R_k$ is computed from the **composite** meso reward, but for logging and explainability we also accumulate per-component segment returns (`R_perf_k`, `R_flow_k`, `R_cost_k`). These are NOT used for the PPO loss — they are logged to TensorBoard for attribution.

### 2.6 Termination & timeout handling

- If the episode terminates inside a segment (`done=True` at some $t \in [t_k, t_{k+1})$), we **close the macro-transition early**: $t_{k+1} \leftarrow t + 1$, and the bootstrap value $V(s_{t_{k+1}})$ is set to $0$.
- If a session times out without reaching $\tau_m$ on the current concept, the dangling segment is still emitted with the bootstrap value of the final state. This is critical: dropping the dangling segment biases PPO toward "fast mastery" concepts.

### 2.7 Macro-action distribution: factorized categorical

The macro action is a tuple `(instruction_type, concept_id)`. We use a **factorized categorical**:

$$
\pi_\theta(a^{\text{macro}} \mid s) = \pi^{\text{inst}}_\theta(a^{\text{inst}} \mid s) \cdot \pi^{\text{concept}}_\theta(a^{\text{concept}} \mid s, a^{\text{inst}}),
$$

with concept conditioned on instruction-type so that, e.g., "review" can only target already-introduced concepts.

The action mask (ADR cross-ref: safety layer) is multiplied into the logits **before** the softmax for *both* heads. Entropy is computed as the sum of the two head entropies — this is correct only because we factorize.

## 3. Rationale

- **Segment-level transitions** match what PPO actually controls. Treating raw env steps as PPO transitions causes the gradient to assign credit to macro actions for events they never controlled.
- **Discounting the bootstrap by segment length** is what makes a segment-based MDP a valid *semi-Markov* MDP (Bradtke & Duff, 1995; Sutton, Precup & Singh, 1999 — options framework).
- **GAE on macro-transitions** trades off bias and variance the same way as in standard PPO; the only modification is that $\gamma^{\text{macro}}_k$ depends on segment length.
- **Per-batch normalization** of advantages is well-known to stabilize PPO; we keep it.
- **Factorized categorical** keeps the parameter count low and makes masking tractable. A flat softmax over $|I| \times |C|$ would explode with curriculum size.
- **Dangling segments not dropped** prevents survivorship bias toward easy concepts.

## 4. Consequences

- Macro buffer size $\ll$ DQN replay size. A typical run produces 50–200 macro transitions per episode vs 1000–5000 meso transitions. Macro buffer capacity defaults to a few rollouts ($T_{\text{rollout}}^{\text{macro}} = 256$ macro transitions).
- The macro critic sees a smaller, more variant sample. We compensate with **value-loss clipping** (Schulman et al., 2017 OpenAI Five-style):

$$
L^V = \max\Big((V_\theta - V^{\text{target}})^2, (V^{\text{old}} + \mathrm{clip}(V_\theta - V^{\text{old}}, -\epsilon_V, \epsilon_V) - V^{\text{target}})^2\Big).
$$

This makes the critic learn more slowly but more reliably.

- Memory & curriculum providers must expose **concept boundary signals** to the coordinator. The coordinator emits `macro_step_done=True` only when a boundary is hit.
- The DQN replay buffer indexes its transitions with the **macro segment ID** they belong to; this enables ablation studies that re-attribute meso rewards to macro segments.

## 5. Alternatives considered

- **Option-critic** (Bacon, Harb, Precup, 2017): full options framework with learned termination. Rejected for v1 — we have a natural termination signal ($\tau_m$), and option-critic adds variance and termination-gradient complexity that isn't justified at this stage. Reserved as a Phase-12+ extension.
- **Hierarchical PPO with HIRO-style sub-goals**: rejected; our sub-controller is DQN, not a learned subgoal-conditioned policy.
- **Single PPO over the full env step**: rejected (see Context).
- **Macro action conditioned on a learned trajectory embedding**: deferred; needs a transformer-based encoder which violates the "lightweight memory" constraint (v1 priority of simplicity).

## 6. Test plan (referenced from `tests/test_ppo.py` when Phase 10 lands)

- **Segment construction:** given a synthetic env trace with hand-coded boundaries, verify segment indices, segment returns, and bootstrap values match analytical expectations.
- **Discount factor:** verify $\gamma^{\text{macro}}_k$ on a length-3 segment equals $\gamma^3$.
- **Dangling segment:** verify a session that times out mid-segment still produces a transition with a non-zero bootstrap.
- **Clipping invariant:** verify that $L^{\text{CLIP}}$ is exactly the unclipped surrogate when $\rho \in [1-\epsilon, 1+\epsilon]$.
- **Factorized entropy:** verify $H(\pi) = H(\pi^{\text{inst}}) + \mathbb{E}_{a^{\text{inst}}} H(\pi^{\text{concept}})$.

## 7. References

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016.
- Sutton, Precup, Singh, *Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning*, AIJ 1999.
- Bradtke & Duff, *Reinforcement learning methods for continuous-time Markov decision problems*, NIPS 1995.
