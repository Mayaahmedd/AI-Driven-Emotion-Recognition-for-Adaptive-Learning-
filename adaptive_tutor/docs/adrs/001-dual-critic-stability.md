# ADR-001 — Dual-Critic Gradient Detachment & Adaptive-Weighting Stability

**Status:** Accepted
**Owners:** RL Architect, Research Auditor
**Date:** 2026-05-11

---

## 1. Context

The architecture uses two critics that share a state encoder:

- $Q_{\text{perf}}(s, a)$ — value of action $a$ at state $s$ measured against **performance reward** $r^{\text{perf}}$ (correctness, mastery gain, retention).
- $Q_{\text{flow}}(s, a)$ — value of action $a$ at state $s$ measured against **emotional-flow reward** $r^{\text{flow}}$ (engagement gain, frustration/boredom/confusion penalty).

A single composite Q-value drives action selection:

$$
Q(s, a) = w_{\text{perf}}(s) \cdot Q_{\text{perf}}(s, a) \;+\; w_{\text{flow}}(s) \cdot Q_{\text{flow}}(s, a),
$$

with state-dependent weights

$$
w_{\text{flow}}(s) = \frac{E_{\text{conf}} + E_{\text{bored}} + E_{\text{frust}}}{E_{\text{eng}} + E_{\text{conf}} + E_{\text{bored}} + E_{\text{frust}} + \epsilon},\qquad w_{\text{perf}}(s) = 1 - w_{\text{flow}}(s).
$$

Two coupled failure modes appear if this is implemented naively:

- **F1. Critic interference.** A single shared encoder receives gradients from both TD losses. When the two reward signals are strongly anti-correlated (which is common: a hard problem raises mastery gain but also frustration), the encoder is pulled in opposite directions and learning slows or oscillates.
- **F2. Weighting loop instability.** $w_{\text{flow}}$ depends on $E_t$, which is part of $s$. If $w_{\text{flow}}$ carries gradients into the encoder via the composite value, the policy can learn to **drive emotions** simply to reweight Q-values, which is reward hacking.

## 2. Decision

We adopt the following five rules. They are mandatory in every dual-critic implementation in this codebase.

### Rule R1 — Two independent value heads on a shared encoder

```
state s ??? encoder ?(s) ????? head_perf ??? Q_perf(s, a)
                           ??? head_flow ??? Q_flow(s, a)
```

Both heads receive gradients from their *own* TD losses, but the encoder receives **only their sum**. To prevent F1 we balance the encoder gradient magnitudes with **GradNorm** (Chen et al., 2018) or a simpler **fixed-ratio scaling** with `?_perf`, `?_flow` decay parameters. v1 uses the simpler fixed-ratio scaling; GradNorm is a Phase-12 ablation.

### Rule R2 — Detach the weights from the value path

When the composite value $Q$ is used by the policy (for action selection or in the policy gradient), the weights $w_{\text{perf}}$ and $w_{\text{flow}}$ are **detached from the computation graph**:

```python
w_flow_det = w_flow.detach()   # critical
Q_composite = (1 - w_flow_det) * Q_perf + w_flow_det * Q_flow
```

This breaks the gradient path `policy ? action ? next emotions ? weights ? Q`, preventing F2. The weights still vary with state at inference and during forward passes, but the **agent cannot learn to manipulate them**.

### Rule R3 — Separate target networks per critic

Each critic has its own Polyak-averaged target network:

$$
\theta^{-}_{\text{perf}} \leftarrow \tau \theta_{\text{perf}} + (1-\tau)\theta^{-}_{\text{perf}}, \qquad
\theta^{-}_{\text{flow}} \leftarrow \tau \theta_{\text{flow}} + (1-\tau)\theta^{-}_{\text{flow}}.
$$

We do **not** share the target encoder, even though we share the online encoder. The asymmetry is intentional: sharing the online encoder reduces parameter count and forces a common state representation; separating target encoders prevents the slow-moving target from coupling the two TD signals.

### Rule R4 — Independent priorities in PER

The PER priority for transition $i$ is

$$
p_i \propto \big(|\delta_i^{\text{perf}}|^{\alpha_p} + |\delta_i^{\text{flow}}|^{\alpha_f} + \epsilon\big)^{\beta_{\text{PER}}},
$$

where $\delta_i^{\text{perf}}$ and $\delta_i^{\text{flow}}$ are the **per-critic** TD errors. Using only the composite TD error biases sampling toward whichever objective is dominating, defeating the purpose of the dual critic.

### Rule R5 — Numerical clamps on the weights

Always clamp:

$$
w_{\text{flow}} \in [w_{\min}, w_{\max}], \qquad 0 < w_{\min} \le w_{\max} < 1.
$$

Defaults: $w_{\min} = 0.05$, $w_{\max} = 0.80$. Without clamps, a learner with $E = (0, 0, 0, 0)$ (no FER signal) gives a degenerate weight. The $\epsilon$ in the formula handles the zero-denominator case; the clamp handles the **policy-effective** range.

## 3. Rationale (why these specific rules)

- **Detachment (R2)** is the single most important rule. Without it, the adaptive weighting is differentiable through the state, and even a small policy capacity is enough to discover that "increase frustration ? increase $w_{\text{flow}}$ ? bigger weight on the head that currently scores best" is locally rewarding. We have validated this failure mode in prior literature on auto-tuned multi-objective RL.
- **Independent targets (R3)** decouple the bias of the two value estimators. If they shared a target network, an overestimation in one objective leaks into the bootstrap of the other.
- **Per-critic priorities (R4)** keep the replay buffer's sampling distribution faithful to both objectives. A learner who has *mastered* the curriculum but is still emotionally noisy must still see high-priority emotional-shock transitions; otherwise the flow critic stops learning.
- **Clamping (R5)** is a numerical safeguard that also gives the engineer a knob: $w_{\max}=0.80$ guarantees the performance head always carries at least 20% of the composite, preventing degenerate policies that ignore learning gain.

## 4. Consequences

**Positive**

- Eliminates the reward-hacking path via weight manipulation.
- Reduces critic interference compared to a single Q-head on the same encoder.
- Per-critic PER preserves coverage of both objectives.

**Negative / costs**

- ~2× parameter count for value heads (encoder is shared, so total is much less than 2× the network).
- Extra hyperparameters: $\lambda_{\text{perf}}, \lambda_{\text{flow}}, \tau_{\text{perf}}, \tau_{\text{flow}}, w_{\min}, w_{\max}, \alpha_p, \alpha_f$. All exposed in `configs/agents/dual_critics.yaml`.

## 5. Alternatives considered

- **Single critic with weighted reward** ($r = w_{\text{perf}} r^{\text{perf}} + w_{\text{flow}} r^{\text{flow}}$). Rejected: collapses both signals into one Q, loses interpretability for the dashboard, and inherits all of F2 if weights are state-dependent.
- **Hyperparameter-only weights** (no state dependence). Rejected: the entire research thesis of emotion-driven adaptive weighting requires state-dependent weights.
- **GradNorm or PCGrad for encoder gradient balancing.** Reserved for Phase-12 ablation; default v1 uses fixed-ratio scaling for simplicity.
- **Two fully separate networks (no shared encoder).** Rejected: doubles parameter count and loses the structural inductive bias that performance and flow share a state representation.

## 6. Test plan (referenced from `tests/test_dual_critics.py` when Phase 11 lands)

- Unit test: weight detachment — verify `Q_composite.backward()` does not produce gradients on the FER inputs.
- Unit test: clamp — verify $w_{\text{flow}}$ stays in `[w_min, w_max]` across pathological FER inputs (all zeros, all ones).
- Unit test: per-critic PER priorities — verify priorities track both `?_perf` and `?_flow` independently.
- Stability test: run 10k steps on a fixed simulator cohort and verify TD-loss variance for each critic is bounded.

## 7. References

- Chen et al., *GradNorm: Gradient Normalization for Adaptive Loss Balancing*, ICML 2018.
- Schaul et al., *Prioritized Experience Replay*, ICLR 2016.
- van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016.
- Yu et al., *Gradient Surgery for Multi-Task Learning* (PCGrad), NeurIPS 2020.
