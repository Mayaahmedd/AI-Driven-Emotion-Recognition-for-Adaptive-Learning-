# ADR-004 — Offline Policy Evaluation: FQE / WIS Test Harness

**Status:** Accepted
**Owner:** Evaluation Scientist
**Date:** 2026-05-11

---

## 1. Context

The system is **simulator-first** in v1. After Phase 14 it may be deployed against real learner logs, but we **must not** train a policy in simulation, claim it is "better", and deploy it without an evaluation that does not require re-deployment. That is the canonical offline policy evaluation (OPE) problem:

> Given a logged dataset $\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1})\}$ collected by a *behavior* policy $\pi_b$, estimate the value $J(\pi_e) = \mathbb{E}_{\pi_e}[\sum_t \gamma^t r_t]$ of a *new* evaluation policy $\pi_e$ without rolling $\pi_e$ out in the real world.

Two complementary estimators are industry-standard:

- **Importance-sampling family (IS / WIS / step-WIS / PDIS).** Unbiased (for IS) but high variance. Sensitive to action-coverage of $\pi_b$.
- **Direct method (FQE — Fitted Q Evaluation).** Low variance, biased by model class. Robust when $\pi_b$ is unknown or has poor coverage.

The combination of the two (or the *doubly robust* estimator that mixes them) is the de facto standard for medical-, dialogue-, and tutoring-style OPE.

## 2. Decision

We implement **three estimators** in `evaluation/offline_eval.py`. All three are produced for every OPE report so reviewers can cross-check.

### 2.1 Per-decision Weighted Importance Sampling (Step-WIS / PDIS)

For each trajectory $\tau$ of length $T_\tau$,

$$
\rho_t^{(\tau)} = \prod_{u = 0}^{t} \frac{\pi_e(a_u^{(\tau)} \mid s_u^{(\tau)})}{\pi_b(a_u^{(\tau)} \mid s_u^{(\tau)})},
\qquad
\hat J^{\text{PDIS}} = \frac{1}{N} \sum_\tau \sum_t \gamma^t \cdot \rho_t^{(\tau)} \cdot r_t^{(\tau)}.
$$

Weighted variant (self-normalized) reduces variance at the cost of bias:

$$
\hat J^{\text{WIS}} = \sum_\tau \sum_t \frac{\gamma^t \rho_t^{(\tau)}}{\sum_{\tau'} \rho_t^{(\tau')}} \cdot r_t^{(\tau)}.
$$

We log **both** PDIS and WIS. The *effective sample size* (ESS) is logged as a coverage diagnostic:

$$
\text{ESS}_t = \frac{(\sum_\tau \rho_t^{(\tau)})^2}{\sum_\tau (\rho_t^{(\tau)})^2}.
$$

If ESS falls below 10% of $N$ in early time steps, WIS is reported as **unreliable** in the report.

#### 2.1.1 Behavior policy estimation

When $\pi_b$ is unknown (the typical case for logged real-learner data), we fit a behavior policy with a calibrated softmax classifier $\hat \pi_b(a \mid s)$ on $\mathcal{D}$ before computing ratios. The classifier is small (MLP, 2 layers) and is cross-validated. **Reporting includes the classifier's calibration error (ECE).**

#### 2.1.2 Action masking and zero ratios

Both $\pi_e$ and $\pi_b$ pass through the **same safety mask**. If the mask makes $\pi_e(a \mid s) = 0$ for a logged action, the ratio is set to $0$ for that step (and downstream — this is the correct PDIS behavior for action-conditioned non-coverage).

### 2.2 Fitted Q Evaluation (FQE)

We fit $Q^{\pi_e}$ on $\mathcal{D}$ by iterated regression:

$$
Q^{(k+1)}(s, a) \leftarrow r + \gamma \cdot \mathbb{E}_{a' \sim \pi_e(\cdot \mid s')}\big[Q^{(k)}(s', a')\big].
$$

In practice this is implemented as $K$ ($\sim 50$) gradient-descent rounds on an MLP regressor with target frozen for `target_sync` iterations (analogous to DQN). The estimate is

$$
\hat J^{\text{FQE}} = \mathbb{E}_{s_0 \sim \mathcal{D}}\big[\mathbb{E}_{a_0 \sim \pi_e(\cdot \mid s_0)} Q^{(K)}(s_0, a_0)\big].
$$

#### 2.2.1 Model class & overfitting control

- MLP, 2–3 hidden layers, width matched to the policy critic (config-driven).
- 5-fold cross-validation on $\mathcal{D}$; the reported $\hat J^{\text{FQE}}$ is the mean of held-out predictions; the std across folds is reported as a "model-class variance" bar.
- Early stopping on a held-out fold's TD-loss.

### 2.3 Doubly Robust (DR)

The doubly robust estimator (Jiang & Li, 2016) combines the two:

$$
\hat J^{\text{DR}} = \frac{1}{N}\sum_\tau \sum_t \gamma^t \Big( \rho_t^{(\tau)}(r_t^{(\tau)} - Q^{(K)}(s_t, a_t)) + V^{(K)}(s_t) \Big),
\quad V^{(K)}(s) = \mathbb{E}_{a \sim \pi_e}[Q^{(K)}(s, a)].
$$

DR is **unbiased when either** $\hat \pi_b$ **or** $Q^{(K)}$ is correct (hence "doubly robust"). It is our headline number in OPE reports.

### 2.4 Report contract

`evaluation/offline_eval.py::offline_evaluate(policy, dataset, config) -> OfflineReport` returns:

```python
OfflineReport(
    pdis=float, pdis_ci=(low, high),
    wis=float,  wis_ci=(low, high),
    fqe=float,  fqe_ci=(low, high),
    dr=float,   dr_ci=(low, high),
    ess_per_step=np.ndarray,
    behavior_calibration_ece=float,
    fqe_cv_std=float,
    reliability_flags=Sequence[str],   # e.g., "low_ess", "high_fqe_cv"
)
```

Confidence intervals are produced by **bootstrap over trajectories** (1000 resamples) — not over transitions, because transitions inside a trajectory are correlated.

## 3. Rationale

- **Three estimators** because no single OPE estimator is reliable across all coverage regimes. PDIS/WIS are unbiased / low-bias but explode under low coverage. FQE is stable but biased by model class. DR is the principled combination.
- **Behavior policy estimation** is unavoidable when shipping against real logs; we make it explicit and report calibration.
- **Bootstrap CIs over trajectories** because trajectories are the unit of independence, not transitions.
- **Reliability flags** force the reviewer to read coverage diagnostics, not just the headline number.

## 4. Consequences

- FQE training is non-trivial. We expose an `OfflineEvaluationConfig` with the same hyperparameter discipline as the trainers (seeds, target-sync, optimizer settings).
- Reports include both numeric outputs and a markdown summary section explaining which estimator is most trustworthy for this dataset.
- Tests will use a small synthetic MDP with closed-form ground-truth value to validate all three estimators within a tolerance.

## 5. Alternatives considered

- **Marginalized Importance Sampling (MIS)** — better for long horizons, but more complex; reserved for Phase-14+ if PDIS variance becomes the bottleneck.
- **Direct simulator evaluation only.** Rejected — defeats the purpose of OPE (sim-to-real gap).
- **Off-the-shelf libraries (DICE, d3rlpy)** — we may import them as reference implementations, but the report integration is custom.

## 6. Test plan (referenced from `tests/test_offline_eval.py` when Phase 13 lands)

- **Closed-form check:** on a 2-state, 2-action MDP with known transition dynamics, FQE/PDIS/WIS/DR all converge to the analytical $J(\pi_e)$.
- **High coverage:** when $\pi_e \approx \pi_b$, PDIS variance should be small and PDIS ? DR ? FQE.
- **Low coverage stress test:** when $\pi_e$ deviates strongly from $\pi_b$, PDIS variance explodes, WIS biases toward $\pi_b$, FQE is stable, DR sits between PDIS and FQE.
- **Reliability flag:** synthetic dataset with low ESS triggers the `low_ess` flag.

## 7. References

- Precup, Sutton, Singh, *Eligibility Traces for Off-Policy Policy Evaluation*, ICML 2000 (PDIS).
- Jiang & Li, *Doubly Robust Off-policy Value Evaluation for Reinforcement Learning*, ICML 2016.
- Le, Voloshin, Yue, *Batch Policy Learning under Constraints*, ICML 2019 (FQE recipe).
- Thomas & Brunskill, *Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning*, ICML 2016.
