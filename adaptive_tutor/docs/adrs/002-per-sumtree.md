# ADR-002 — Prioritized Experience Replay: Sum-Tree Implementation Choices

**Status:** Accepted
**Owner:** Infrastructure Engineer
**Date:** 2026-05-11

---

## 1. Context

Prioritized Experience Replay (PER, Schaul et al., 2016) replaces uniform sampling from a replay buffer with sampling proportional to priorities $p_i$, typically derived from per-transition TD-error magnitude. The defining operation is

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \qquad \alpha \in [0,1].
$$

To avoid the bias introduced by non-uniform sampling, samples are reweighted with importance-sampling (IS) weights

$$
w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^{\beta}, \qquad \beta \in [0,1],
$$

then normalized by $\max_i w_i$ inside the batch.

For replay sizes of $N \sim 10^5{-}10^6$, the *sampling* and *priority update* operations must be efficient. The canonical structure is a **sum-tree**.

## 2. Decision

We adopt the **proportional variant** of PER with a complete binary sum-tree implemented over a flat NumPy array. We do **not** use the rank-based variant.

### 2.1 Data layout

For capacity $N$ (rounded up to the nearest power of two), the sum-tree has $2N - 1$ nodes stored contiguously:

```
indices  : [0]                              ? root (sum of all priorities)
           [1, 2]                           ? internal nodes
           ...
           [N-1, N-2, ..., 2N-2]            ? leaves (one per transition slot)
```

Two auxiliary arrays:

- `priority: np.ndarray, shape (2N-1,), dtype=float32` — the tree node values.
- `data: object[]` (or contiguous tensors when transitions are fixed-size) — the transitions themselves, indexed `0 .. N-1`. Leaf `i` corresponds to `data[i - (N - 1)]`.

### 2.2 Operations and complexity

| Operation | Complexity |
|---|---|
| `push(transition, priority)` | $O(\log N)$ |
| `sample_batch(batch_size)` | $O(\text{batch\_size} \cdot \log N)$ |
| `update_priority(idx, p)` | $O(\log N)$ |
| `total_priority` | $O(1)$ (root) |

Sampling a single index: draw $u \sim \mathcal{U}(0, \text{root})$, then descend the tree:

```python
def _retrieve(node_idx, s):
    left = 2 * node_idx + 1
    right = left + 1
    if left >= len(priority):           # leaf
        return node_idx
    if s <= priority[left]:
        return _retrieve(left, s)
    return _retrieve(right, s - priority[left])
```

We use the **iterative** form (not recursive) in production to avoid Python recursion overhead.

### 2.3 Stratified sampling within a batch

To reduce variance of the sampled batch's empirical distribution, we partition $[0, \text{root}]$ into `batch_size` equal-width segments and draw one $u$ from each. This is the original Schaul et al. recommendation and is cheap.

### 2.4 Priority transform

Stored priorities are $|\delta_i| + \epsilon$ ("the absolute TD error plus a small constant"), **not** $(|\delta_i| + \epsilon)^\alpha$. The exponent $\alpha$ is applied only at sampling time conceptually — but because all priorities in the tree are raised to the same $\alpha$, we **store $(|\delta_i| + \epsilon)^\alpha$ directly**. Sampling becomes proportional to the stored value and we save a per-sample power operation. `?` is therefore fixed at insertion; changing ? retroactively requires a tree rebuild.

### 2.5 IS-weight annealing

$\beta$ is linearly annealed from $\beta_0 = 0.4$ to $\beta_1 = 1.0$ over the training horizon $T$, as recommended in the original paper. The schedule is a simple function `beta(step) = beta_0 + (beta_1 - beta_0) * min(1, step / T)`.

### 2.6 New transitions get max priority

A freshly pushed transition has not yet been used for an update, so its TD error is unknown. We assign it priority `max_priority_seen_so_far` to guarantee it is sampled at least once before its priority is corrected. This is critical — without it, fresh transitions can be starved indefinitely.

### 2.7 Dual-critic PER (cross-reference ADR-001 §R4)

In the dual-critic setting we store *two* priorities per transition (`?_perf`, `?_flow`) and combine them as

$$
p_i \propto (|\delta_i^{\text{perf}}|^{\alpha_p} + |\delta_i^{\text{flow}}|^{\alpha_f} + \epsilon)
$$

inserted directly into a *single* sum-tree. Per-critic priorities are kept in a separate `priority_perf` / `priority_flow` NumPy array for inspection and logging; only their combined value lives in the tree.

### 2.8 Capacity & overwrite policy

Circular buffer: when capacity is reached, the oldest transition is overwritten. The sum-tree update propagates the priority delta to the root in $O(\log N)$.

### 2.9 Numerical robustness

- `priority dtype = float32` for memory; `total_priority` accumulator is `float64` to avoid catastrophic cancellation when $N$ is large.
- Periodic (every `T_resync` updates) **full rebuild from leaves** to correct accumulated floating-point drift in internal-node sums. Default `T_resync = 1e6` updates.

## 3. Rationale

- **Proportional vs rank-based.** Proportional is simpler, faster, and within statistical noise of rank-based for typical RL settings; rank-based is preferred only when TD-error distributions are pathologically heavy-tailed, which our reward design prevents.
- **Sum-tree over heap-based variants.** Sum-tree gives $O(\log N)$ sampling proportional to weight; heaps would give $O(\log N)$ top-k retrieval but not weighted sampling.
- **NumPy-backed flat array.** Pure-Python sum-trees are 50–100× slower; sticking to NumPy lets us push >10k transitions/sec on a laptop.
- **Stored pre-exponent (?) priorities.** Halves the per-sample work; ? changes are rare.
- **`max_priority` for new transitions.** Otherwise zero-priority newcomers can starve.
- **Periodic resync.** Numerical drift is real in float32 sum-trees at large $N$; the resync cost is amortized.

## 4. Consequences

- Memory: `~ 12 N bytes` for the tree (float32 × (2N?1)) plus the transition payload itself.
- Hot path is `update_priority`, called once per transition per gradient step — must be vectorized when batched. We provide a `update_priorities(idxs, priorities)` batched API.
- Resampling is not deterministic across hardware unless we seed `numpy.random.Generator` per buffer (we do).

## 5. Alternatives considered

- **Rank-based PER** — rejected for v1 (added complexity for marginal gains).
- **Reservoir + importance sampling** — does not give proportional control over priorities.
- **Off-the-shelf libraries (e.g., `stable-baselines3`, `tianshou`)** — rejected for the core buffer because we need dual-critic priorities and tight integration with our `Transition` schema. We may import their tests as regression checks.

## 6. Test plan (referenced from `tests/test_per.py` when Phase 5 lands)

- **Correctness:** for a hand-crafted tree of size 8 with priorities `[1, 2, 3, 4, 5, 6, 7, 8]`, verify that 100k samples produce empirical probabilities within 1% of the analytical proportions.
- **Update propagation:** modify one leaf, verify all ancestors update correctly.
- **Capacity rollover:** push `N + k` transitions and verify the oldest `k` are overwritten and the tree sum equals the new sum of priorities.
- **IS-weights:** verify normalized weights are in `(0, 1]` and the largest weight in a batch is `1.0`.
- **Determinism:** with a fixed seed, two buffers receiving identical pushes produce identical batches.

## 7. References

- Schaul, Quan, Antonoglou, Silver, *Prioritized Experience Replay*, ICLR 2016.
- Hessel et al., *Rainbow: Combining Improvements in Deep Reinforcement Learning*, AAAI 2018 (PER as a Rainbow component).
