"""Double DQN with factorised meso heads (Phase 8).

Action values are additive across heads.  Double DQN selects each head's
index using the **online** network and evaluates with the **target**
network to reduce over-estimation bias.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from adaptive_tutor.rl.dqn.decode import decode_meso_to_assistments
from adaptive_tutor.rl.dqn.model import FactorisedDQN

if TYPE_CHECKING:
    from adaptive_tutor.state.state import LearnerState


def q_sum_gather(
    qs: torch.Tensor,
    qd: torch.Tensor,
    qp: torch.Tensor,
    idx_s: torch.Tensor,
    idx_d: torch.Tensor,
    idx_p: torch.Tensor,
) -> torch.Tensor:
    """Sum of gathered Q-values for batch indices. Shapes: heads ``[B, n]``, idx ``[B]``."""
    b = torch.arange(qs.size(0), device=qs.device, dtype=torch.long)
    return qs[b, idx_s] + qd[b, idx_d] + qp[b, idx_p]


class DoubleDQNAgent:
    """Online + target factorised Q-networks."""

    def __init__(
        self,
        *,
        state_dim: int = 13,
        hidden: tuple[int, ...] = (128, 128),
        device: torch.device | None = None,
    ) -> None:
        dev = device or torch.device("cpu")
        self.device = dev
        self.online = FactorisedDQN(state_dim=state_dim, hidden=hidden).to(dev)
        self.target = FactorisedDQN(state_dim=state_dim, hidden=hidden).to(dev)
        self.target.load_state_dict(self.online.state_dict())

    def sync_target(self) -> None:
        """Hard-copy online weights into target (call after construction)."""
        self.target.load_state_dict(self.online.state_dict())

    def act(
        self,
        state: LearnerState,
        *,
        num_concepts: int,
        epsilon: float,
        rng: random.Random,
    ) -> tuple[int, int, int, str]:
        """Epsilon-greedy independent per head; return indices + decoded ASSISTments string."""
        if rng.random() < epsilon:
            s = rng.randint(0, 3)
            d = rng.randint(0, 2)
            p = rng.randint(0, 1)
            return s, d, p, decode_meso_to_assistments(s, d, p)

        x = state.to_tensor(num_concepts=num_concepts).to(self.device).unsqueeze(0)
        with torch.no_grad():
            qs, qd, qp = self.online(x)
            s = int(qs.argmax(dim=-1).item())
            d = int(qd.argmax(dim=-1).item())
            p = int(qp.argmax(dim=-1).item())
        return s, d, p, decode_meso_to_assistments(s, d, p)

    @torch.no_grad()
    def next_state_value_double(self, s_next: torch.Tensor) -> torch.Tensor:
        """Double DQN bootstrap: greedy head choices from online, evaluation from target."""
        qs_o, qd_o, qp_o = self.online(s_next)
        bs = qs_o.argmax(dim=-1)
        bd = qd_o.argmax(dim=-1)
        bp = qp_o.argmax(dim=-1)
        qs_t, qd_t, qp_t = self.target(s_next)
        return q_sum_gather(qs_t, qd_t, qp_t, bs, bd, bp)
