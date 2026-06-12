"""
Model Exponential Moving Average (EMA).

The EMA module maintains a separate copy of every parameter and buffer
of the live model. After each ``optimizer.step()`` it updates each EMA
parameter with the rule:

    v_ema  <-  decay * v_ema + (1 - decay) * v_live

Non-floating-point buffers (e.g. ``BatchNorm.num_batches_tracked``) are
copied verbatim; integer EMA-averaging is not defined.

Validation and test evaluation are performed against ``ema.module``;
the live model continues to be optimised. This is a standard practice
in vision pipelines because the EMA copy produces smoother predictions
and tends to give a small but reliable Macro-F1 gain at no architectural
cost.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}.")
        self.decay = float(decay)
        # Deep-copy and freeze. EMA weights are not part of the optimiser.
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA state from the live model.

        Both ``state_dict`` ordered traversals must be consistent across
        calls; PyTorch guarantees insertion-ordered ``state_dict`` so the
        zip below is safe.
        """
        ema_state = self.module.state_dict()
        live_state = model.state_dict()
        for k, ema_v in ema_state.items():
            live_v = live_state[k]
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(live_v.detach(), alpha=1.0 - self.decay)
            else:
                ema_v.copy_(live_v)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)

    def to(self, device):
        self.module.to(device)
        return self
