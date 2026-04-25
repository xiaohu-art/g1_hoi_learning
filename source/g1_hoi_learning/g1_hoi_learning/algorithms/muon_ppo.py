from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent


class OptimizerGroup(torch.optim.Optimizer):
    """
    Wrapper around multiple optimizers so they can be used through a single
    optimizer-like interface (step/zero_grad/param_groups).
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        if len(optimizers) == 0:
            raise ValueError("OptimizerGroup requires at least one optimizer.")

        all_params: list[nn.Parameter] = []
        for opt in optimizers:
            for group in opt.param_groups:
                all_params.extend(group["params"])
        if len(all_params) == 0:
            raise ValueError("OptimizerGroup underlying optimizers have no parameters.")

        super().__init__(params=all_params, defaults={})
        self.optimizers = optimizers

        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = self.optimizers[0].step(closure)
            for opt in self.optimizers[1:]:
                opt.step()
            return loss
        for opt in self.optimizers:
            _loss = opt.step()
            if loss is None:
                loss = _loss
        return loss

    def zero_grad(self, set_to_none: bool | None = None):
        for opt in self.optimizers:
            if set_to_none is None:
                opt.zero_grad()
            else:
                opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "class": self.__class__.__name__,
        }

    def load_state_dict(self, state_dict):
        opt_states = state_dict.get("optimizers", None)
        if opt_states is None:
            return
        if len(opt_states) != len(self.optimizers):
            warnings.warn(
                f"OptimizerGroup state has {len(opt_states)} optimizers, "
                f"but current instance has {len(self.optimizers)}. "
                "Loading states for the matching prefix only."
            )
        for opt, opt_state in zip(self.optimizers, opt_states):
            opt.load_state_dict(opt_state)


class MuonAdamWWrapper(OptimizerGroup):
    """Split 2-D (Muon) vs other (AdamW) params across modules.

    Parameters
    ----------
    ignore_frozen
        If True (default), parameters with ``requires_grad=False`` are omitted
        from both optimizers (no optimizer state, no updates). If False, frozen
        parameters are still registered; they are typically skipped at ``step``
        when their gradient is ``None``.
    """

    def __init__(
        self,
        modules: list[nn.Module],
        lr: float,
        weight_decay: float = 0.01,
        ignore_frozen: bool = True,
    ):
        try:
            from torch.optim import Muon
        except ImportError as exc:
            raise ImportError(
                "torch.optim.Muon is unavailable. Active torch is "
                f"{torch.__version__} at {torch.__file__}. Muon requires torch>=2.10."
            ) from exc

        seen: set[int] = set()
        muon_params: list[nn.Parameter] = []
        adamw_params: list[nn.Parameter] = []
        for module in modules:
            for _, p in module.named_parameters():
                if id(p) in seen:
                    continue
                if ignore_frozen and not p.requires_grad:
                    continue
                seen.add(id(p))
                if p.dim() == 2 and not getattr(p, "_non_muon", False):
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

        optimizers: list[torch.optim.Optimizer] = []
        if len(muon_params) > 0:
            optimizers.append(Muon(muon_params, lr=lr, adjust_lr_fn="match_rms_adamw"))
        if len(adamw_params) > 0:
            optimizers.append(AdamW(adamw_params, lr=lr, weight_decay=weight_decay))
        if len(optimizers) == 0:
            raise ValueError(
                "MuonAdamWWrapper: no parameters were assigned. With ignore_frozen=True, "
                "every parameter may have requires_grad=False."
            )

        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_params)
        print(
            f"[MuonAdamWWrapper] Muon: {len(muon_params)} tensors / {n_muon:,} elems"
            f"  |  AdamW: {len(adamw_params)} tensors / {n_adamw:,} elems"
        )

        super().__init__(optimizers)


class MuonPPO(PPO):
    """PPO with Muon (2-D weights) + AdamW (everything else) optimizers.
    """

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent,
        weight_decay: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, **kwargs)
        # Replace the Adam built by PPO.__init__ with Muon + AdamW.
        self.optimizer = MuonAdamWWrapper(
            modules=[self.policy],
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
