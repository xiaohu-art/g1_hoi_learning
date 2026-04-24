from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization


class SimBaBlock(nn.Module):
    """Pre-LN residual MLP block from SimBa (Lee et al., ICLR 2025).

    x -> LayerNorm -> Linear(d, d*m) -> ReLU -> Linear(d*m, d) -> (+ x)
    """

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class SimBa(nn.Module):
    """SimBa backbone: linear input projection -> N residual blocks -> post-LN -> linear head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_blocks: int,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[SimBaBlock(hidden_dim, expansion) for _ in range(num_blocks)])
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.output_proj.weight, gain=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.post_norm(x)
        return self.output_proj(x)

    def __getitem__(self, idx: int) -> nn.Module:
        # Compat shim for isaaclab_rl's policy exporter, which assumes
        # ``actor[0].in_features`` / ``actor[-1].out_features`` on an nn.Sequential.
        if idx == 0:
            return self.input_proj
        if idx == -1:
            return self.output_proj
        raise IndexError(f"SimBa only exposes indices 0 (input_proj) and -1 (output_proj); got {idx}")


class SimBaActorCritic(ActorCritic):
    """Actor-critic with SimBa backbones.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 512,
        actor_num_blocks: int = 2,
        critic_num_blocks: int = 2,
        expansion: int = 4,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: Any,
    ) -> None:
        if state_dependent_std:
            raise NotImplementedError("SimBaActorCritic currently supports state-independent std only.")
        if kwargs:
            print(
                "SimBaActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )

        nn.Module.__init__(self)

        # Observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for g in obs_groups["policy"]:
            assert len(obs[g].shape) == 2, "SimBaActorCritic only supports 1D observations."
            num_actor_obs += obs[g].shape[-1]
        num_critic_obs = 0
        for g in obs_groups["critic"]:
            assert len(obs[g].shape) == 2, "SimBaActorCritic only supports 1D observations."
            num_critic_obs += obs[g].shape[-1]

        self.state_dependent_std = False

        # Actor
        self.actor = SimBa(
            input_dim=num_actor_obs,
            output_dim=num_actions,
            hidden_dim=actor_hidden_dim,
            num_blocks=actor_num_blocks,
            expansion=expansion,
        )
        print(f"Actor SimBa: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = SimBa(
            input_dim=num_critic_obs,
            output_dim=1,
            hidden_dim=critic_hidden_dim,
            num_blocks=critic_num_blocks,
            expansion=expansion,
        )
        print(f"Critic SimBa: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise (state-independent)
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated by ActorCritic._update_distribution)
        self.distribution: Normal | None = None

        # Disable args validation for speedup (matches parent)
        Normal.set_default_validate_args(False)
