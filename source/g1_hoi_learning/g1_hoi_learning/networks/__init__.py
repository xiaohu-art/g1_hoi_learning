"""Custom network architectures for g1_hoi_learning.

Importing this package registers the networks into rsl_rl's runner namespace,
so that the OnPolicyRunner can resolve them via ``class_name`` in the agent cfg.
"""

import rsl_rl.runners.on_policy_runner as _opr

from .simba import SimBa, SimBaActorCritic, SimBaBlock

_opr.SimBaActorCritic = SimBaActorCritic

__all__ = ["SimBa", "SimBaActorCritic", "SimBaBlock"]
