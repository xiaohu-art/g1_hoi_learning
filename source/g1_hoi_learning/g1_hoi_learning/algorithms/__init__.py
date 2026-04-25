"""Custom algorithms for g1_hoi_learning.

Importing this package registers them into rsl_rl's runner namespace,
so ``OnPolicyRunner`` can resolve them via ``class_name`` in the agent cfg.
"""

import rsl_rl.runners.on_policy_runner as _opr

from .muon_ppo import MuonPPO

_opr.MuonPPO = MuonPPO

__all__ = ["MuonPPO"]
