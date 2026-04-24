# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register actor-critic networks
from . import networks  # noqa: F401

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *
