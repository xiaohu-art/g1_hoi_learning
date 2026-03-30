from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .commands import MotionCommand


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    motion_projected_gravity_b = math_utils.quat_apply_inverse(
        command.anchor_quat_w, asset.data.GRAVITY_VEC_W
    )
    robot_projected_gravity_b = math_utils.quat_apply_inverse(
        command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W
    )

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.robot.find_bodies(body_names, preserve_order=True)[0]
    error = torch.norm(
        command.body_pos_w[:, body_indices] - command.robot_body_pos_w[:, body_indices],
        dim=-1,
    )
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.robot.find_bodies(body_names, preserve_order=True)[0]
    error = torch.abs(
        command.body_pos_w[:, body_indices, -1] - command.robot_body_pos_w[:, body_indices, -1]
    )
    return torch.any(error > threshold, dim=-1)
