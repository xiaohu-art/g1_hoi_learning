from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_error_magnitude

from .commands import MotionCommand


def motion_anchor_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(
            command.body_pos_w[:, command.body_indices] - command.robot_body_pos_w[:, command.body_indices]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = (
        quat_error_magnitude(
            command.body_quat_w[:, command.body_indices],
            command.robot_body_quat_w[:, command.body_indices],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(
            command.body_lin_vel_w[:, command.body_indices] - command.robot_body_lin_vel_w[:, command.body_indices]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(
            command.body_ang_vel_w[:, command.body_indices] - command.robot_body_ang_vel_w[:, command.body_indices]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


# -- Object tracking rewards --


def object_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.ref_obj_pos_w - command.obj_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def object_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.ref_obj_quat_w, command.obj_quat_w) ** 2
    return torch.exp(-error / std**2)


def object_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.ref_obj_lin_vel_w - command.obj_lin_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def object_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.ref_obj_ang_vel_w - command.obj_ang_vel_w), dim=-1)
    return torch.exp(-error / std**2)
