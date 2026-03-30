from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionCommand


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal anchor position in the robot's anchor (body) frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal anchor orientation in the robot's anchor frame (first 2 columns of rotation matrix)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(env.num_envs, -1)


def motion_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal tracked-body positions in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.body_indices
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, len(body_indices), -1),
        command.body_pos_w[:, body_indices],
        command.body_quat_w[:, body_indices],
    )
    return pos_b.view(env.num_envs, -1)


def motion_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal tracked-body orientations in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.body_indices
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, len(body_indices), -1),
        command.body_pos_w[:, body_indices],
        command.body_quat_w[:, body_indices],
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot tracked-body positions in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.body_indices
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_body_pos_w[:, body_indices],
        command.robot_body_quat_w[:, body_indices],
    )
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot tracked-body orientations in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.body_indices
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_body_pos_w[:, body_indices],
        command.robot_body_quat_w[:, body_indices],
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)
