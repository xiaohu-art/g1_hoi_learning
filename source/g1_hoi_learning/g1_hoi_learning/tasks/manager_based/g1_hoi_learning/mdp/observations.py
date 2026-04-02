from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul, subtract_frame_transforms

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


# -- Object observations --


def object_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Current object position relative to robot anchor in anchor frame. (3 dims)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_pos_w,
        command.obj_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def object_rot_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Current object orientation in anchor frame (6D tangent-normal). (6 dims)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_pos_w,
        command.obj_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def diff_object_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Position error between reference and current object in anchor frame. (3 dims)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.ref_obj_pos_w,
        command.ref_obj_quat_w,
    )
    cur_pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_pos_w,
        command.obj_quat_w,
    )
    return (ref_pos_b - cur_pos_b).view(env.num_envs, -1)


def diff_object_rot_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Rotation error between reference and current object in anchor frame (6D). (6 dims)

    Computes the relative rotation q_diff = q_ref^{-1} * q_cur, then transforms
    it into the anchor frame and encodes as 6D (first 2 columns of rotation matrix).
    When ref == cur, this yields the identity matrix columns [1,0,0], [0,1,0].
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Relative rotation in world frame: how much cur deviates from ref
    q_diff_w = quat_mul(quat_conjugate(command.ref_obj_quat_w), command.obj_quat_w)
    # Transform into anchor frame
    q_anchor_inv = quat_conjugate(command.robot_anchor_quat_w)
    q_diff_local = quat_mul(quat_mul(command.robot_anchor_quat_w, q_diff_w), q_anchor_inv)
    mat = matrix_from_quat(q_diff_local)
    return mat[..., :2].reshape(env.num_envs, -1)
