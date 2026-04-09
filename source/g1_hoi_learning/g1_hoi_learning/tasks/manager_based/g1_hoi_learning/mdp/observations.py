from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionCommand


def motion_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference joint positions from motion data."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_pos


def motion_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference joint velocities from motion data."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_vel


def motion_future_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future reference joint positions for configured offsets."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.future_joint_pos.reshape(env.num_envs, -1)


def motion_future_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future reference joint velocities for configured offsets."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.future_joint_vel.reshape(env.num_envs, -1)


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


def motion_future_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future anchor positions in current robot anchor frame. (num_envs, num_offsets * 3)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    # future_anchor_pos_w: (num_envs, num_offsets, 3)
    n_offsets = len(command.cfg.future_offsets)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, n_offsets, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, n_offsets, -1),
        command.future_anchor_pos_w,
        command.future_anchor_quat_w,
    )
    return pos.reshape(env.num_envs, -1)


def motion_future_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future anchor orientations in current robot anchor frame. (num_envs, num_offsets * 6)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    n_offsets = len(command.cfg.future_offsets)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, n_offsets, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, n_offsets, -1),
        command.future_anchor_pos_w,
        command.future_anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(env.num_envs, -1)


def motion_future_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future tracked-body positions in current robot anchor frame. (num_envs, num_offsets * num_bodies * 3)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    bi = command.body_indices
    n_offsets = len(command.cfg.future_offsets)
    n_bodies = len(bi)
    # future_body_pos_w: (num_envs, num_offsets, num_all_bodies, 3) -> select tracked bodies
    future_pos = command.future_body_pos_w[:, :, bi]  # (num_envs, num_offsets, num_bodies, 3)
    future_quat = command.future_body_quat_w[:, :, bi]
    anchor_pos = command.robot_anchor_pos_w[:, None, None, :].expand(-1, n_offsets, n_bodies, -1)
    anchor_quat = command.robot_anchor_quat_w[:, None, None, :].expand(-1, n_offsets, n_bodies, -1)
    pos_b, _ = subtract_frame_transforms(anchor_pos, anchor_quat, future_pos, future_quat)
    return pos_b.reshape(env.num_envs, -1)


def motion_future_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future tracked-body orientations in current robot anchor frame. (num_envs, num_offsets * num_bodies * 6)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    bi = command.body_indices
    n_offsets = len(command.cfg.future_offsets)
    n_bodies = len(bi)
    future_pos = command.future_body_pos_w[:, :, bi]
    future_quat = command.future_body_quat_w[:, :, bi]
    anchor_pos = command.robot_anchor_pos_w[:, None, None, :].expand(-1, n_offsets, n_bodies, -1)
    anchor_quat = command.robot_anchor_quat_w[:, None, None, :].expand(-1, n_offsets, n_bodies, -1)
    _, ori_b = subtract_frame_transforms(anchor_pos, anchor_quat, future_pos, future_quat)
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


def motion_future_obj_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future reference object positions in current robot anchor frame. (num_envs, num_offsets * 3)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    n_offsets = len(command.cfg.future_offsets)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, n_offsets, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, n_offsets, -1),
        command.future_obj_pos_w,
        command.future_obj_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def motion_future_obj_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future reference object orientations in current robot anchor frame. (num_envs, num_offsets * 6)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    n_offsets = len(command.cfg.future_offsets)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, n_offsets, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, n_offsets, -1),
        command.future_obj_pos_w,
        command.future_obj_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)