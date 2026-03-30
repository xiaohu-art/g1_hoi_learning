import os
from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude, quat_from_euler_xyz, quat_mul, sample_uniform


class MotionLoader:
    def __init__(self, motion_file: str, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file, allow_pickle=True)

        self.fps = int(data["fps"][0])
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.time_step_total = self.joint_pos.shape[0]


class MotionCommand(CommandTerm):
    cfg: "MotionCommandCfg"

    def __init__(self, cfg: "MotionCommandCfg", env: ManagerBasedRLEnv):
        # Set up robot references before super().__init__ because it calls _set_debug_vis_impl
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.anchor_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.body_indices, self.body_names = self.robot.find_bodies(cfg.body_names, preserve_order=True)

        super().__init__(cfg, env)

        self.motion = MotionLoader(self.cfg.motion_file, device=self.device)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

    # -- command property (fed as observation)
    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    # -- motion data properties
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion._body_pos_w[self.time_steps, self.anchor_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion._body_quat_w[self.time_steps, self.anchor_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion._body_lin_vel_w[self.time_steps, self.anchor_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion._body_ang_vel_w[self.time_steps, self.anchor_index]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion._body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion._body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion._body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion._body_ang_vel_w[self.time_steps]

    # -- robot data properties
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, :]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, :]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.anchor_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.anchor_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.anchor_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.anchor_index]

    # -- CommandTerm interface
    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_w[:, self.body_indices] - self.robot_body_pos_w[:, self.body_indices], dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_w[:, self.body_indices], self.robot_body_quat_w[:, self.body_indices]
        ).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        random_frames = torch.randint(
            0, self.motion.time_step_total, (len(env_ids),),
            device=self.device, dtype=torch.long,
        )
        self.time_steps[env_ids] = random_frames

        # Randomize root pose
        root_pos = self.anchor_pos_w.clone()
        root_ori = self.anchor_quat_w.clone()
        root_lin_vel = self.anchor_lin_vel_w.clone()
        root_ang_vel = self.anchor_ang_vel_w.clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

        # Randomize root velocity
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        # Randomize joint positions
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )

        # Write randomized state to sim
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)

    def _update_command(self):
        self.time_steps += 1
        env_ids_to_reset = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids_to_reset)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_anchor_vis"):
                self.goal_anchor_vis = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/World/Visuals/Command/goal/anchor")
                )
                self.current_anchor_vis = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/World/Visuals/Command/current/anchor")
                )
                num_bodies = len(self.body_indices)
                self.goal_body_vis = VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(prim_path="/World/Visuals/Command/goal/bodies")
                )
                self.current_body_vis = VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(prim_path="/World/Visuals/Command/current/bodies")
                )

            self.goal_anchor_vis.set_visibility(True)
            self.current_anchor_vis.set_visibility(True)
            self.goal_body_vis.set_visibility(True)
            self.current_body_vis.set_visibility(True)
        else:
            if hasattr(self, "goal_anchor_vis"):
                self.goal_anchor_vis.set_visibility(False)
                self.current_anchor_vis.set_visibility(False)
                self.goal_body_vis.set_visibility(False)
                self.current_body_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        # Anchor (pelvis) markers
        self.goal_anchor_vis.visualize(self.anchor_pos_w, self.anchor_quat_w)
        self.current_anchor_vis.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)

        # Tracked body markers — flatten (num_envs, num_bodies) into (num_envs * num_bodies)
        bi = self.body_indices
        goal_pos = self.body_pos_w[:, bi].reshape(-1, 3)
        goal_quat = self.body_quat_w[:, bi].reshape(-1, 4)
        curr_pos = self.robot_body_pos_w[:, bi].reshape(-1, 3)
        curr_quat = self.robot_body_quat_w[:, bi].reshape(-1, 4)

        self.goal_body_vis.visualize(goal_pos, goal_quat)
        self.current_body_vis.visualize(curr_pos, curr_quat)


@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand
    asset_name: str = "robot"

    anchor_body_name: str = "pelvis"
    body_names: list[str] = [
        # legs
        "left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link",
        # torso
        "torso_link",
        # arms
        "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    ]

    motion_file: str = "./data/output.npz"

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.1, 0.1)

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/Command/anchor")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/Command/bodies")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
