"""Replay retargeted G1 motion from a pkl file and save simulation data as npz for RL training.

.. code-block:: bash

    # Usage
    python ./scripts/data_replay.py --input_file ./data/sub16_clothesstand_000_retargeted.pkl --output_file ./data/output.npz --input_fps 30 --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay retargeted G1 motion and save simulation data.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the retargeted pkl file.")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output npz file.")
parser.add_argument("--input_fps", type=int, default=30, help="FPS of the input motion.")
parser.add_argument("--output_fps", type=int, default=50, help="FPS of the output motion.")
parser.add_argument("--num_surface_points", type=int, default=1024, help="Number of surface points to sample on the object mesh.")
parser.add_argument("--contact_distance_threshold", type=float, default=0.05, help="Distance threshold (meters) for binary contact label.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import joblib

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp, quat_from_matrix, quat_unique

from g1_hoi_learning.robots.g1_inspire import G1_INSPIRE_CFG
from g1_hoi_learning.objects.object_cfg import CLOTHESSTAND_CFG

# Joint ordering from pytorch-kinematics URDF parse
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
]


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    """Configuration for a G1 motion replay scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    obj = CLOTHESSTAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        history_length=2,
        track_air_time=True,
        filter_prim_paths_expr=[
            # robot.body_names order
            "{ENV_REGEX_NS}/Robot/pelvis",
            "{ENV_REGEX_NS}/Robot/left_hip_pitch_link", "{ENV_REGEX_NS}/Robot/right_hip_pitch_link",
            "{ENV_REGEX_NS}/Robot/waist_yaw_link",
            "{ENV_REGEX_NS}/Robot/left_hip_roll_link", "{ENV_REGEX_NS}/Robot/right_hip_roll_link",
            "{ENV_REGEX_NS}/Robot/waist_roll_link",
            "{ENV_REGEX_NS}/Robot/left_hip_yaw_link", "{ENV_REGEX_NS}/Robot/right_hip_yaw_link",
            "{ENV_REGEX_NS}/Robot/torso_link",
            "{ENV_REGEX_NS}/Robot/left_knee_link", "{ENV_REGEX_NS}/Robot/right_knee_link",
            "{ENV_REGEX_NS}/Robot/left_shoulder_pitch_link", "{ENV_REGEX_NS}/Robot/right_shoulder_pitch_link",
            "{ENV_REGEX_NS}/Robot/left_ankle_pitch_link", "{ENV_REGEX_NS}/Robot/right_ankle_pitch_link",
            "{ENV_REGEX_NS}/Robot/left_shoulder_roll_link", "{ENV_REGEX_NS}/Robot/right_shoulder_roll_link",
            "{ENV_REGEX_NS}/Robot/left_ankle_roll_link", "{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
            "{ENV_REGEX_NS}/Robot/left_shoulder_yaw_link", "{ENV_REGEX_NS}/Robot/right_shoulder_yaw_link",
            "{ENV_REGEX_NS}/Robot/left_elbow_link", "{ENV_REGEX_NS}/Robot/right_elbow_link",
            "{ENV_REGEX_NS}/Robot/left_wrist_roll_link", "{ENV_REGEX_NS}/Robot/right_wrist_roll_link",
            "{ENV_REGEX_NS}/Robot/left_wrist_pitch_link", "{ENV_REGEX_NS}/Robot/right_wrist_pitch_link",
            "{ENV_REGEX_NS}/Robot/left_wrist_yaw_link", "{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
            "{ENV_REGEX_NS}/Robot/L_index_proximal", "{ENV_REGEX_NS}/Robot/L_middle_proximal", "{ENV_REGEX_NS}/Robot/L_pinky_proximal", "{ENV_REGEX_NS}/Robot/L_ring_proximal", "{ENV_REGEX_NS}/Robot/L_thumb_proximal_base",
            "{ENV_REGEX_NS}/Robot/R_index_proximal", "{ENV_REGEX_NS}/Robot/R_middle_proximal", "{ENV_REGEX_NS}/Robot/R_pinky_proximal", "{ENV_REGEX_NS}/Robot/R_ring_proximal", "{ENV_REGEX_NS}/Robot/R_thumb_proximal_base",
            "{ENV_REGEX_NS}/Robot/L_index_intermediate", "{ENV_REGEX_NS}/Robot/L_middle_intermediate", "{ENV_REGEX_NS}/Robot/L_pinky_intermediate", "{ENV_REGEX_NS}/Robot/L_ring_intermediate", "{ENV_REGEX_NS}/Robot/L_thumb_proximal",
            "{ENV_REGEX_NS}/Robot/R_index_intermediate", "{ENV_REGEX_NS}/Robot/R_middle_intermediate", "{ENV_REGEX_NS}/Robot/R_pinky_intermediate", "{ENV_REGEX_NS}/Robot/R_ring_intermediate", "{ENV_REGEX_NS}/Robot/R_thumb_proximal",
            "{ENV_REGEX_NS}/Robot/L_thumb_intermediate", "{ENV_REGEX_NS}/Robot/R_thumb_intermediate",
            "{ENV_REGEX_NS}/Robot/L_thumb_distal", "{ENV_REGEX_NS}/Robot/R_thumb_distal",
        ],
    )


class MotionLoader:
    def __init__(self, data: dict, input_fps: int, output_fps: int, device: torch.device):
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self._load_motion(data)
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self, data: dict):
        # Root: (T, 3) position, (T, 4) quaternion wxyz
        self.root_poss_input = torch.from_numpy(data["root_trans"]).to(self.device, dtype=torch.float32)
        self.root_rots_input = torch.from_numpy(data["root_quat"]).to(self.device, dtype=torch.float32)

        # Joint positions: (T, 53)
        self.dof_poss_input = torch.from_numpy(data["joint_pos"]).to(self.device, dtype=torch.float32)

        # Object
        obj = data["object"]
        self.object_poss_input = torch.from_numpy(obj["trans"]).to(self.device, dtype=torch.float32)
        self.object_rots_input = quat_unique(
            quat_from_matrix(torch.from_numpy(obj["rot"]).to(self.device, dtype=torch.float32))
        )

        # Contact labels
        self.contact_label_input = torch.from_numpy(data["contact_label"]).to(self.device, dtype=torch.float32)

        self.input_frames = self.root_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded, duration: {self.duration:.2f} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        idx0, idx1, blend = self._compute_frame_blend(times)

        self.root_poss = self._lerp(self.root_poss_input[idx0], self.root_poss_input[idx1], blend.unsqueeze(1))
        self.root_rots = self._slerp(self.root_rots_input[idx0], self.root_rots_input[idx1], blend)
        self.dof_poss = self._lerp(self.dof_poss_input[idx0], self.dof_poss_input[idx1], blend.unsqueeze(1))
        self.object_poss = self._lerp(self.object_poss_input[idx0], self.object_poss_input[idx1], blend.unsqueeze(1))
        self.object_rots = self._slerp(self.object_rots_input[idx0], self.object_rots_input[idx1], blend)

        # Contact labels: nearest-neighbor interpolation (binary, don't blend)
        nearest = torch.where(blend < 0.5, idx0, idx1)
        self.contact_labels = self.contact_label_input[nearest]  # (output_frames, n_sensor_bodies)

        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps},"
            f" output frames: {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _compute_frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - idx0
        return idx0, idx1, blend

    def _compute_velocities(self):
        self.root_lin_vels = torch.gradient(self.root_poss, spacing=self.output_dt, dim=0)[0]
        self.dof_vels = torch.gradient(self.dof_poss, spacing=self.output_dt, dim=0)[0]
        self.root_ang_vels = self._so3_derivative(self.root_rots, self.output_dt)
        self.object_lin_vels = torch.gradient(self.object_poss, spacing=self.output_dt, dim=0)[0]
        self.object_ang_vels = self._so3_derivative(self.object_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        state = (
            self.root_poss[self.current_idx : self.current_idx + 1],
            self.root_rots[self.current_idx : self.current_idx + 1],
            self.root_lin_vels[self.current_idx : self.current_idx + 1],
            self.root_ang_vels[self.current_idx : self.current_idx + 1],
            self.dof_poss[self.current_idx : self.current_idx + 1],
            self.dof_vels[self.current_idx : self.current_idx + 1],
            self.object_poss[self.current_idx : self.current_idx + 1],
            self.object_rots[self.current_idx : self.current_idx + 1],
            self.object_lin_vels[self.current_idx : self.current_idx + 1],
            self.object_ang_vels[self.current_idx : self.current_idx + 1],
            self.contact_labels[self.current_idx],
        )
        self.current_idx += 1
        reset_flag = self.current_idx >= self.output_frames
        if reset_flag:
            self.current_idx = 0
        return state, reset_flag


def process_motion(sim: SimulationContext, scene: InteractiveScene, joint_indices: list, data: dict) -> dict:
    """Replay a single motion and collect simulation data."""
    motion = MotionLoader(
        data=data,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
    )

    robot = scene["robot"]
    obj_asset = scene["obj"]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "object_pos_w": [],
        "object_quat_w": [],
        "object_lin_vel_w": [],
        "object_ang_vel_w": [],
        "contact_label": [],
    }

    while simulation_app.is_running():
        (
            (
                root_pos, root_rot, root_lin_vel, root_ang_vel,
                dof_pos, dof_vel,
                obj_pos, obj_rot, obj_lin_vel, obj_ang_vel,
                contact_label,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # Set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = root_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = root_rot
        root_states[:, 7:10] = root_lin_vel
        root_states[:, 10:] = root_ang_vel
        robot.write_root_state_to_sim(root_states)

        # Set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indices] = dof_pos
        joint_vel[:, joint_indices] = dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Set object state
        obj_root_state = obj_asset.data.default_root_state.clone()
        obj_root_state[:, :3] = obj_pos
        obj_root_state[:, :2] += scene.env_origins[:, :2]
        obj_root_state[:, 3:7] = obj_rot
        obj_root_state[:, 7:10] = obj_lin_vel
        obj_root_state[:, 10:] = obj_ang_vel
        obj_asset.write_root_state_to_sim(obj_root_state)

        sim.render()
        scene.update(sim.get_physics_dt())

        # Record robot data
        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        # Record object data from scene entity
        log["object_pos_w"].append(obj_asset.data.body_pos_w[0, 0].cpu().numpy().copy())
        log["object_quat_w"].append(obj_asset.data.body_quat_w[0, 0].cpu().numpy().copy())
        log["object_lin_vel_w"].append(obj_asset.data.body_lin_vel_w[0, 0].cpu().numpy().copy())
        log["object_ang_vel_w"].append(obj_asset.data.body_ang_vel_w[0, 0].cpu().numpy().copy())

        # Record contact labels 
        log["contact_label"].append(contact_label.cpu().numpy().copy())

        if reset_flag:
            for k in list(log.keys()):
                if k != "fps":
                    log[k] = np.stack(log[k], axis=0)
            print("[INFO]: Motion processed successfully")
            break

    return log


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    # Map pkl joint ordering to robot joint indices
    robot = scene["robot"]

    joint_indices, _ = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)

    # Load retargeted data
    print(f"[INFO]: Loading motion from {args_cli.input_file}")
    data = joblib.load(args_cli.input_file)

    # Reorder contact labels from pkl (pytorch-kinematics order) to robot body order
    pk_body_names = {name: i for i, name in enumerate(data["link_names"])}
    reorder_idx = [pk_body_names[name] for name in robot.body_names]
    data["contact_label"] = data["contact_label"][:, reorder_idx]

    log = process_motion(sim, scene, joint_indices, data)

    print(f"[INFO]: Saving to {args_cli.output_file}")
    np.savez_compressed(args_cli.output_file, **log)
    print("[INFO]: Done")


if __name__ == "__main__":
    main()
    simulation_app.close()