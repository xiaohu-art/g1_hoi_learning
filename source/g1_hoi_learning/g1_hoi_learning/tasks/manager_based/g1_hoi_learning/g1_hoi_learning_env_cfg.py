import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.commands import MotionCommandCfg

##
# Pre-defined configs
##
from g1_hoi_learning.robots.g1_inspire import G1_INSPIRE_CFG  # isort:skip
from g1_hoi_learning.objects.object_cfg import CLOTHESSTAND_CFG  # isort:skip


##
# Scene definition
##


@configclass
class G1HoiLearningSceneCfg(InteractiveSceneCfg):
    """Configuration for a G1 HOI learning scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    object: RigidObjectCfg = CLOTHESSTAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

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
            "{ENV_REGEX_NS}/Robot/L_index_proximal", "{ENV_REGEX_NS}/Robot/L_middle_proximal",
            "{ENV_REGEX_NS}/Robot/L_pinky_proximal", "{ENV_REGEX_NS}/Robot/L_ring_proximal",
            "{ENV_REGEX_NS}/Robot/L_thumb_proximal_base",
            "{ENV_REGEX_NS}/Robot/R_index_proximal", "{ENV_REGEX_NS}/Robot/R_middle_proximal",
            "{ENV_REGEX_NS}/Robot/R_pinky_proximal", "{ENV_REGEX_NS}/Robot/R_ring_proximal",
            "{ENV_REGEX_NS}/Robot/R_thumb_proximal_base",
            "{ENV_REGEX_NS}/Robot/L_index_intermediate", "{ENV_REGEX_NS}/Robot/L_middle_intermediate",
            "{ENV_REGEX_NS}/Robot/L_pinky_intermediate", "{ENV_REGEX_NS}/Robot/L_ring_intermediate",
            "{ENV_REGEX_NS}/Robot/L_thumb_proximal",
            "{ENV_REGEX_NS}/Robot/R_index_intermediate", "{ENV_REGEX_NS}/Robot/R_middle_intermediate",
            "{ENV_REGEX_NS}/Robot/R_pinky_intermediate", "{ENV_REGEX_NS}/Robot/R_ring_intermediate",
            "{ENV_REGEX_NS}/Robot/R_thumb_proximal",
            "{ENV_REGEX_NS}/Robot/L_thumb_intermediate", "{ENV_REGEX_NS}/Robot/R_thumb_intermediate",
            "{ENV_REGEX_NS}/Robot/L_thumb_distal", "{ENV_REGEX_NS}/Robot/R_thumb_distal",
        ],
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = MotionCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        rsi=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range={
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        },
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        # joint_names=["^(?!.*(thumb|index|middle|ring|pinky)).*$"]
        joint_names=[".*"]
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # reference motion
        motion_future_joint_pos = ObsTerm(func=mdp.motion_future_joint_pos, params={"command_name": "motion"})
        motion_future_joint_vel = ObsTerm(func=mdp.motion_future_joint_vel, params={"command_name": "motion"})
        # reference body tracking (current + future)
        motion_future_anchor_pos_b = ObsTerm(func=mdp.motion_future_anchor_pos_b, params={"command_name": "motion"})
        motion_future_anchor_ori_b = ObsTerm(func=mdp.motion_future_anchor_ori_b, params={"command_name": "motion"})
        motion_future_body_pos_b = ObsTerm(func=mdp.motion_future_body_pos_b, params={"command_name": "motion"})
        motion_future_body_ori_b = ObsTerm(func=mdp.motion_future_body_ori_b, params={"command_name": "motion"})
        # reference object tracking (current + future)
        motion_future_obj_pos_b = ObsTerm(func=mdp.motion_future_obj_pos_b, params={"command_name": "motion"})
        motion_future_obj_ori_b = ObsTerm(func=mdp.motion_future_obj_ori_b, params={"command_name": "motion"})
        # reference contact
        motion_future_contact_label = ObsTerm(func=mdp.motion_future_contact_label, params={"command_name": "motion"})
        contact = ObsTerm(func=mdp.contact, params={"sensor_name": "contact_sensor"})
        # robot state
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # object state (current + future)
        object_pos_b = ObsTerm(func=mdp.object_pos_b, params={"command_name": "motion"})
        object_rot_b = ObsTerm(func=mdp.object_rot_b, params={"command_name": "motion"})
        # robot proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic (same as policy)."""

        # reference motion (current + future)
        motion_future_joint_pos = ObsTerm(func=mdp.motion_future_joint_pos, params={"command_name": "motion"})
        motion_future_joint_vel = ObsTerm(func=mdp.motion_future_joint_vel, params={"command_name": "motion"})
        # reference body tracking (current + future)
        motion_future_anchor_pos_b = ObsTerm(func=mdp.motion_future_anchor_pos_b, params={"command_name": "motion"})
        motion_future_anchor_ori_b = ObsTerm(func=mdp.motion_future_anchor_ori_b, params={"command_name": "motion"})
        motion_future_body_pos_b = ObsTerm(func=mdp.motion_future_body_pos_b, params={"command_name": "motion"})
        motion_future_body_ori_b = ObsTerm(func=mdp.motion_future_body_ori_b, params={"command_name": "motion"})
        # reference object tracking (current + future)
        motion_future_obj_pos_b = ObsTerm(func=mdp.motion_future_obj_pos_b, params={"command_name": "motion"})
        motion_future_obj_ori_b = ObsTerm(func=mdp.motion_future_obj_ori_b, params={"command_name": "motion"})
        # contact (reference + sim)
        motion_future_contact_label = ObsTerm(func=mdp.motion_future_contact_label, params={"command_name": "motion"})
        contact = ObsTerm(func=mdp.contact, params={"sensor_name": "contact_sensor"})
        # robot state
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # object state (current + future)
        object_pos_b = ObsTerm(func=mdp.object_pos_b, params={"command_name": "motion"})
        object_rot_b = ObsTerm(func=mdp.object_rot_b, params={"command_name": "motion"})
        # robot proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_anchor_pos = RewTerm(
        func=mdp.motion_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_anchor_ori = RewTerm(
        func=mdp.motion_anchor_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    # object tracking
    object_pos = RewTerm(
        func=mdp.object_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    object_ori = RewTerm(
        func=mdp.object_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    # contact
    contact = RewTerm(
        func=mdp.contact_reward,
        weight=1.0,
        params={
            "command_name": "motion",
            "sensor_name": "contact_sensor",
            "hand_body_names": ["L_.*", "R_.*"],
        },
    )
    # regularization
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*(thumb|index|middle|ring|pinky)).*$"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    object_pos = DoneTerm(
        func=mdp.bad_object_pos,
        params={"command_name": "motion", "threshold": 0.25},
    )
    object_ori = DoneTerm(
        func=mdp.bad_object_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )
    bad_contact = DoneTerm(
        func=mdp.bad_contact,
        params={
            "command_name": "motion",
            "sensor_name": "contact_sensor",
            "hand_body_names": ["L_.*", "R_.*"],
            "max_lost_frames": 20,
        },
    )


##
# Environment configuration
##


@configclass
class G1HoiLearningEnvCfg(ManagerBasedRLEnvCfg):
    """G1 HOI learning environment config."""

    scene: G1HoiLearningSceneCfg = G1HoiLearningSceneCfg(num_envs=4096, env_spacing=4.0)
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 10.0
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**16
