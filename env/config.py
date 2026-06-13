from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .robots.g1 import G1_29DOF_ACTION_NAMES, G1_BASE_CFG, make_g1_cfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_FILE = (
    PROJECT_ROOT
    / "assets"
    / "motions"
    / "g1_crawl"
    / "motion_crawl_slope.npz"
)

# Crawl terrain imported from the holosoma crawl_slope task. The crawl motion is recorded
# in this terrain's frame (the robot climbs the ramp), so the slope is spawned at each
# env origin and acts as the ground for that env.
SLOPE_USD_FILE = PROJECT_ROOT / "assets" / "motions" / "g1_crawl" / "terrain_slope.usd"
SLOPE_OFFSET = (0.0, 0.0, 0.0)

MIMIC_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
MIMIC_EE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
# Bodies whose z-tracking error actually triggers episode termination. The wrists are
# intentionally excluded: the PD reference (zero-residual teacher) physically cannot hold
# the wrists within EE_Z_TERMINATION_THRESHOLD during the crawl, so killing on wrist error
# caps even a perfect teacher at ~35 steps. Wrist tracking is still rewarded (it stays in
# the tracked-body reward set) and wrist z error is logged for diagnostics.
MIMIC_TERMINATION_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
MIMIC_FOOT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
MIMIC_ANCHOR_BODY_NAME = "torso_link"
# Body-name substrings whose ground/self contact is NOT penalized. Mirrors the official
# holosoma crawl whitelist (feet, ankles, wrists, foot contact points, half-sphere hands)
# so the robot may support itself on hands/knees/feet while crawling the slope.
CONTACT_ALLOWED_SUBSTRINGS = (
    "ankle_roll_link",
    "wrist_yaw_link",
    "foot_contact_point",
    "sphere_hand_link",
)
OBS_DIM = 163
CRITIC_OBS_DIM = 286
UNDESIRED_CONTACT_THRESHOLD = 1.0
ANCHOR_Z_TERMINATION_THRESHOLD = 0.5
ANCHOR_ORI_TERMINATION_THRESHOLD = 0.8
EE_Z_TERMINATION_THRESHOLD = 0.35
RESET_ROOT_POSE_RANGE = (
    (-0.05, 0.05),
    (-0.05, 0.05),
    (-0.01, 0.01),
    (-0.1, 0.1),
    (-0.1, 0.1),
    (-0.2, 0.2),
)
VELOCITY_RANGE = (
    (-0.5, 0.5),
    (-0.5, 0.5),
    (-0.2, 0.2),
    (-0.52, 0.52),
    (-0.52, 0.52),
    (-0.78, 0.78),
)
RESET_JOINT_POSITION_RANGE = (-0.1, 0.1)
STARTUP_JOINT_DEFAULT_POS_RANGE = (-0.01, 0.01)
STARTUP_BASE_COM_RANGE = (
    (-0.025, 0.025),
    (-0.05, 0.05),
    (-0.05, 0.05),
)
PUSH_INTERVAL_STEP_RANGE = (50, 150)


def _match_joint_expr(expr: str, joint_name: str) -> bool:
    return re.fullmatch(expr, joint_name) is not None


def _compute_g1_mimic_action_scale_values() -> tuple[float, ...]:
    scale_values: list[float] = []
    for joint_name in G1_29DOF_ACTION_NAMES:
        matched_scale: float | None = None
        for actuator_cfg in G1_BASE_CFG.actuators.values():
            effort_limit = actuator_cfg.effort_limit_sim
            stiffness = actuator_cfg.stiffness
            for joint_expr in actuator_cfg.joint_names_expr:
                if not _match_joint_expr(joint_expr, joint_name):
                    continue
                effort_value = effort_limit[joint_expr] if isinstance(effort_limit, dict) else effort_limit
                stiffness_value = stiffness[joint_expr] if isinstance(stiffness, dict) else stiffness
                matched_scale = 0.25 * float(effort_value) / float(stiffness_value)
                break
            if matched_scale is not None:
                break
        if matched_scale is None:
            raise KeyError(f"Failed to resolve mimic action scale for joint: {joint_name}")
        scale_values.append(matched_scale)
    return tuple(scale_values)


G1_MIMIC_ACTION_SCALE_VALUES = _compute_g1_mimic_action_scale_values()


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    slope = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Slope",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SLOPE_USD_FILE),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.55, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=SLOPE_OFFSET),
    )
    robot = make_g1_cfg("{ENV_REGEX_NS}/Robot", fix_root_link=False)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True,
    )


@dataclass(slots=True)
class EnvConfig:
    device: str = "cuda:0"
    num_envs: int = 1
    sim_dt: float = 0.02
    decimation: int = 4
    render: bool = False
    render_every: int = 1
    contact_debug_vis: bool = False
    action_scale_multiplier: float = 1.0
    env_spacing: float = 2.5
    fix_root_link: bool = False
    startup_randomization: bool = True
    camera_eye: tuple[float, float, float] = (2.5, 2.5, 1.6)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.8)


@dataclass(slots=True)
class MimicEnvConfig(EnvConfig):
    motion_file: str = str(DEFAULT_MOTION_FILE)
    max_episode_steps: int = -1
    motion_start_phase: int = 0
    motion_end_phase: int = -1
    adaptive_motion_sampling: bool = True
    adaptive_uniform_ratio: float = 0.1
    motion_start_phase_ratio: float = 0.25
    adaptive_alpha: float = 0.001
    adaptive_kernel_size: int = 1
    reset_noise: bool = True
    interval_pushes: bool = True
    observation_noise: bool = True
    joint_acc_weight: float = 2.5e-7
    joint_torque_weight: float = 1.0e-5
    action_rate_weight: float = 1.0e-1
    action_accel_weight: float = 0.0
    action_l2_weight: float = 0.0
    track_body_names: tuple[str, ...] = MIMIC_BODY_NAMES
    ee_body_names: tuple[str, ...] = MIMIC_EE_BODY_NAMES
    termination_body_names: tuple[str, ...] = MIMIC_TERMINATION_BODY_NAMES
    foot_body_names: tuple[str, ...] = MIMIC_FOOT_BODY_NAMES
    anchor_body_name: str = MIMIC_ANCHOR_BODY_NAME
    # Deprecated compatibility field. Actor observations use the verified legacy input:
    # current reference only, no future reference frames.
    future_ref_steps: int = 0
