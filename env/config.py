from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
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
# env origin and acts as the SOLE ground for that env. There is intentionally NO global
# flat /World/ground plane: holosoma's load_obj path is a single mesh terrain, not a
# plane+mesh double ground. Spawning both made the robot rest on a phantom flat plane that
# the crawl motion was never recorded against.
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
# Bodies whose z-tracking error triggers episode termination. Aligned with the official
# holosoma BadTrackingZOnly `bad_motion_body_pos_body_names`, which is feet + wrists. The
# wrists were previously dropped here on the theory that the PD teacher could not hold them
# under EE_Z_TERMINATION_THRESHOLD -- but that was measured BEFORE the scene physics was
# aligned to the official contract (slope friction 0.5->1.0, self-collision off->on). With
# the official scene restored we go back to the official termination set and let a teacher
# probe decide feasibility rather than silently weakening the task.
MIMIC_TERMINATION_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
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
OBS_DIM = 171
CRITIC_OBS_DIM = 286
UNDESIRED_CONTACT_THRESHOLD = 1.0
# Official holosoma BadTrackingZOnly thresholds (g1_29dof_wbt_termination):
#   bad_ref_pos_threshold       = 0.5  -> ANCHOR_Z_TERMINATION_THRESHOLD
#   bad_ref_ori_threshold       = 0.8  -> ANCHOR_ORI_TERMINATION_THRESHOLD (z-only grav)
#   bad_motion_body_pos_threshold = 0.25 -> EE_Z_TERMINATION_THRESHOLD
ANCHOR_Z_TERMINATION_THRESHOLD = 0.5
ANCHOR_ORI_TERMINATION_THRESHOLD = 0.8
EE_Z_TERMINATION_THRESHOLD = 0.25
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
    # No global flat plane. The per-env Slope mesh below is the only ground; the crawl
    # motion was recorded climbing this slope. Environment origins are still defined by the
    # scene's GridCloner (env_spacing), so removing the TerrainImporter does not break env
    # placement.
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
    # Holosoma failure-bin sampler + optional causal second stage (see core/config.py for the math).
    adaptive_num_bins: int = 0          # 0 -> auto ⌊num_frames/fps⌋+1 (~1s bins)
    adaptive_uniform_ratio: float = 0.1  # additive floor (official), NOT a fixed mixture weight
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_alpha: float = 0.001
    adaptive_causal_max_ratio: float = 0.5
    adaptive_causal_horizon: int = 0    # 0 -> injected by algorithm (= num_steps_per_env)
    adaptive_causal_decay: float = 0.0  # 0 -> injected by algorithm (= gamma·lambda; gamma for GRPO)
    adaptive_causal_arm_lo: float = 0.5
    adaptive_causal_arm_hi: float = 0.8
    reset_noise: bool = True
    interval_pushes: bool = True
    observation_noise: bool = True
    # GRPO group size. When > 1, observation noise is drawn once per group and shared by
    # all generation branches in that group, so sibling branches that start from the
    # identical group state also observe the identical noisy observation (the only
    # intra-group difference is the SDE action noise). 1 keeps fully independent per-env
    # noise (non-GRPO / single-branch behavior).
    num_generations: int = 1
    # Action-rate penalty weight (official Holosoma WBT action_rate_l2 reward, weight -0.1).
    action_rate_weight: float = 1.0e-1
    track_body_names: tuple[str, ...] = MIMIC_BODY_NAMES
    ee_body_names: tuple[str, ...] = MIMIC_EE_BODY_NAMES
    termination_body_names: tuple[str, ...] = MIMIC_TERMINATION_BODY_NAMES
    foot_body_names: tuple[str, ...] = MIMIC_FOOT_BODY_NAMES
    anchor_body_name: str = MIMIC_ANCHOR_BODY_NAME
