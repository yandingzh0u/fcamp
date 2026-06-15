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
# Support bodies whose contact state is fed to the actor. A crawl is a multi-contact gait:
# the robot supports itself on feet, knees and hands, so the policy must observe all three
# pairs (not just the feet) to coordinate the support transitions. Order is left/right
# foot, left/right knee, left/right hand (wrist).
MIMIC_SUPPORT_CONTACT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
MIMIC_ANCHOR_BODY_NAME = "torso_link"
# Body-name substrings whose ground/self contact is NOT penalized. The crawl is a
# hands+knees+feet gait, so the legitimate support contacts (feet/ankles, knees, wrists,
# foot contact points and the half-sphere hands) are whitelisted; every other body contact
# is still penalized as undesired. This now matches the comment's intent: knee_link is
# included so a correct kneeling crawl is not punished.
CONTACT_ALLOWED_SUBSTRINGS = (
    "ankle_roll_link",
    "knee_link",
    "wrist_yaw_link",
    "foot_contact_point",
    "sphere_hand_link",
)
# 58 reference joint state + 3 anchor pos + 6 anchor ori + 1 anchor z err + 3 root lin vel
# + 6 support contacts (feet/knees/hands) + 3 base ang vel + 29 joint pos rel
# + 29 joint vel rel + 29 last_action (a_{t-1}) + 29 prev_action (a_{t-2}) = 196.
# The decoder derives the latent velocity from (last_action, prev_action) internally; the
# observation carries both action-space anchors, not an action-space velocity.
OBS_DIM = 196
CRITIC_OBS_DIM = 286
UNDESIRED_CONTACT_THRESHOLD = 1.0
ANCHOR_Z_TERMINATION_THRESHOLD = 0.5
# Termination on the RELATIVE tilt angle (radians) between the robot and reference anchor
# orientations: tilt_error = acos(dot(g_ref, g_robot)). Single physically-meaningful quantity
# shared by termination, the tilt_quality reward term and the validation log.
# 1.05 rad ~= 60 deg of robot-vs-reference tilt: 0.6 rad (34 deg) was too strict for a crawl
# with hand-weight-bearing transitions; side-lying is ~90 deg and still terminates. Should be
# re-calibrated to teacher_p99.9_tilt + 5..10 deg (kept within ~0.9-1.2 rad) once teacher
# probe data exists.
ANCHOR_TILT_TERMINATION_THRESHOLD = 1.05
# Sigma (radians) of the tilt_quality reward term. 0.6 keeps a usable gradient out to the
# ~60 deg death line (exp(-(1.05/0.6)^2) ~= 0.046), unlike sigma 0.4 which underflowed there.
TILT_REWARD_SIGMA = 0.6
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
    # No separate flat ground plane: the crawl motion is recorded on the slope terrain, so the
    # slope (spawned per env below) IS the ground. A global z=0 plane would add a phantom floor
    # under the ramp that the robot can rest on, breaking the contact contract.
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
    motion_start_phase_ratio: float = 0.25
    reset_noise: bool = True
    interval_pushes: bool = True
    observation_noise: bool = True
    action_rate_weight: float = 1.0e-1
    track_body_names: tuple[str, ...] = MIMIC_BODY_NAMES
    ee_body_names: tuple[str, ...] = MIMIC_EE_BODY_NAMES
    termination_body_names: tuple[str, ...] = MIMIC_TERMINATION_BODY_NAMES
    foot_body_names: tuple[str, ...] = MIMIC_FOOT_BODY_NAMES
    support_contact_body_names: tuple[str, ...] = MIMIC_SUPPORT_CONTACT_BODY_NAMES
    anchor_body_name: str = MIMIC_ANCHOR_BODY_NAME
    # Number of GRPO generations per group. The trainer keeps all generations inside a
    # group on an identical physical state, so the per-step environment stochasticity
    # (observation noise, interval-push timing and push velocity) must also be shared
    # within the group. The env reads this to broadcast those random draws group-wise.
    # 1 disables sharing (every env independent), which is the correct non-GRPO behaviour.
    group_size: int = 1
    # Deprecated compatibility field. Actor observations use the verified legacy input:
    # current reference only, no future reference frames.
    future_ref_steps: int = 0
