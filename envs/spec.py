from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .imitation_data import G1_IMITATION_FRAME_DIM
from .robots.g1 import (
    G1_29DOF_ACTION_NAMES,
    G1_LOCAL_URDF_PATH,
    make_g1_cfg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CRAWL_TERRAIN_USD = PROJECT_ROOT / "assets" / "motions" / "g1_crawl" / "terrain_slope.usd"

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


MIMIC_TERMINATION_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
MIMIC_ANCHOR_BODY_NAME = "torso_link"

OBS_DIM = (
    G1_IMITATION_FRAME_DIM
    - 2  # discard global root x/y
)
# Standard AMP uses the same self-state observation for Actor and Critic.
CRITIC_OBS_DIM = OBS_DIM


ANCHOR_Z_TERMINATION_THRESHOLD = 0.5
ANCHOR_ORI_TERMINATION_THRESHOLD = 0.8
EE_Z_TERMINATION_THRESHOLD = 0.25
def _compute_g1_amp_action_scale_values() -> tuple[float, ...]:
    """Return MimicKit's zero-centered physical action half ranges.

    For one-dimensional position-controlled joints MimicKit sets the physical
    action interval to ``[-1.4*max(|lower|,|upper|),
    +1.4*max(|lower|,|upper|)]`` and then normalizes that interval to
    ``[-1,1]`` for the Gaussian policy.
    """

    root = ET.parse(G1_LOCAL_URDF_PATH).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[joint.attrib["name"]] = (
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
        )
    scale_values: list[float] = []
    for joint_name in G1_29DOF_ACTION_NAMES:
        try:
            lower, upper = limits[joint_name]
        except KeyError as exc:
            raise KeyError(
                f"URDF action joint {joint_name!r} has no finite position limit"
            ) from exc
        scale_values.append(1.4 * max(abs(lower), abs(upper)))
    return tuple(scale_values)


G1_AMP_ACTION_SCALE_VALUES = _compute_g1_amp_action_scale_values()


@configclass
class G1SceneConfig(InteractiveSceneCfg):


    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    terrain = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Slope",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CRAWL_TERRAIN_USD),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.55, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    robot = make_g1_cfg("{ENV_REGEX_NS}/Robot", fix_root_link=False)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True,
    )
