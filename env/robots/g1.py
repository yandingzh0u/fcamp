from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Holosoma G1 (29 DOF, half-sphere hand) converted from the URDF that ships with the
# holosoma whole-body-tracking task. This replaces the original Unitree factory USD so
# the crawl_slope motion (recorded on this exact robot) tracks correctly.
G1_LOCAL_USD_PATH = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "holosoma_g1"
    / "g1_29dof.usd"
)

# Action / observation / motion joint order. Matches the holosoma motion file's
# `joint_names` (URDF serial-chain order). The env resolves these names against the
# articulation with find_joints(preserve_order=True), so the physical asset order does
# not need to match this list.
G1_29DOF_ASSET_JOINT_NAMES = [
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
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_29DOF_ACTION_NAMES = G1_29DOF_ASSET_JOINT_NAMES


G1_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(G1_LOCAL_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 40.17923847137318,
                ".*_hip_roll_joint": 99.09842777666113,
                ".*_hip_yaw_joint": 40.17923847137318,
                ".*_knee_joint": 99.09842777666113,
            },
            damping={
                ".*_hip_pitch_joint": 2.5578897650279457,
                ".*_hip_roll_joint": 6.3088018534966395,
                ".*_hip_yaw_joint": 2.5578897650279457,
                ".*_knee_joint": 6.3088018534966395,
            },
            armature={
                ".*_hip_pitch_joint": 0.01017752,
                ".*_hip_roll_joint": 0.025101925,
                ".*_hip_yaw_joint": 0.01017752,
                ".*_knee_joint": 0.025101925,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=28.50124619574858,
            damping=1.814445686584846,
            armature=0.00721945,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=28.50124619574858,
            damping=1.814445686584846,
            armature=0.00721945,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=40.17923847137318,
            damping=2.5578897650279457,
            armature=0.01017752,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 14.25062309787429,
                ".*_shoulder_roll_joint": 14.25062309787429,
                ".*_shoulder_yaw_joint": 14.25062309787429,
                ".*_elbow_joint": 14.25062309787429,
                ".*_wrist_roll_joint": 14.25062309787429,
                ".*_wrist_pitch_joint": 16.77832748089279,
                ".*_wrist_yaw_joint": 16.77832748089279,
            },
            damping={
                ".*_shoulder_pitch_joint": 0.907222843292423,
                ".*_shoulder_roll_joint": 0.907222843292423,
                ".*_shoulder_yaw_joint": 0.907222843292423,
                ".*_elbow_joint": 0.907222843292423,
                ".*_wrist_roll_joint": 0.907222843292423,
                ".*_wrist_pitch_joint": 1.06814150219,
                ".*_wrist_yaw_joint": 1.06814150219,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.003609725,
                ".*_shoulder_roll_joint": 0.003609725,
                ".*_shoulder_yaw_joint": 0.003609725,
                ".*_elbow_joint": 0.003609725,
                ".*_wrist_roll_joint": 0.003609725,
                ".*_wrist_pitch_joint": 0.00425,
                ".*_wrist_yaw_joint": 0.00425,
            },
        ),
    },
    prim_path="/World/Robot",
)


def make_g1_cfg(prim_path: str = "/World/Robot", fix_root_link: bool = False) -> ArticulationCfg:
    if not G1_LOCAL_USD_PATH.is_file():
        raise FileNotFoundError(f"G1 USD not found: {G1_LOCAL_USD_PATH}")

    cfg = deepcopy(G1_BASE_CFG)
    cfg.prim_path = prim_path
    cfg.spawn.usd_path = str(G1_LOCAL_USD_PATH)
    cfg.spawn.articulation_props.fix_root_link = fix_root_link
    return cfg
