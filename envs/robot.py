from __future__ import annotations

import torch
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from .spec import (
    G1SceneConfig,
    G1_MIMIC_ACTION_SCALE_VALUES,
    PUSH_INTERVAL_STEP_RANGE,
    STARTUP_BASE_COM_RANGE,
    STARTUP_JOINT_DEFAULT_POS_RANGE,
)
from .robots.g1 import G1_29DOF_ACTION_NAMES, make_g1_cfg
from .tasks import TaskSpec
from .contracts import require_finite_tensors, validate_actions_in_bounds
from engine.config import EnvironmentConfig


class G1Env:
    def __init__(
        self,
        cfg: EnvironmentConfig,
        task: TaskSpec,
        *,
        render: bool = False,
        render_every: int = 1,
        contact_debug_vis: bool = False,
    ):
        if cfg.decimation < 1:
            raise ValueError(f"decimation must be >= 1, got {cfg.decimation}")
        if str(cfg.root_velocity_mode) != "link":
            raise ValueError(
                "Fixed-reward G1 requires environment.root_velocity_mode='link'"
            )
        self.dt = cfg.sim_dt
        self.decimation = int(cfg.decimation)
        self.physics_dt = cfg.sim_dt / float(self.decimation)
        self._render_step_index = 0
        self.render = render
        self.render_every = max(1, int(render_every))

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=self.physics_dt,
            render_interval=self.decimation * self.render_every,
        )


        combine_mode = "average"
        sim_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode=combine_mode,
            restitution_combine_mode=combine_mode,
        )
        sim_cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = G1SceneConfig(num_envs=cfg.num_envs, env_spacing=2.5)
        # Contact sensing is a property of the shared physical platform, not of
        # the algorithm consuming it. Every largebox benchmark method sees the
        # same ground-filtered force tensor.
        use_ground_filter = task.terrain == "plane" and cfg.platform_profile == "g1_largebox_50hz"
        self.uses_ground_contact_filter = use_ground_filter
        if task.terrain == "plane":
            if use_ground_filter:
                scene_cfg.contact_forces.filter_prim_paths_expr = ["/World/ground.*"]
            scene_cfg.terrain = AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    size=(100.0, 100.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                        friction_combine_mode=combine_mode,
                        restitution_combine_mode=combine_mode,
                    ),
                ),
            )
        scene_cfg.robot = make_g1_cfg("{ENV_REGEX_NS}/Robot", fix_root_link=cfg.fix_root_link)
        scene_cfg.contact_forces.update_period = cfg.sim_dt
        scene_cfg.contact_forces.debug_vis = contact_debug_vis
        self.scene = InteractiveScene(scene_cfg)
        if use_ground_filter:
            ground_path = "/World/ground"
            UsdPhysics.RigidBodyAPI.Apply(self.scene.stage.GetPrimAtPath(ground_path))
            UsdPhysics.RigidBodyAPI.Get(
                self.scene.stage, ground_path
            ).GetKinematicEnabledAttr().Set(True)
        self.robot: Articulation = self.scene["robot"]

        self.sim.set_camera_view((2.5, 2.5, 1.6), (0.0, 0.0, 0.8))
        self.sim.reset()
        if cfg.startup_randomization:
            self._apply_official_startup_events()

        action_joint_ids = self.robot.find_joints(G1_29DOF_ACTION_NAMES, preserve_order=True)[0]
        self.action_joint_ids = torch.tensor(action_joint_ids, dtype=torch.long, device=self.sim.device)

        self.default_root_state = self.robot.data.default_root_state.clone()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.default_action_joint_pos = self.default_joint_pos.index_select(1, self.action_joint_ids)
        self.default_action_joint_vel = self.default_joint_vel.index_select(1, self.action_joint_ids)
        self.action_scale = torch.tensor(
            G1_MIMIC_ACTION_SCALE_VALUES,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # Flow-CPS explicitly installs its algorithmic command domain after the
        # policy is built.  Joint-position metadata never defines that domain:
        # normalized actions are PD target commands, not physical joint poses.
        self._policy_action_low: torch.Tensor | None = None
        self._policy_action_high: torch.Tensor | None = None
        self._push_interval_step_range = PUSH_INTERVAL_STEP_RANGE
        min_push, max_push = self._push_interval_step_range
        self.next_push_step = torch.randint(
            min_push,
            max_push + 1,
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        # per-env episode-step of the FIRST interval push (-1 = not yet pushed).
        # Used by validation to decompose the 50-100 cliff into "died before
        # push" (early collapse) vs "pushed then died" (push-recovery failure).
        self.first_push_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._last_interval_push_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._reset_default_pose()

    @property
    def device(self) -> torch.device:
        return self.sim.device

    @property
    def num_envs(self) -> int:
        return int(self.scene.num_envs)

    @property
    def action_dim(self) -> int:
        return int(self.action_joint_ids.numel())

    def enable_strict_action_contract(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> None:
        """Install the policy-owned command domain used by fixed-reward Flow-CPS.

        This domain is deliberately independent of URDF joint-position limits.
        The simulator writer maps the command to a PD target without projection.
        """

        low = torch.as_tensor(low, device=self.device, dtype=torch.float32)
        high = torch.as_tensor(high, device=self.device, dtype=torch.float32)
        if low.shape != (self.action_dim,) or high.shape != (self.action_dim,):
            raise ValueError(
                f"Policy action bounds must have shape {(self.action_dim,)}"
            )
        validate_actions_in_bounds(
            torch.stack((low, high)), low, high, tolerance=0.0
        )
        if bool((low >= high).any()):
            raise ValueError("Every policy action lower bound must be below its upper bound")
        self._policy_action_low = low.detach().clone()
        self._policy_action_high = high.detach().clone()

    def validate_policy_actions(
        self,
        actions: torch.Tensor,
        *,
        tolerance: float = 1.0e-6,
    ) -> None:
        if (
            self._policy_action_low is None
            or self._policy_action_high is None
        ):
            raise RuntimeError("No strict policy action contract is installed")
        validate_actions_in_bounds(
            actions,
            self._policy_action_low,
            self._policy_action_high,
            tolerance=float(tolerance),
        )

    def get_action_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos.index_select(1, self.action_joint_ids)
        joint_vel = self.robot.data.joint_vel.index_select(1, self.action_joint_ids)
        return joint_pos, joint_vel

    def _apply_official_startup_events(self) -> None:
        self._randomize_joint_default_pos()
        self._randomize_torso_com()
        self._randomize_rigid_body_material()

    def _randomize_joint_default_pos(self) -> None:
        low, high = STARTUP_JOINT_DEFAULT_POS_RANGE
        self.robot.data.default_joint_pos += low + (high - low) * torch.rand_like(self.robot.data.default_joint_pos)

    def _randomize_torso_com(self) -> None:
        torso_id = self.robot.body_names.index("torso_link")
        env_ids_cpu = torch.arange(self.num_envs, device="cpu")
        body_ids_cpu = torch.tensor([torso_id], dtype=torch.int, device="cpu")
        range_tensor = torch.tensor(STARTUP_BASE_COM_RANGE, dtype=torch.float32, device="cpu")
        rand_samples = (
            range_tensor[:, 0]
            + (range_tensor[:, 1] - range_tensor[:, 0])
            * torch.rand((self.num_envs, 3), device="cpu")
        ).unsqueeze(1)
        coms = self.robot.root_physx_view.get_coms().clone()
        coms[env_ids_cpu[:, None], body_ids_cpu, :3] += rand_samples
        self.robot.root_physx_view.set_coms(coms, env_ids_cpu)

    def _randomize_rigid_body_material(self) -> None:
        env_ids_cpu = torch.arange(self.num_envs, device="cpu")
        total_num_shapes = self.robot.root_physx_view.max_shapes
        ranges = torch.tensor(
            ((0.3, 1.6), (0.3, 1.2), (0.0, 0.5)),
            dtype=torch.float32,
            device="cpu",
        )
        buckets = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand((64, 3), device="cpu")
        bucket_ids = torch.randint(0, 64, (self.num_envs, total_num_shapes), device="cpu")
        materials = self.robot.root_physx_view.get_material_properties()
        materials[env_ids_cpu] = buckets[bucket_ids]
        self.robot.root_physx_view.set_material_properties(materials, env_ids_cpu)

    def get_observation(self) -> torch.Tensor:
        joint_pos, joint_vel = self.get_action_joint_state()
        return torch.cat([joint_pos, joint_vel], dim=-1)

    def get_mimic_root_velocity_w(self) -> torch.Tensor:
        return self.robot.data.root_link_vel_w

    def write_mimic_root_velocity_to_sim(
        self,
        root_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write the configured world root-link velocity without conversion."""
        require_finite_tensors(
            {"root_velocity": root_velocity},
            context="Root-link velocity writer",
        )
        self.robot.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)

    def _write_robot_state(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids.ndim != 1:
            raise ValueError(f"env_ids must be a 1-D tensor, got {tuple(env_ids.shape)}")
        state_fields = {
            "root_pos": root_pos,
            "root_quat": root_quat,
            "root_lin_vel": root_lin_vel,
            "root_ang_vel": root_ang_vel,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }
        require_finite_tensors(state_fields, context="Robot state writer")

        root_state = self.default_root_state.index_select(0, env_ids).clone()
        root_state[:, :3] = root_pos + self.scene.env_origins.index_select(0, env_ids)
        root_state[:, 3:7] = root_quat
        root_state[:, 7:10] = root_lin_vel
        root_state[:, 10:13] = root_ang_vel

        sim_joint_pos = self.default_joint_pos.index_select(0, env_ids).clone()
        sim_joint_vel = self.default_joint_vel.index_select(0, env_ids).clone()
        sim_joint_pos[:, self.action_joint_ids] = joint_pos
        sim_joint_vel[:, self.action_joint_ids] = joint_vel

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_mimic_root_velocity_to_sim(
            root_state[:, 7:],
            env_ids,
        )
        self.robot.write_joint_state_to_sim(sim_joint_pos, sim_joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(sim_joint_pos, env_ids=env_ids)
        self.scene.write_data_to_sim()

    def _reset_default_pose(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        root_state = self.default_root_state.index_select(0, env_ids).clone()
        self._write_robot_state(
            root_pos=root_state[:, :3],
            root_quat=root_state[:, 3:7],
            root_lin_vel=root_state[:, 7:10],
            root_ang_vel=root_state[:, 10:13],
            joint_pos=self.default_action_joint_pos.index_select(0, env_ids).clone(),
            joint_vel=self.default_action_joint_vel.index_select(0, env_ids).clone(),
            env_ids=env_ids,
        )
        self.scene.reset(env_ids=env_ids)
        self.scene.update(self.physics_dt)
        observation = G1Env.get_observation(self)
        return observation if env_ids.numel() == self.num_envs else observation.index_select(0, env_ids)
