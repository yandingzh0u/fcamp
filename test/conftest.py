from __future__ import annotations

import sys
import types
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _Cfg:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _configclass(cls):
    if "__init__" not in cls.__dict__:
        def __init__(self, **kwargs):
            for name, value in cls.__dict__.items():
                if not name.startswith("_") and not callable(value):
                    setattr(self, name, value)
            for key, value in kwargs.items():
                setattr(self, key, value)
        cls.__init__ = __init__
    return cls


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs = _normalize_quat(lhs)
    rhs = _normalize_quat(rhs)
    w1, x1, y1, z1 = lhs.unbind(dim=-1)
    w2, x2, y2, z2 = rhs.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_inv(quat: torch.Tensor) -> torch.Tensor:
    inv = quat.clone()
    inv[..., 1:] = -inv[..., 1:]
    return inv / quat.square().sum(dim=-1, keepdim=True).clamp(min=1e-8)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat(quat)
    qvec = quat[..., 1:]
    uv = torch.cross(qvec, vec, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vec + 2.0 * (quat[..., :1] * uv + uuv)


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return _quat_apply(_quat_inv(quat), vec)


def _quat_error_magnitude(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs = _normalize_quat(lhs)
    rhs = _normalize_quat(rhs)
    dot = torch.sum(lhs * rhs, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat(quat)
    w, x, y, z = quat.unbind(dim=-1)
    two = 2.0
    row0 = torch.stack([1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)], dim=-1)
    row1 = torch.stack([two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)], dim=-1)
    row2 = torch.stack([two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat(quat)
    w, x, y, z = quat.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    result = torch.zeros_like(quat)
    result[..., 0] = torch.cos(yaw * 0.5)
    result[..., 3] = torch.sin(yaw * 0.5)
    return result


def _subtract_frame_transforms(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel_pos = _quat_apply_inverse(parent_quat, child_pos - parent_pos)
    rel_quat = _quat_mul(_quat_inv(parent_quat), child_quat)
    return rel_pos, rel_quat


def _install_isaaclab_stubs() -> None:
    isaaclab = types.ModuleType("isaaclab")
    sim = types.ModuleType("isaaclab.sim")
    assets = types.ModuleType("isaaclab.assets")
    scene = types.ModuleType("isaaclab.scene")
    sensors = types.ModuleType("isaaclab.sensors")
    terrains = types.ModuleType("isaaclab.terrains")
    utils = types.ModuleType("isaaclab.utils")
    math_mod = types.ModuleType("isaaclab.utils.math")
    actuators = types.ModuleType("isaaclab.actuators")

    for name in (
        "RigidBodyMaterialCfg",
        "MdlFileCfg",
        "DistantLightCfg",
        "DomeLightCfg",
        "UsdFileCfg",
        "RigidBodyPropertiesCfg",
        "ArticulationRootPropertiesCfg",
    ):
        setattr(sim, name, type(name, (_Cfg,), {}))

    class SimulationCfg(_Cfg):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.physx = _Cfg(gpu_max_rigid_patch_count=0)

    class SimulationContext(_Cfg):
        pass

    sim.SimulationCfg = SimulationCfg
    sim.SimulationContext = SimulationContext

    class ArticulationCfg(_Cfg):
        class InitialStateCfg(_Cfg):
            pass

    assets.AssetBaseCfg = type("AssetBaseCfg", (_Cfg,), {})
    assets.Articulation = type("Articulation", (_Cfg,), {})
    assets.ArticulationCfg = ArticulationCfg
    scene.InteractiveSceneCfg = type("InteractiveSceneCfg", (_Cfg,), {})
    scene.InteractiveScene = type("InteractiveScene", (_Cfg,), {})
    sensors.ContactSensorCfg = type("ContactSensorCfg", (_Cfg,), {})
    terrains.TerrainImporterCfg = type("TerrainImporterCfg", (_Cfg,), {})
    utils.configclass = _configclass
    actuators.ImplicitActuatorCfg = type("ImplicitActuatorCfg", (_Cfg,), {})

    math_mod.quat_error_magnitude = _quat_error_magnitude
    math_mod.quat_apply_inverse = _quat_apply_inverse
    math_mod.quat_apply = _quat_apply
    math_mod.quat_inv = _quat_inv
    math_mod.quat_mul = _quat_mul
    math_mod.subtract_frame_transforms = _subtract_frame_transforms
    math_mod.yaw_quat = _yaw_quat
    math_mod.matrix_from_quat = _matrix_from_quat
    math_mod.quat_from_euler_xyz = lambda roll, pitch, yaw: torch.stack(
        [
            torch.cos(roll * 0.5) * torch.cos(pitch * 0.5) * torch.cos(yaw * 0.5)
            + torch.sin(roll * 0.5) * torch.sin(pitch * 0.5) * torch.sin(yaw * 0.5),
            torch.sin(roll * 0.5) * torch.cos(pitch * 0.5) * torch.cos(yaw * 0.5)
            - torch.cos(roll * 0.5) * torch.sin(pitch * 0.5) * torch.sin(yaw * 0.5),
            torch.cos(roll * 0.5) * torch.sin(pitch * 0.5) * torch.cos(yaw * 0.5)
            + torch.sin(roll * 0.5) * torch.cos(pitch * 0.5) * torch.sin(yaw * 0.5),
            torch.cos(roll * 0.5) * torch.cos(pitch * 0.5) * torch.sin(yaw * 0.5)
            - torch.sin(roll * 0.5) * torch.sin(pitch * 0.5) * torch.cos(yaw * 0.5),
        ],
        dim=-1,
    )

    isaaclab.sim = sim
    isaaclab.assets = assets
    isaaclab.scene = scene
    isaaclab.sensors = sensors
    isaaclab.terrains = terrains
    isaaclab.utils = utils
    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.sim"] = sim
    sys.modules["isaaclab.assets"] = assets
    sys.modules["isaaclab.scene"] = scene
    sys.modules["isaaclab.sensors"] = sensors
    sys.modules["isaaclab.terrains"] = terrains
    sys.modules["isaaclab.utils"] = utils
    sys.modules["isaaclab.utils.math"] = math_mod
    sys.modules["isaaclab.actuators"] = actuators


_install_isaaclab_stubs()
