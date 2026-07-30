from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias


Terrain: TypeAlias = Literal["plane", "slope"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    motion_file: Path
    terrain: Terrain
    contact_termination: bool
    allowed_contact_bodies: tuple[str, ...]


TASKS = {
    "crawl_slope": TaskSpec(
        name="crawl_slope",
        motion_file=PROJECT_ROOT / "assets" / "motions" / "g1_crawl" / "motion_crawl_slope.npz",
        terrain="slope",
        # Standard AMP disables contact termination for roll/crawl motions.
        contact_termination=False,
        allowed_contact_bodies=(),
    ),
    "largebox_plane": TaskSpec(
        name="largebox_plane",
        motion_file=(
            PROJECT_ROOT.parent
            / "holosoma"
            / "src"
            / "holosoma"
            / "holosoma"
            / "data"
            / "motions"
            / "g1_29dof"
            / "whole_body_tracking"
            / "sub3_largebox_003_mj.npz"
        ),
        terrain="plane",
        contact_termination=True,
        allowed_contact_bodies=(
            "left_knee_link",
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
        ),
    ),
}


def resolve_task(name: str) -> TaskSpec:
    try:
        task = TASKS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown task {name!r}; available tasks: {sorted(TASKS)}") from exc
    if not task.motion_file.is_file():
        raise FileNotFoundError(f"Motion file for task {name!r} does not exist: {task.motion_file}")
    return task
