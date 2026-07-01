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


TASKS = {
    "crawl_slope": TaskSpec(
        name="crawl_slope",
        motion_file=PROJECT_ROOT / "assets" / "motions" / "g1_crawl" / "motion_crawl_slope.npz",
        terrain="slope",
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
