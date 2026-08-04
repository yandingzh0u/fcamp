"""Isaac-free observation layout discovery for the diagnostic suite.

The production environment owns the observation order.  This module reads the
``torch.cat`` expressions in :mod:`envs.observation` instead of duplicating
that order in diagnostic scripts.  Widths are resolved from environment
cardinalities (or an explicitly supplied runtime width mapping), so no probe
needs to know that a particular feature starts at a hard-coded offset.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Mapping, Sequence


class ObservationSpecError(ValueError):
    """Raised when the source layout and the declared cardinalities disagree."""


class FeatureRole(str, Enum):
    REFERENCE = "reference"
    PROPRIO = "proprio"


# This is the single semantic policy for reference leakage.  Widths and
# offsets intentionally do not appear here.
_REFERENCE_TERMS = frozenset(
    {
        "reference_joint_state",
        "motion_anchor_pos_b",
        "motion_anchor_ori_b",
        "anchor_z_err",
        "termination_z_err",
    }
)

_PROPRIO_TERMS = frozenset(
    {
        "root_lin_vel_b",
        "foot_contact",
        "termination_contact",
        "base_ang_vel",
        "root_ang_vel_b",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
        "robot_body_pos_b",
        "robot_body_ori_b",
    }
)


def _feature_role(name: str) -> FeatureRole:
    if name in _REFERENCE_TERMS:
        return FeatureRole.REFERENCE
    if name in _PROPRIO_TERMS:
        return FeatureRole.PROPRIO
    # Never default an unfamiliar production term to proprio: that could leak
    # a newly added reference feature into actor_no_reference without warning.
    raise ObservationSpecError(
        f"observation term {name!r} has no audited reference/proprio role"
    )


@dataclass(frozen=True)
class ObservationCardinality:
    """Counts needed to turn symbolic observation terms into widths."""

    action_joint_count: int
    termination_body_count: int
    termination_contact_body_count: int
    foot_body_count: int
    track_body_count: int

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or value <= 0:
                raise ObservationSpecError(f"{name} must be a positive integer")

    def term_widths(self) -> dict[str, int]:
        joints = self.action_joint_count
        spatial = 3
        rotation_6d = 2 * spatial
        return {
            "reference_joint_state": 2 * joints,
            "motion_anchor_pos_b": spatial,
            "motion_anchor_ori_b": rotation_6d,
            "anchor_z_err": 1,
            "termination_z_err": self.termination_body_count,
            "root_lin_vel_b": spatial,
            "foot_contact": self.foot_body_count,
            "termination_contact": self.termination_contact_body_count,
            "base_ang_vel": spatial,
            "root_ang_vel_b": spatial,
            "joint_pos_rel": joints,
            "joint_vel_rel": joints,
            "last_action": joints,
            "robot_body_pos_b": spatial * self.track_body_count,
            "robot_body_ori_b": rotation_6d * self.track_body_count,
        }


@dataclass(frozen=True)
class ObservationTerm:
    name: str
    start: int
    stop: int
    role: FeatureRole
    source_expression: str

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class ObservationSpec:
    stream: str
    source_function: str
    source_sha256: str
    terms: tuple[ObservationTerm, ...]

    def __post_init__(self) -> None:
        if self.stream not in {"actor", "critic"}:
            raise ObservationSpecError(f"unknown observation stream: {self.stream}")
        cursor = 0
        seen: set[str] = set()
        for term in self.terms:
            if term.name in seen:
                raise ObservationSpecError(f"duplicate observation term: {term.name}")
            if term.start != cursor or term.stop <= term.start:
                raise ObservationSpecError("observation terms must be contiguous and nonempty")
            seen.add(term.name)
            cursor = term.stop

    @property
    def total_dim(self) -> int:
        return self.terms[-1].stop if self.terms else 0

    @property
    def named_slices(self) -> dict[str, slice]:
        return {term.name: term.slice for term in self.terms}

    @property
    def reference_terms(self) -> tuple[ObservationTerm, ...]:
        return tuple(term for term in self.terms if term.role is FeatureRole.REFERENCE)

    @property
    def proprio_terms(self) -> tuple[ObservationTerm, ...]:
        return tuple(term for term in self.terms if term.role is FeatureRole.PROPRIO)

    def term(self, name: str) -> ObservationTerm:
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {
            "stream": self.stream,
            "source_function": self.source_function,
            "source_sha256": self.source_sha256,
            "total_dim": self.total_dim,
            "terms": [
                {
                    "name": term.name,
                    "start": term.start,
                    "stop": term.stop,
                    "width": term.width,
                    "role": term.role.value,
                    "source_expression": term.source_expression,
                }
                for term in self.terms
            ],
        }


def _attribute_path(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [*_attribute_path(node.value), node.attr]
    return []


def _unwrap_layout_expression(node: ast.AST) -> ast.AST:
    """Strip shape-only calls such as ``reshape`` from a cat element."""

    current = node
    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        if current.func.attr not in {"reshape", "flatten", "view"}:
            break
        current = current.func.value
    return current


def _canonical_term_name(node: ast.AST) -> str:
    unwrapped = _unwrap_layout_expression(node)
    path = _attribute_path(unwrapped)
    if not path:
        raise ObservationSpecError(
            "observation torch.cat elements must be named tensors or shape-only views; "
            f"got {ast.unparse(node)!r}"
        )
    # Local variables use their only path element.  Object attributes use the
    # final attribute because that is the physical tensor name in this module.
    return path[-1]


def _find_function(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            if isinstance(node, ast.AsyncFunctionDef):
                raise ObservationSpecError(f"{name} unexpectedly became async")
            return node
    raise ObservationSpecError(f"could not find {name} in observation source")


def _is_torch_cat(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "cat"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "torch"
    )


def _cat_elements(function: ast.FunctionDef) -> Sequence[ast.AST]:
    candidates: list[ast.Call] = []
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "observation" for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and _is_torch_cat(value):
            candidates.append(value)
    if len(candidates) != 1:
        raise ObservationSpecError(
            "expected exactly one `observation = torch.cat(...)` assignment in "
            f"{function.name}, found {len(candidates)}"
        )
    call = candidates[0]
    if not call.args or not isinstance(call.args[0], (ast.List, ast.Tuple)):
        raise ObservationSpecError("torch.cat observation terms must be a literal list or tuple")
    return call.args[0].elts


def observation_term_order(source_path: str | Path, *, stream: str) -> tuple[tuple[str, str], ...]:
    """Return ``(canonical_name, source_expression)`` in production order."""

    path = Path(source_path)
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    function_name = "build_observation" if stream == "actor" else "build_critic_observation"
    if stream not in {"actor", "critic"}:
        raise ObservationSpecError(f"unknown observation stream: {stream}")
    function = _find_function(module, function_name)
    return tuple(
        (_canonical_term_name(node), ast.unparse(node))
        for node in _cat_elements(function)
    )


def derive_observation_spec(
    source_path: str | Path,
    *,
    stream: str,
    cardinality: ObservationCardinality | None = None,
    term_widths: Mapping[str, int] | None = None,
) -> ObservationSpec:
    """Derive named slices from production source and runtime dimensions.

    ``term_widths`` is useful when a collector has the actual pre-concatenation
    tensors.  Otherwise ``cardinality`` derives every current actor/critic term
    without importing Isaac Lab.
    """

    if (cardinality is None) == (term_widths is None):
        raise ObservationSpecError("provide exactly one of cardinality or term_widths")
    widths = dict(term_widths if term_widths is not None else cardinality.term_widths())
    source = Path(source_path).read_bytes()
    ordered = observation_term_order(source_path, stream=stream)
    cursor = 0
    terms: list[ObservationTerm] = []
    for name, expression in ordered:
        if name not in widths:
            raise ObservationSpecError(f"no width is available for observation term {name!r}")
        width = widths[name]
        if not isinstance(width, int) or width <= 0:
            raise ObservationSpecError(f"width for {name!r} must be a positive integer")
        role = _feature_role(name)
        terms.append(
            ObservationTerm(
                name=name,
                start=cursor,
                stop=cursor + width,
                role=role,
                source_expression=expression,
            )
        )
        cursor += width
    return ObservationSpec(
        stream=stream,
        source_function="build_observation" if stream == "actor" else "build_critic_observation",
        source_sha256=hashlib.sha256(source).hexdigest(),
        terms=tuple(terms),
    )


def derive_observation_specs(
    source_path: str | Path,
    cardinality: ObservationCardinality,
) -> dict[str, ObservationSpec]:
    return {
        stream: derive_observation_spec(
            source_path,
            stream=stream,
            cardinality=cardinality,
        )
        for stream in ("actor", "critic")
    }


def widths_from_term_tensors(term_tensors: Mapping[str, object]) -> dict[str, int]:
    """Extract last-axis widths from tensor-like objects without importing Isaac."""

    widths: dict[str, int] = {}
    for name, value in term_tensors.items():
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) < 1:
            raise ObservationSpecError(f"term {name!r} has no tensor-like shape")
        widths[name] = int(shape[-1])
    return widths


def referenced_names(spec: ObservationSpec) -> frozenset[str]:
    return frozenset(term.name for term in spec.reference_terms)


def proprio_names(spec: ObservationSpec) -> frozenset[str]:
    return frozenset(term.name for term in spec.proprio_terms)
