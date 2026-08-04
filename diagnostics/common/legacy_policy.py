"""Strict legacy-domain policy adapters for canonical collection.

Only actors whose exact inference semantics can be reconstructed are admitted.
In particular, the AMP H=1 MLP is supported.  The historical FCAMP actor is
deliberately rejected here: it is an H=4 Flow-CPS policy whose flow-step noise,
chunk scheduler, previous-action anchor, and internal history cannot be
approximated by a one-step MLP adapter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from envs.imitation_data import build_g1_amp_actor_observation

from .canonical_collection import DenseCheckpointRecord
from .manifest import DependencyUnavailable, ProtocolError, sha256_file
from .noise_bank import CollectorMode
from .rollout_collector import ActionRecord, construct_action_record


A_AMP_DOMAIN = "A_amp_official"
A_AMP_SOURCE_COMMIT = "6901e302499711e2207687e1342348a4078330f8"


class _ExactAMPH1Actor(nn.Module):
    """Commit-6901 GaussianActor inference subset, byte-for-byte topology."""

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(512, action_dim)
        self.register_buffer(
            "log_std", torch.full((1, action_dim), math.log(0.05))
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self.trunk(observation))


def _resolve_legacy_paths(
    repo_root: Path,
    spec: Mapping[str, Any],
    domain_key: str,
) -> tuple[Path, Path, Path, Mapping[str, Any], Mapping[str, Any]]:
    raw = spec.get("legacy_inputs") or spec.get("inputs")
    if not isinstance(raw, Mapping):
        raise DependencyUnavailable("suite spec has no legacy_inputs mapping")
    manifest_path = repo_root / str(raw.get("legacy_domain_manifest", ""))
    checkpoint_path = repo_root / str(
        (raw.get("domain_checkpoints") or {}).get(domain_key, "")
    )
    source_path = repo_root / str(
        (raw.get("domain_source_snapshots") or {}).get(domain_key, "")
    )
    for label, path in (
        ("legacy manifest", manifest_path),
        (f"{domain_key} checkpoint", checkpoint_path),
        (f"{domain_key} source snapshot", source_path),
    ):
        if not path.is_file():
            raise DependencyUnavailable(f"{label} is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    domain = (manifest.get("domains") or {}).get(domain_key)
    if not isinstance(domain, Mapping):
        raise ProtocolError(f"legacy manifest has no domain {domain_key}")
    return manifest_path, checkpoint_path, source_path, manifest, domain


def _validate_legacy_files(
    *,
    checkpoint_path: Path,
    source_path: Path,
    domain: Mapping[str, Any],
) -> None:
    expected_checkpoint = str(domain.get("checkpoint_sha256", ""))
    expected_source = str(domain.get("source_snapshot_sha256", ""))
    if sha256_file(checkpoint_path) != expected_checkpoint:
        raise ProtocolError("legacy checkpoint hash differs from frozen manifest")
    if sha256_file(source_path) != expected_source:
        raise ProtocolError("legacy source snapshot hash differs from frozen manifest")


class AAMPPolicyAdapter:
    """Exact source-6901 H=1 AMP policy running in the shared current env."""

    policy_domain = A_AMP_DOMAIN

    def __init__(
        self,
        *,
        actor: _ExactAMPH1Actor,
        normalizer_mean: torch.Tensor,
        normalizer_std: torch.Tensor,
        normalizer_clip: float,
        source_snapshot_sha256: str,
        resolved_config_sha256: str,
    ) -> None:
        self.actor = actor.eval()
        self.normalizer_mean = normalizer_mean
        self.normalizer_std = normalizer_std
        self.normalizer_clip = float(normalizer_clip)
        self.source_snapshot_sha256 = source_snapshot_sha256
        self.resolved_config_sha256 = resolved_config_sha256

    @classmethod
    def load(
        cls,
        *,
        repo_root: str | Path,
        spec: Mapping[str, Any],
        device: torch.device | str,
    ) -> tuple[DenseCheckpointRecord, "AAMPPolicyAdapter"]:
        started = time.monotonic()

        def progress(message: str) -> None:
            print(
                f"[A_amp_load] t={time.monotonic()-started:.3f}s {message}",
                flush=True,
            )

        root = Path(repo_root).expanduser().resolve()
        progress("resolving frozen input identities")
        _, checkpoint_path, source_path, manifest, domain = _resolve_legacy_paths(
            root, spec, "A_amp"
        )
        progress("validating checkpoint/source hashes")
        _validate_legacy_files(
            checkpoint_path=checkpoint_path,
            source_path=source_path,
            domain=domain,
        )
        if str(domain.get("git_commit")) != A_AMP_SOURCE_COMMIT:
            raise ProtocolError("A_amp source commit differs from the frozen 6901 actor")
        shared = manifest.get("shared_task_identity") or {}
        resolved_path = checkpoint_path.parents[1] / "resolved_config.json"
        if not resolved_path.is_file():
            raise DependencyUnavailable(f"A_amp resolved config is missing: {resolved_path}")
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        if str(resolved.get("resolved_config_sha256")) != str(
            domain.get("resolved_config_sha256")
        ):
            raise ProtocolError("A_amp resolved config identity differs from legacy manifest")
        if (
            resolved.get("method") != "amp"
            or int((resolved.get("parameters") or {}).get("horizon", -1)) != 1
            or list((resolved.get("parameters") or {}).get("actor_hidden_dims", []))
            != [1024, 512]
        ):
            raise ProtocolError("A_amp is not the frozen H=1 1024/512 AMP actor")

        # This trusted local payload is large because it also contains replay
        # and optimizer state.  Retain only the tiny actor/normalizer tensors
        # after strict validation so no training state reaches the collector.
        progress("loading trusted checkpoint with mmap")
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        progress("checkpoint metadata loaded; extracting actor tensors")
        if int(payload.get("update_idx", -1)) != int(domain.get("checkpoint_update")):
            raise ProtocolError("A_amp checkpoint update differs from legacy manifest")
        expected_platform = {
            "dataset_sha256": str(shared.get("motion_sha256")),
            "robot_asset_sha256": str(shared.get("robot_asset_sha256")),
            "action_schema_sha256": str(shared.get("action_schema_sha256")),
        }
        if dict(payload.get("platform_identity") or {}) != expected_platform:
            raise ProtocolError("A_amp platform identity differs from legacy manifest")
        state = payload.get("policy")
        if not isinstance(state, Mapping):
            raise ProtocolError("A_amp checkpoint has no policy state mapping")
        actor_keys = {
            "actor.log_std",
            "actor.trunk.0.weight",
            "actor.trunk.0.bias",
            "actor.trunk.2.weight",
            "actor.trunk.2.bias",
            "actor.mean_head.weight",
            "actor.mean_head.bias",
        }
        missing = actor_keys - set(state)
        if missing:
            raise ProtocolError(f"A_amp checkpoint is missing actor tensors: {sorted(missing)}")
        input_dim = int(state["actor.trunk.0.weight"].shape[1])
        action_dim = int(state["actor.mean_head.weight"].shape[0])
        if (input_dim, action_dim) != (237, 29):
            raise ProtocolError(
                f"A_amp actor dimensions changed: input={input_dim}, action={action_dim}"
            )
        actor = _ExactAMPH1Actor(input_dim, action_dim)
        actor_state = {
            key[len("actor.") :]: state[key].detach().clone()
            for key in sorted(actor_keys)
        }
        actor.load_state_dict(actor_state, strict=True)
        log_std = actor.log_std.detach()
        if not bool(torch.allclose(log_std, torch.full_like(log_std, math.log(0.05)))):
            raise ProtocolError("A_amp fixed Gaussian std is not 0.05")
        normalizer_keys = {
            "actor_obs_normalizer._mean",
            "actor_obs_normalizer._std",
        }
        missing_normalizer = normalizer_keys - set(state)
        if missing_normalizer:
            raise ProtocolError(
                f"A_amp checkpoint is missing normalizer tensors: {sorted(missing_normalizer)}"
            )
        mean = state["actor_obs_normalizer._mean"].detach().clone()
        std = state["actor_obs_normalizer._std"].detach().clone()
        if tuple(mean.shape) != (1, 237) or tuple(std.shape) != (1, 237):
            raise ProtocolError("A_amp actor normalizer dimensions changed")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise ProtocolError("A_amp actor normalizer contains NaN or Inf")
        if bool((std <= 0.0).any()):
            raise ProtocolError("A_amp actor normalizer std must be positive")
        del state, payload
        target_device = torch.device(device)
        actor = actor.to(target_device).eval()
        adapter = cls(
            actor=actor,
            normalizer_mean=mean.to(target_device),
            normalizer_std=std.to(target_device),
            normalizer_clip=float(
                (resolved.get("parameters") or {})
                .get("style_prior", {})
                .get("normalizer_clip", 10.0)
            ),
            source_snapshot_sha256=str(domain["source_snapshot_sha256"]),
            resolved_config_sha256=str(domain["resolved_config_sha256"]),
        )
        record = DenseCheckpointRecord(
            checkpoint_id="A_amp_official_u0350",
            path=checkpoint_path,
            sha256=str(domain["checkpoint_sha256"]),
            update=int(domain["checkpoint_update"]),
            lineage_id="legacy-A_amp-official-6901e302",
            policy_domain=A_AMP_DOMAIN,
        )
        progress("actor and normalizer ready")
        return record, adapter

    def reset_branch(
        self,
        *,
        env: Any,
        mode: CollectorMode,
        snapshot_ids: Sequence[str],
    ) -> None:
        del mode, snapshot_ids
        bound = torch.ones(29, device=env.device)
        env.enable_strict_action_contract(-bound, bound)

    @torch.no_grad()
    def action_record(
        self,
        *,
        env: Any,
        observation: torch.Tensor,
        imitation: Mapping[str, torch.Tensor],
        common_epsilon: torch.Tensor,
        mode: CollectorMode,
        common_sigma: float,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        step: int,
    ) -> ActionRecord:
        del env, observation, action_low, action_high, step
        amp_observation = build_g1_amp_actor_observation(
            imitation["agent_physx_raw_frame"]
        )
        normalized = torch.clamp(
            (amp_observation - self.normalizer_mean) / self.normalizer_std,
            -self.normalizer_clip,
            self.normalizer_clip,
        )
        mean = self.actor(normalized)
        std = torch.exp(self.actor.log_std).expand_as(mean)
        low = mean.new_full((mean.shape[-1],), -1.0)
        high = mean.new_full((mean.shape[-1],), 1.0)
        return construct_action_record(
            mean,
            std,
            common_epsilon,
            mode=mode,
            common_sigma=common_sigma,
            action_low=low,
            action_high=high,
        )

    def provenance(self) -> Mapping[str, Any]:
        return {
            "policy_method": "amp_h1_official",
            "policy_source_commit": A_AMP_SOURCE_COMMIT,
            "policy_source_snapshot_sha256": self.source_snapshot_sha256,
            "policy_resolved_config_sha256": self.resolved_config_sha256,
        }


__all__ = [
    "AAMPPolicyAdapter",
    "A_AMP_DOMAIN",
]
