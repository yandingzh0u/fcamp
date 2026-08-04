"""Frozen offline reproduction of the repository's standard AMP critic.

The equations and initialization mirror commit
``6901e302499711e2207687e1342348a4078330f8``.  This module is diagnostic
only; it is not wired into PPO and does not alter the current fixed-reward
branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from .manifest import ProtocolError, canonical_sha256


SOURCE_COMMIT = "6901e302499711e2207687e1342348a4078330f8"


class OfflineStyleDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        layers: list[nn.Module] = []
        previous = self.input_dim
        for width in self.hidden_dims:
            linear = nn.Linear(previous, int(width))
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.ReLU()))
            previous = int(width)
        self.trunk = nn.Sequential(*layers)
        self.logit = nn.Linear(previous, 1)
        nn.init.uniform_(self.logit.weight, -1.0, 1.0)
        nn.init.zeros_(self.logit.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.logit(self.trunk(observations)).squeeze(-1)


@dataclass(slots=True)
class OfflineAMPFit:
    model: OfflineStyleDiscriminator
    mean: torch.Tensor
    variance: torch.Tensor
    clip: float
    reward_scale: float
    reward_epsilon: float
    seed: int
    history: list[dict[str, float]]
    source_commit: str = SOURCE_COMMIT

    def _normalized(self, values: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.mean.device)
        std = torch.sqrt(torch.clamp(self.variance, min=1.0e-8))
        return torch.clamp((tensor - self.mean) / std, -self.clip, self.clip)

    @torch.no_grad()
    def logits(self, values: np.ndarray, *, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        normalized = self._normalized(values)
        parts = [self.model(chunk).cpu() for chunk in normalized.split(int(batch_size))]
        return torch.cat(parts).numpy().astype(np.float64, copy=False)

    def rewards(self, values: np.ndarray, *, batch_size: int = 4096) -> np.ndarray:
        logits = self.logits(values, batch_size=batch_size)
        unclipped = self.reward_scale * np.logaddexp(0.0, logits)
        maximum = -self.reward_scale * np.log(self.reward_epsilon)
        return np.minimum(unclipped, maximum)


OFFLINE_AMP_ARTIFACT_SCHEMA = "largebox_offline_amp_fit_v1"


def directed_offline_amp_protocol(
    base_protocol: Mapping[str, Any],
    *,
    negative_domain: str,
    positive_domain: str,
) -> dict[str, Any]:
    """Bind frozen AMP hyperparameters to one explicit ordered domain edge."""

    if str(base_protocol.get("source_commit")) != SOURCE_COMMIT:
        raise ProtocolError("directed AMP base protocol has the wrong source commit")
    negative = str(negative_domain)
    positive = str(positive_domain)
    if not negative or not positive or negative == positive:
        raise ProtocolError("directed AMP edge must name two distinct domains")
    return {
        **dict(base_protocol),
        "negative_domain": negative,
        "positive_domains": [positive],
        "directed_edge_id": f"{negative}_to_{positive}",
        "protocol_role": "directed_offline_standard_amp_reward_probe",
        "base_protocol_sha256": canonical_sha256(dict(base_protocol)),
    }


def _fit_payload(
    fit: OfflineAMPFit,
    *,
    protocol: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    training_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a tensor-only-safe, self-auditing critic artifact payload."""

    if fit.source_commit != SOURCE_COMMIT:
        raise ProtocolError("offline AMP fit source commit is not the frozen baseline")
    if str(protocol.get("source_commit")) != SOURCE_COMMIT:
        raise ProtocolError("offline AMP artifact protocol has the wrong source commit")
    if not feature_contract:
        raise ProtocolError("offline AMP artifact requires an explicit feature contract")
    input_dim = int(fit.mean.numel())
    if (
        fit.variance.shape != fit.mean.shape
        or fit.model.input_dim != input_dim
        or not bool(torch.isfinite(fit.mean).all())
        or not bool(torch.isfinite(fit.variance).all())
        or bool(torch.any(fit.variance <= 0.0))
    ):
        raise ProtocolError("offline AMP fit normalizer/model dimensions are invalid")
    return {
        "artifact_schema": OFFLINE_AMP_ARTIFACT_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "seed": int(fit.seed),
        "input_dim": input_dim,
        "hidden_dims": list(fit.model.hidden_dims),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in fit.model.state_dict().items()
        },
        "normalizer_mean": fit.mean.detach().cpu(),
        "normalizer_variance": fit.variance.detach().cpu(),
        "normalizer_clip": float(fit.clip),
        "reward_scale": float(fit.reward_scale),
        "reward_epsilon": float(fit.reward_epsilon),
        "history": [dict(record) for record in fit.history],
        "protocol": dict(protocol),
        "protocol_sha256": canonical_sha256(dict(protocol)),
        "feature_contract": dict(feature_contract),
        "feature_contract_sha256": canonical_sha256(dict(feature_contract)),
        "training_provenance": dict(training_provenance),
    }


def save_offline_amp_fit(
    path: str | Path,
    fit: OfflineAMPFit,
    *,
    protocol: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    training_provenance: Mapping[str, Any],
) -> Path:
    """Persist a fit with exclusive creation and all replay-critical metadata."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _fit_payload(
        fit,
        protocol=protocol,
        feature_contract=feature_contract,
        training_provenance=training_provenance,
    )
    with target.open("xb") as handle:
        torch.save(payload, handle)
    return target


def load_offline_amp_fit(
    path: str | Path,
    *,
    device: str = "cpu",
    expected_protocol_sha256: str | None = None,
    expected_feature_contract_sha256: str | None = None,
) -> tuple[OfflineAMPFit, dict[str, Any]]:
    """Load and validate a reusable frozen offline critic artifact."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = torch.load(source, map_location=device, weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProtocolError(f"cannot load offline AMP fit {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("offline AMP fit payload is not a mapping")
    if payload.get("artifact_schema") != OFFLINE_AMP_ARTIFACT_SCHEMA:
        raise ProtocolError("offline AMP fit artifact schema mismatch")
    if payload.get("source_commit") != SOURCE_COMMIT:
        raise ProtocolError("offline AMP fit source commit mismatch")
    protocol = payload.get("protocol")
    feature_contract = payload.get("feature_contract")
    if not isinstance(protocol, Mapping) or not isinstance(feature_contract, Mapping):
        raise ProtocolError("offline AMP fit lacks protocol/feature contract")
    protocol_hash = canonical_sha256(dict(protocol))
    feature_hash = canonical_sha256(dict(feature_contract))
    if payload.get("protocol_sha256") != protocol_hash:
        raise ProtocolError("offline AMP fit protocol hash mismatch")
    if payload.get("feature_contract_sha256") != feature_hash:
        raise ProtocolError("offline AMP fit feature-contract hash mismatch")
    if expected_protocol_sha256 is not None and protocol_hash != expected_protocol_sha256:
        raise ProtocolError("offline AMP fit does not match the requested protocol")
    if (
        expected_feature_contract_sha256 is not None
        and feature_hash != expected_feature_contract_sha256
    ):
        raise ProtocolError("offline AMP fit does not match the requested feature contract")
    try:
        input_dim = int(payload["input_dim"])
        hidden_dims = tuple(int(value) for value in payload["hidden_dims"])
        model = OfflineStyleDiscriminator(input_dim, hidden_dims).to(torch.device(device))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        mean = torch.as_tensor(payload["normalizer_mean"], dtype=torch.float32, device=device)
        variance = torch.as_tensor(
            payload["normalizer_variance"], dtype=torch.float32, device=device
        )
        history = [dict(record) for record in payload["history"]]
        fit = OfflineAMPFit(
            model=model,
            mean=mean,
            variance=variance,
            clip=float(payload["normalizer_clip"]),
            reward_scale=float(payload["reward_scale"]),
            reward_epsilon=float(payload["reward_epsilon"]),
            seed=int(payload["seed"]),
            history=history,
            source_commit=str(payload["source_commit"]),
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ProtocolError(f"offline AMP fit payload is incomplete: {exc}") from exc
    # Re-run all model/normalizer invariants before returning the object.
    _fit_payload(
        fit,
        protocol=protocol,
        feature_contract=feature_contract,
        training_provenance=payload.get("training_provenance", {}),
    )
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {
            "model_state_dict", "normalizer_mean", "normalizer_variance", "history"
        }
    }
    metadata["history"] = history
    return fit, metadata


def _matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N,D]")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return array


def _gradient_norm_sq(logits: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(
        logits,
        observations,
        grad_outputs=torch.ones_like(logits),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return gradient.reshape(gradient.shape[0], -1).square().sum(dim=-1)


def train_offline_amp_critic(
    negative_train: np.ndarray,
    positive_train: np.ndarray,
    *,
    seed: int,
    protocol: Mapping[str, Any],
    device: str = "cpu",
) -> OfflineAMPFit:
    """Train the final epoch critic; validation never selects a checkpoint."""

    negative = _matrix("negative_train", negative_train)
    positive = _matrix("positive_train", positive_train)
    if negative.shape[1] != positive.shape[1]:
        raise ValueError("positive/negative feature dimensions differ")
    if str(protocol.get("source_commit")) != SOURCE_COMMIT:
        raise ValueError("offline AMP source commit is not the frozen baseline commit")
    if str(protocol.get("optimizer")) != "sgd_momentum":
        raise ValueError("offline AMP optimizer must be frozen as sgd_momentum")
    if str(protocol.get("checkpoint_selection")) != "final_epoch_no_validation_selection":
        raise ValueError("offline AMP must use the final epoch without validation selection")

    maximum = int(protocol["balanced_train_windows_per_domain_max"])
    count = min(maximum, negative.shape[0], positive.shape[0])
    batch_size = min(int(protocol["batch_size_per_class"]), count)
    epochs = int(protocol["epochs"])
    if count < 2 or batch_size < 1 or epochs < 1:
        raise ValueError("offline AMP training budget is invalid")
    rng = np.random.default_rng(int(seed))
    negative = negative[rng.choice(negative.shape[0], count, replace=False)]
    positive = positive[rng.choice(positive.shape[0], count, replace=False)]

    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    target = torch.device(device)
    raw_negative = torch.as_tensor(negative, dtype=torch.float32, device=target)
    raw_positive = torch.as_tensor(positive, dtype=torch.float32, device=target)
    moments = torch.cat((raw_negative, raw_positive), dim=0)
    mean = moments.mean(dim=0)
    variance = torch.clamp(moments.var(dim=0, unbiased=False), min=1.0e-8)
    clip = float(protocol["normalizer_clip"])

    def normalize(value: torch.Tensor) -> torch.Tensor:
        return torch.clamp((value - mean) / torch.sqrt(variance), -clip, clip)

    negative_normalized = normalize(raw_negative)
    positive_normalized = normalize(raw_positive)
    model = OfflineStyleDiscriminator(
        negative.shape[1], tuple(int(value) for value in protocol["hidden_dims"])
    ).to(target)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        momentum=float(protocol["momentum"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    gp_weight = float(protocol["gradient_penalty"])
    logit_weight = float(protocol["logit_regularization"])
    history: list[dict[str, float]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for epoch in range(epochs):
        order_negative = torch.randperm(count, generator=generator)
        order_positive = torch.randperm(count, generator=generator)
        totals = {"loss": 0.0, "bce": 0.0, "gp": 0.0, "batches": 0.0}
        model.train()
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            negative_batch = negative_normalized[order_negative[start:stop].to(target)]
            positive_batch = positive_normalized[order_positive[start:stop].to(target)]
            # Current and replay are the same frozen negative population in
            # this offline substitution probe.  Duplication preserves the
            # original 0.5 fake / 0.5 expert class weighting exactly.
            expert = positive_batch.detach().requires_grad_(gp_weight > 0.0)
            fake = torch.cat((negative_batch, negative_batch), dim=0).detach().requires_grad_(gp_weight > 0.0)
            expert_logits = model(expert)
            fake_logits = model(fake)
            expert_bce = F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
            fake_bce = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            bce = 0.5 * (expert_bce + fake_bce)
            if gp_weight:
                gp = 0.5 * (
                    _gradient_norm_sq(expert_logits, expert).mean()
                    + _gradient_norm_sq(fake_logits, fake).mean()
                )
            else:
                gp = bce.new_zeros(())
            logit_reg = model.logit.weight.reshape(-1).square().sum()
            loss = bce + gp_weight * gp + logit_weight * logit_reg
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("offline AMP critic loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["bce"] += float(bce.detach())
            totals["gp"] += float(gp.detach())
            totals["batches"] += 1.0
        history.append(
            {
                "epoch": float(epoch + 1),
                **{
                    key: value / totals["batches"]
                    for key, value in totals.items()
                    if key != "batches"
                },
            }
        )
    return OfflineAMPFit(
        model=model,
        mean=mean.detach(),
        variance=variance.detach(),
        clip=clip,
        reward_scale=float(protocol["reward_scale"]),
        reward_epsilon=float(protocol["reward_epsilon"]),
        seed=int(seed),
        history=history,
    )


def held_out_amp_metrics(
    fit: OfflineAMPFit,
    negative_test: np.ndarray,
    positive_test: np.ndarray,
) -> dict[str, Any]:
    negative = _matrix("negative_test", negative_test)
    positive = _matrix("positive_test", positive_test)
    logits = np.concatenate((fit.logits(negative), fit.logits(positive)))
    labels = np.concatenate((np.zeros(negative.shape[0]), np.ones(positive.shape[0])))
    return {
        "auc": float(roc_auc_score(labels, logits)),
        "negative_reward_mean": float(np.mean(fit.rewards(negative))),
        "positive_reward_mean": float(np.mean(fit.rewards(positive))),
        "negative_count": int(negative.shape[0]),
        "positive_count": int(positive.shape[0]),
        "final_epoch": fit.history[-1],
    }


__all__ = [
    "OFFLINE_AMP_ARTIFACT_SCHEMA", "OfflineAMPFit", "OfflineStyleDiscriminator",
    "SOURCE_COMMIT", "directed_offline_amp_protocol", "held_out_amp_metrics",
    "load_offline_amp_fit",
    "save_offline_amp_fit", "train_offline_amp_critic",
]
