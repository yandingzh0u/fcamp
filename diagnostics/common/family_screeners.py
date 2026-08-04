"""Pure, fail-closed analysis utilities for discovery diagnostics 60--62.

The three screeners in this module are deliberately *not* policy objectives.
They consume frozen rollout evidence and train small offline probes only.  In
particular, no helper imports or accepts the retired A_mix/FCAMP/H4 domain.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from .domain_data import (
    AMP_FRAME_DIM,
    AMP_WINDOW_STEPS,
    DomainBundle,
    amp_domain_feature_contract,
    chronological_amp_windows,
)
from .manifest import DependencyUnavailable, ProtocolError, canonical_sha256, read_json


AMP_WINDOW_DIM = AMP_FRAME_DIM * AMP_WINDOW_STEPS
EXECUTABILITY_BANK_SCHEMA = "largebox_policy_relative_executability_v1"
_ENDPOINT = re.compile(r":end=(\d+)$")


def _strings(values: np.ndarray | Sequence[Any]) -> np.ndarray:
    return np.asarray([str(value) for value in np.asarray(values).reshape(-1)])


def _forbid_retired_name(value: Any, *, context: str) -> None:
    text = str(value).lower()
    if "a_mix" in text or "fcamp" in text or "h4" in text or "flow-cps" in text:
        raise ProtocolError(f"retired A_mix/FCAMP/H4 entered {context}")


@dataclass(frozen=True, slots=True)
class PairedAMPWindows:
    reference: np.ndarray
    execution: np.ndarray
    split: np.ndarray
    snapshot_ids: np.ndarray
    trajectory_ids: np.ndarray
    phase: np.ndarray
    endpoint: np.ndarray
    contact_mode: np.ndarray
    failure: np.ndarray

    def validate(self) -> None:
        x = np.asarray(self.reference, dtype=np.float64)
        y = np.asarray(self.execution, dtype=np.float64)
        if x.shape != y.shape or x.ndim != 2 or x.shape[1] != AMP_WINDOW_DIM:
            raise ProtocolError("paired AMP windows must be aligned [N,2390]")
        if x.shape[0] < 6 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ProtocolError("paired AMP windows are empty or non-finite")
        for name, values in (
            ("split", self.split),
            ("snapshot_ids", self.snapshot_ids),
            ("trajectory_ids", self.trajectory_ids),
            ("phase", self.phase),
            ("endpoint", self.endpoint),
            ("contact_mode", self.contact_mode),
            ("failure", self.failure),
        ):
            if np.asarray(values).shape != (x.shape[0],):
                raise ProtocolError(f"paired field {name} is not aligned")
        if set(_strings(self.split)) != {"train", "validation", "test"}:
            raise ProtocolError("paired windows must cover the frozen three splits")
        keys = list(zip(_strings(self.snapshot_ids), self.endpoint.tolist()))
        if len(keys) != len(set(keys)):
            raise ProtocolError("paired snapshot/window endpoint keys are duplicated")


def _endpoint_vector(bundle: DomainBundle) -> np.ndarray:
    result: list[int] = []
    for sample_id in _strings(bundle.sample_ids):
        match = _ENDPOINT.search(str(sample_id))
        if match is None:
            raise ProtocolError(
                f"domain {bundle.name} sample ID lacks the exact window endpoint: {sample_id}"
            )
        result.append(int(match.group(1)))
    return np.asarray(result, dtype=np.int64)


def exact_paired_amp_windows(
    reference: DomainBundle,
    execution: DomainBundle,
) -> PairedAMPWindows:
    """Pair K and T_u500 only by the preregistered physical condition key.

    No nearest-neighbour, phase tolerance, truncation, or synthetic pairing is
    allowed.  A mismatched key set is a protocol error, not a smaller dataset.
    """

    if reference.name != "K" or execution.name != "T_u500":
        raise ProtocolError("diag_60 accepts only K<->T_u500")
    for bundle in (reference, execution):
        _forbid_retired_name(bundle.name, context="diag_60 paired domains")
        amp_domain_feature_contract(bundle)
    if reference.feature_schema_sha256 != execution.feature_schema_sha256:
        raise ProtocolError("K/T_u500 AMP coordinate schemas differ")
    ref_end = _endpoint_vector(reference)
    exe_end = _endpoint_vector(execution)

    def keys(bundle: DomainBundle, endpoints: np.ndarray) -> list[tuple[str, str, int]]:
        # float.hex is an exact continuous-phase identity, not a tolerance bin.
        return [
            (str(snapshot), float(phase).hex(), int(endpoint))
            for snapshot, phase, endpoint in zip(
                _strings(bundle.snapshot_ids), np.asarray(bundle.phase), endpoints
            )
        ]

    ref_keys = keys(reference, ref_end)
    exe_keys = keys(execution, exe_end)
    if len(ref_keys) != len(set(ref_keys)) or len(exe_keys) != len(set(exe_keys)):
        raise ProtocolError("K/T_u500 contains duplicate exact pair keys")
    if set(ref_keys) != set(exe_keys):
        missing_execution = sorted(set(ref_keys) - set(exe_keys))[:5]
        missing_reference = sorted(set(exe_keys) - set(ref_keys))[:5]
        raise ProtocolError(
            "K/T_u500 exact pair-key sets differ; approximate pairing is forbidden: "
            f"missing_execution={missing_execution}, missing_reference={missing_reference}"
        )
    exe_lookup = {key: index for index, key in enumerate(exe_keys)}
    order = np.asarray([exe_lookup[key] for key in ref_keys], dtype=np.int64)
    checks = (
        ("split", _strings(reference.split), _strings(execution.split)[order]),
        ("contact_mode", _strings(reference.contact_mode), _strings(execution.contact_mode)[order]),
        ("phase", np.asarray(reference.phase), np.asarray(execution.phase)[order]),
    )
    for name, left, right in checks:
        if not np.array_equal(left, right):
            raise ProtocolError(f"paired K/T_u500 {name} metadata differs")
    result = PairedAMPWindows(
        reference=np.asarray(reference.features, dtype=np.float32),
        execution=np.asarray(execution.features, dtype=np.float32)[order],
        split=_strings(reference.split),
        snapshot_ids=_strings(reference.snapshot_ids),
        trajectory_ids=_strings(execution.trajectory_ids)[order],
        phase=np.asarray(reference.phase, dtype=np.float64),
        endpoint=ref_end,
        contact_mode=_strings(reference.contact_mode),
        failure=np.asarray(execution.failure, dtype=bool)[order],
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class FrozenStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FrozenStandardizer":
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
            raise ProtocolError("normalizer training bank must be finite [N,D]")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1.0e-6, scale, 1.0)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.mean.size:
            raise ProtocolError("normalizer input shape differs from its fitted bank")
        result = (x - self.mean) / self.scale
        if not np.isfinite(result).all():
            raise ProtocolError("normalization produced NaN or Inf")
        return result

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.mean.tobytes() + self.scale.tobytes()).hexdigest()


def binary_auc(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(train_y, dtype=np.int64)
    target = np.asarray(test_y, dtype=np.int64)
    if len(np.unique(labels)) != 2 or len(np.unique(target)) != 2:
        raise DependencyUnavailable("binary AUC needs both classes in train and test")
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=20260803, solver="lbfgs"
    )
    model.fit(np.asarray(train_x, dtype=np.float64), labels)
    return float(roc_auc_score(target, model.predict_proba(test_x)[:, 1]))


def cross_domain_retrieval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    snapshot_ids: np.ndarray,
    phase: np.ndarray,
    phase_bins: int,
) -> dict[str, float]:
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] < 2:
        raise ProtocolError("retrieval latents must be paired [N,D]")
    labels_snapshot = _strings(snapshot_ids)
    labels_phase = np.minimum(
        int(phase_bins) - 1,
        np.floor(np.asarray(phase, dtype=np.float64) * int(phase_bins)).astype(np.int64),
    )

    def direction(query: np.ndarray, bank: np.ndarray) -> float:
        nearest = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
        nearest.fit(bank)
        index = nearest.kneighbors(query, return_distance=False).reshape(-1)
        return float(
            np.mean(
                (labels_snapshot[index] == labels_snapshot)
                & (labels_phase[index] == labels_phase)
            )
        )

    forward = direction(x, y)
    reverse = direction(y, x)
    return {"K_to_T_u500": forward, "T_u500_to_K": reverse, "mean": (forward + reverse) / 2.0}


def _probe_contact(
    train_x: np.ndarray,
    train_label: np.ndarray,
    test_x: np.ndarray,
    test_label: np.ndarray,
) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    labels = _strings(train_label)
    target = _strings(test_label)
    if len(set(labels)) < 2 or not set(target) <= set(labels):
        raise DependencyUnavailable("contact probe lacks train/test mode coverage")
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=20260803, solver="lbfgs"
    )
    model.fit(train_x, labels)
    return float(balanced_accuracy_score(target, model.predict(test_x)))


def _future_pairs(data: PairedAMPWindows, split: str) -> tuple[np.ndarray, np.ndarray]:
    selected = np.where(_strings(data.split) == split)[0]
    lookup = {
        (str(data.trajectory_ids[index]), int(data.endpoint[index])): int(index)
        for index in selected
    }
    current: list[int] = []
    future: list[int] = []
    for index in selected:
        successor = lookup.get((str(data.trajectory_ids[index]), int(data.endpoint[index]) + 1))
        if successor is not None:
            current.append(int(index))
            future.append(successor)
    if len(current) < 32:
        raise DependencyUnavailable(f"{split} has too few within-trajectory future pairs")
    return np.asarray(current, dtype=np.int64), np.asarray(future, dtype=np.int64)


def representation_preservation(
    data: PairedAMPWindows,
    latent_execution: Mapping[str, np.ndarray],
    raw_execution: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Measure contact and next-frame retention against the raw-window probe."""

    train = _strings(data.split) == "train"
    test = _strings(data.split) == "test"
    contact_latent = _probe_contact(
        latent_execution["train"], data.contact_mode[train],
        latent_execution["test"], data.contact_mode[test],
    )
    contact_raw = _probe_contact(
        raw_execution["train"], data.contact_mode[train],
        raw_execution["test"], data.contact_mode[test],
    )

    from sklearn.linear_model import Ridge

    train_current, train_future = _future_pairs(data, "train")
    test_current, test_future = _future_pairs(data, "test")
    target_train = np.asarray(data.execution[train_future, -AMP_FRAME_DIM:], dtype=np.float64)
    target_test = np.asarray(data.execution[test_future, -AMP_FRAME_DIM:], dtype=np.float64)
    target_norm = FrozenStandardizer.fit(target_train)
    target_train = target_norm.transform(target_train)
    target_test = target_norm.transform(target_test)

    split_positions = {
        split: {int(global_index): local for local, global_index in enumerate(np.where(_strings(data.split) == split)[0])}
        for split in ("train", "test")
    }

    def local_indices(global_indices: np.ndarray, split: str) -> np.ndarray:
        return np.asarray([split_positions[split][int(index)] for index in global_indices], dtype=np.int64)

    tr = local_indices(train_current, "train")
    te = local_indices(test_current, "test")

    def utility(train_features: np.ndarray, test_features: np.ndarray) -> float:
        model = Ridge(alpha=1.0)
        model.fit(train_features[tr], target_train)
        prediction = model.predict(test_features[te])
        mse = float(np.mean(np.square(prediction - target_test)))
        return float(math.exp(-mse))

    future_latent = utility(latent_execution["train"], latent_execution["test"])
    future_raw = utility(raw_execution["train"], raw_execution["test"])
    contact_ratio = contact_latent / max(contact_raw, 1.0e-12)
    future_ratio = future_latent / max(future_raw, 1.0e-12)
    return {
        "contact": {
            "latent_balanced_accuracy": contact_latent,
            "raw_balanced_accuracy": contact_raw,
            "relative_to_raw": contact_ratio,
        },
        "future": {
            "target": "next_T_u500_raw_239D_frame",
            "metric": "exp(-mean_standardized_squared_prediction_error)",
            "latent_utility": future_latent,
            "raw_utility": future_raw,
            "relative_to_raw": future_ratio,
        },
        "joint_min_relative_to_raw": min(contact_ratio, future_ratio),
    }


class PairedContrastiveEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], latent_dim: int) -> None:
        super().__init__()
        if tuple(int(value) for value in hidden) != (256, 256):
            raise ProtocolError("diag_60 contrastive hidden dimensions changed")

        def encoder() -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, int(hidden[0])),
                torch.nn.ReLU(),
                torch.nn.Linear(int(hidden[0]), int(hidden[1])),
                torch.nn.ReLU(),
                torch.nn.Linear(int(hidden[1]), int(latent_dim)),
            )

        self.reference_encoder = encoder()
        self.execution_encoder = encoder()

    def encode_reference(self, values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.reference_encoder(values), dim=-1)

    def encode_execution(self, values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.execution_encoder(values), dim=-1)


def _torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encode_batches(model: PairedContrastiveEncoder, values: np.ndarray, *, reference: bool) -> np.ndarray:
    device = next(model.parameters()).device
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 2048):
            tensor = torch.as_tensor(values[start : start + 2048], dtype=torch.float32, device=device)
            latent = model.encode_reference(tensor) if reference else model.encode_execution(tensor)
            output.append(latent.cpu().numpy())
    return np.concatenate(output)


def train_paired_contrastive(
    normalized_reference: Mapping[str, np.ndarray],
    normalized_execution: Mapping[str, np.ndarray],
    *,
    hidden_dims: Sequence[int],
    latent_dim: int,
    temperature: float,
    batch_size: int,
    epochs_max: int,
    patience: int,
    seed: int,
    validation_metadata: Mapping[str, np.ndarray],
    phase_bins: int,
) -> tuple[PairedContrastiveEncoder, dict[str, Any]]:
    if normalized_reference["train"].shape[0] < batch_size // 4:
        raise DependencyUnavailable("paired contrastive training bank is too small")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = _torch_device()
    model = PairedContrastiveEncoder(
        normalized_reference["train"].shape[1], hidden_dims, latent_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    rng = np.random.default_rng(int(seed))
    best_retrieval = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    count = normalized_reference["train"].shape[0]
    for epoch in range(1, int(epochs_max) + 1):
        model.train()
        order = rng.permutation(count)
        losses: list[float] = []
        for start in range(0, count, int(batch_size)):
            indices = order[start : start + int(batch_size)]
            if indices.size < 2:
                continue
            x = torch.as_tensor(normalized_reference["train"][indices], dtype=torch.float32, device=device)
            y = torch.as_tensor(normalized_execution["train"][indices], dtype=torch.float32, device=device)
            zx = model.encode_reference(x)
            zy = model.encode_execution(y)
            logits = zx @ zy.T / float(temperature)
            labels = torch.arange(indices.size, device=device)
            loss = 0.5 * (
                torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        ref_val = _encode_batches(model, normalized_reference["validation"], reference=True)
        exe_val = _encode_batches(model, normalized_execution["validation"], reference=False)
        retrieval = cross_domain_retrieval(
            ref_val,
            exe_val,
            snapshot_ids=validation_metadata["snapshot_ids"],
            phase=validation_metadata["phase"],
            phase_bins=phase_bins,
        )["mean"]
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_retrieval": retrieval})
        if retrieval > best_retrieval + 1.0e-12:
            best_retrieval = retrieval
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise ProtocolError("paired contrastive training produced no selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_retrieval": float(best_retrieval),
        "epochs_ran": len(history),
        "selection": "validation_cross_domain_retrieval_test_untouched",
        "history": history,
    }


def latent_distance_stability(latents: Sequence[np.ndarray], *, maximum_samples: int = 512) -> dict[str, float]:
    if len(latents) != 3:
        raise ProtocolError("diag_60 stability requires exactly three seeds")
    count = min(int(maximum_samples), *(array.shape[0] for array in latents))
    if count < 8:
        raise DependencyUnavailable("too few aligned test latents for seed stability")
    indices = np.linspace(0, latents[0].shape[0] - 1, count).round().astype(np.int64)
    vectors: list[np.ndarray] = []
    upper = np.triu_indices(count, k=1)
    for latent in latents:
        values = np.asarray(latent, dtype=np.float64)[indices]
        distance = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
        vectors.append(distance[upper])
    correlations = []
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            correlations.append(float(spearmanr(vectors[left], vectors[right]).statistic))
    if not np.isfinite(correlations).all():
        raise ProtocolError("latent seed stability is non-finite")
    return {"pairwise_spearman_mean": float(np.mean(correlations)), "pairwise_spearman_min": float(np.min(correlations))}


class DenoisingMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        if tuple(int(value) for value in hidden_dims) != (512, 256):
            raise ProtocolError("diag_61 denoising hidden dimensions changed")
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(hidden_dims[0])),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dims[0]), int(hidden_dims[1])),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dims[1]), input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


@dataclass(slots=True)
class FrozenDenoisingPrior:
    model: DenoisingMLP
    normalizer: FrozenStandardizer
    noise_scales: tuple[float, ...]
    seed: int

    def score(self, values: np.ndarray, *, batch_size: int = 2048) -> np.ndarray:
        normalized = self.normalizer.transform(values)
        device = next(self.model.parameters()).device
        result: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(normalized), int(batch_size)):
                clean = torch.as_tensor(normalized[start : start + batch_size], dtype=torch.float32, device=device)
                errors = []
                for scale in self.noise_scales:
                    # Evaluation is deterministic and does not add a hidden
                    # sample-level source identifier: one analytic offset is
                    # used for every domain.
                    noisy = clean + float(scale) * torch.sin(clean * 1.61803398875)
                    prediction = self.model(noisy)
                    errors.append(torch.mean(torch.square(prediction - clean), dim=-1))
                result.append((-torch.stack(errors).mean(dim=0)).cpu().numpy())
        values_out = np.concatenate(result).astype(np.float64)
        if not np.isfinite(values_out).all():
            raise ProtocolError("denoising prior score is non-finite")
        return values_out

    def input_gradient_norm(self, values: np.ndarray, *, maximum_samples: int = 256) -> float:
        normalized = self.normalizer.transform(values[:maximum_samples])
        device = next(self.model.parameters()).device
        clean = torch.as_tensor(normalized, dtype=torch.float32, device=device).requires_grad_(True)
        errors = []
        for scale in self.noise_scales:
            noisy = clean + float(scale) * torch.sin(clean * 1.61803398875)
            prediction = self.model(noisy)
            errors.append(torch.mean(torch.square(prediction - clean), dim=-1))
        score = -torch.stack(errors).mean()
        gradient = torch.autograd.grad(score, clean, retain_graph=False)[0]
        return float(torch.linalg.vector_norm(gradient, dim=-1).mean().detach().cpu())


def train_denoising_prior(
    train_values: np.ndarray,
    validation_values: np.ndarray,
    *,
    hidden_dims: Sequence[int],
    noise_scales: Sequence[float],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[FrozenDenoisingPrior, dict[str, Any]]:
    if tuple(float(value) for value in noise_scales) != (0.01, 0.05, 0.10):
        raise ProtocolError("diag_61 noise scales changed")
    normalizer = FrozenStandardizer.fit(train_values)
    train = normalizer.transform(train_values)
    validation = normalizer.transform(validation_values)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = _torch_device()
    model = DenoisingMLP(train.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, float]] = []
    scales = torch.as_tensor(noise_scales, dtype=torch.float32, device=device)
    for epoch in range(1, int(epochs) + 1):
        order = rng.permutation(len(train))
        losses: list[float] = []
        model.train()
        for start in range(0, len(train), int(batch_size)):
            index = order[start : start + int(batch_size)]
            clean = torch.as_tensor(train[index], dtype=torch.float32, device=device)
            scale_index = torch.randint(0, len(scales), (len(index), 1), generator=generator, device=device)
            scale = scales[scale_index]
            noise = torch.randn(clean.shape, generator=generator, device=device)
            noisy = clean + scale * noise
            prediction = model(noisy)
            loss = torch.mean(torch.square(prediction - clean))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            clean = torch.as_tensor(validation[: min(len(validation), 4096)], dtype=torch.float32, device=device)
            validation_errors = []
            for scale in scales:
                noisy = clean + scale * torch.sin(clean * 1.61803398875)
                validation_errors.append(torch.mean(torch.square(model(noisy) - clean)))
            validation_loss = float(torch.stack(validation_errors).mean().cpu())
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return FrozenDenoisingPrior(model, normalizer, tuple(float(v) for v in noise_scales), int(seed)), {
        "seed": int(seed),
        "epochs": int(epochs),
        "final_train_loss": history[-1]["train_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "normalizer_sha256": normalizer.sha256,
        "normalizer_source": "successful_teacher_train_split_only_then_frozen",
    }


@dataclass(frozen=True, slots=True)
class RobotTeacherArchive:
    features: np.ndarray
    split: np.ndarray
    checkpoint_update: np.ndarray
    successful: np.ndarray
    snapshot_ids: np.ndarray
    trajectory_ids: np.ndarray
    selected_successful_updates: tuple[int, ...]
    clean_completion_by_update: Mapping[int, float]
    source_shards: tuple[str, ...]

    def validate(self) -> None:
        values = np.asarray(self.features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != AMP_WINDOW_DIM or values.shape[0] < 32:
            raise ProtocolError("robot teacher archive must contain exact [N,2390] windows")
        if not np.isfinite(values).all():
            raise ProtocolError("robot teacher archive contains NaN or Inf")
        for field in (
            self.split, self.checkpoint_update, self.successful,
            self.snapshot_ids, self.trajectory_ids,
        ):
            if np.asarray(field).shape != (values.shape[0],):
                raise ProtocolError("robot teacher archive metadata is not window aligned")
        if set(_strings(self.split)) != {"train", "validation", "test"}:
            raise ProtocolError("robot teacher archive lacks a frozen split")
        if not self.selected_successful_updates:
            raise DependencyUnavailable("no teacher checkpoint passed the frozen clean completion gate")
        if any(float(self.clean_completion_by_update[update]) < 0.90 for update in self.selected_successful_updates):
            raise ProtocolError("teacher archive contains a checkpoint below clean completion 0.90")
        expected_success = np.isin(
            np.asarray(self.checkpoint_update, dtype=np.int64),
            np.asarray(self.selected_successful_updates, dtype=np.int64),
        )
        if not np.array_equal(np.asarray(self.successful, dtype=bool), expected_success):
            raise ProtocolError("teacher archive successful mask differs from clean selection")
        if not expected_success.any():
            raise DependencyUnavailable("teacher archive has no successful execution windows")


def build_robot_teacher_archive(
    index_path: str | Path,
    split_audit_path: str | Path,
    *,
    clean_completion_threshold: float = 0.90,
) -> RobotTeacherArchive:
    """Build an agent-PhysX-only archive selected by measured clean completion."""

    from .canonical_collection import (
        load_rollout_index,
        load_rollout_trajectory,
        resolve_shard_path,
    )

    if float(clean_completion_threshold) != 0.90:
        raise ProtocolError("successful teacher selection threshold changed from 0.90")
    source = Path(index_path).expanduser().resolve()
    rows = [
        row for row in load_rollout_index(source)
        if str(row.get("policy_domain")) == "teacher_fixed_reward"
        and float(row.get("common_sigma", 0.0)) == 0.0
    ]
    if not rows:
        raise DependencyUnavailable("canonical index has no fixed-reward teacher rows")
    if any("A_mix" in str(row) or "FCAMP" in str(row) for row in rows):
        raise ProtocolError("retired A_mix/FCAMP metadata entered the teacher archive")
    cache: dict[str, dict[str, Any]] = {}
    clean_outcomes: dict[int, list[bool]] = {}
    for row in rows:
        if str(row["collector_mode"]) != "clean_mean":
            continue
        tree = load_rollout_trajectory(source, row, cache=cache)
        done = np.asarray(tree["trajectory"]["done"].detach().cpu(), dtype=bool).reshape(-1)
        complete = np.asarray(
            tree["trajectory"]["motion_complete"].detach().cpu(), dtype=bool
        ).reshape(-1)
        terminal = int(np.where(done)[0][0]) if done.any() else len(done) - 1
        clean_outcomes.setdefault(int(row["checkpoint_update"]), []).append(bool(complete[terminal]))
    completion = {
        update: float(np.mean(values)) for update, values in sorted(clean_outcomes.items())
    }
    successful = tuple(
        update for update, value in completion.items()
        if value >= float(clean_completion_threshold)
    )
    if not successful:
        raise DependencyUnavailable("no checkpoint reaches clean completion >=0.90")

    split_payload = read_json(split_audit_path)
    if split_payload.get("status") != "PASS":
        raise DependencyUnavailable("diag_14 split audit is not PASS")
    sample_to_split = (
        split_payload.get("evidence", {}).get("inner_snapshot_split", {}).get("sample_to_split")
    )
    if not isinstance(sample_to_split, Mapping):
        raise ProtocolError("diag_14 lacks sample_to_split")

    feature_parts: list[np.ndarray] = []
    splits: list[np.ndarray] = []
    updates: list[np.ndarray] = []
    successful_flags: list[np.ndarray] = []
    snapshots: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    selected_rows = [
        row for row in rows
        if str(row["collector_mode"]) == "controlled_environment"
    ]
    if not selected_rows:
        raise DependencyUnavailable("successful checkpoints have no controlled-environment executions")
    for row in selected_rows:
        sample = str(row["sample_id"])
        if sample not in sample_to_split:
            raise ProtocolError(f"successful teacher sample lacks frozen split: {sample}")
        tree = load_rollout_trajectory(source, row, cache=cache)
        frame = np.asarray(
            tree["imitation"]["agent_physx_raw_frame"].detach().cpu(), dtype=np.float32
        )
        done = np.asarray(tree["trajectory"]["done"].detach().cpu(), dtype=bool)
        try:
            windows, _ = chronological_amp_windows(frame, done=done)
        except DependencyUnavailable:
            continue
        count = windows.shape[0]
        feature_parts.append(windows)
        splits.append(np.full(count, str(sample_to_split[sample])))
        updates.append(np.full(count, int(row["checkpoint_update"]), dtype=np.int64))
        successful_flags.append(
            np.full(count, int(row["checkpoint_update"]) in successful, dtype=bool)
        )
        snapshots.append(np.full(count, str(row["snapshot_id"])))
        trajectories.append(np.full(count, str(row["trajectory_id"])))
    if not feature_parts:
        raise DependencyUnavailable("successful teacher archive has no complete alive AMP windows")
    shard_paths = tuple(
        sorted({str(resolve_shard_path(source, row)) for row in selected_rows})
    )
    archive = RobotTeacherArchive(
        features=np.concatenate(feature_parts).astype(np.float32),
        split=np.concatenate(splits),
        checkpoint_update=np.concatenate(updates),
        successful=np.concatenate(successful_flags),
        snapshot_ids=np.concatenate(snapshots),
        trajectory_ids=np.concatenate(trajectories),
        selected_successful_updates=successful,
        clean_completion_by_update=completion,
        source_shards=shard_paths,
    )
    archive.validate()
    return archive


def deterministic_group_split(groups: Sequence[Any], *, seed: int) -> np.ndarray:
    """Hash complete groups into train/validation/test without row leakage."""

    unique = sorted(set(str(value) for value in groups))
    if len(unique) < 7:
        raise DependencyUnavailable("held-out group axis has fewer than seven groups")
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{int(seed)}:{value}".encode()).hexdigest(),
    )
    n = len(ranked)
    train_end = max(1, int(math.floor(0.70 * n)))
    validation_end = max(train_end + 1, int(math.floor(0.85 * n)))
    validation_end = min(validation_end, n - 1)
    assignment = {
        value: ("train" if index < train_end else "validation" if index < validation_end else "test")
        for index, value in enumerate(ranked)
    }
    result = np.asarray([assignment[str(value)] for value in groups])
    for split in ("train", "validation", "test"):
        if not np.any(result == split):
            raise DependencyUnavailable(f"group split has no {split} rows")
    return result


@dataclass(frozen=True, slots=True)
class ExecutabilityDataset:
    features: np.ndarray
    success: np.ndarray
    segment_complete: np.ndarray
    failure: np.ndarray
    tracking_loss: np.ndarray
    joint_limit_event: np.ndarray
    undesired_contact_event: np.ndarray
    snapshot_ids: np.ndarray
    segment_ids: np.ndarray
    phase_bins: np.ndarray
    phase: np.ndarray
    requested_phase_offset: np.ndarray
    time_scale: np.ndarray
    metadata: Mapping[str, Any]

    def validate(self, *, expected_snapshots: int | None = None, expected_segments: int | None = None) -> None:
        x = np.asarray(self.features, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 20 or x.shape[1] < 2 or not np.isfinite(x).all():
            raise ProtocolError("executability features must be finite [N,D]")
        count = x.shape[0]
        aligned = (
            self.success, self.segment_complete, self.failure, self.tracking_loss,
            self.joint_limit_event, self.undesired_contact_event, self.snapshot_ids,
            self.segment_ids, self.phase_bins, self.phase, self.requested_phase_offset,
            self.time_scale,
        )
        if any(np.asarray(value).shape != (count,) for value in aligned):
            raise ProtocolError("executability rows are not aligned")
        booleans = (
            self.success, self.segment_complete, self.failure, self.tracking_loss,
            self.joint_limit_event, self.undesired_contact_event,
        )
        if any(np.asarray(value).dtype != np.bool_ for value in booleans):
            raise ProtocolError("executability outcomes must be boolean PhysX events")
        expected_success = (
            np.asarray(self.segment_complete)
            & ~np.asarray(self.failure)
            & ~np.asarray(self.joint_limit_event)
            & ~np.asarray(self.undesired_contact_event)
        )
        if not np.array_equal(np.asarray(self.success), expected_success):
            raise ProtocolError("success labels differ from the frozen event conjunction")
        snapshots = set(_strings(self.snapshot_ids))
        segments = set(_strings(self.segment_ids))
        if expected_snapshots is not None and len(snapshots) != int(expected_snapshots):
            raise ProtocolError("executability snapshot count differs from the frozen protocol")
        if expected_segments is not None and len(segments) != int(expected_segments):
            raise ProtocolError("executability segment count differs from the frozen protocol")
        keys = list(zip(_strings(self.snapshot_ids), _strings(self.segment_ids)))
        if len(keys) != len(set(keys)) or len(keys) != len(snapshots) * len(segments):
            raise ProtocolError("executability bank is not the exact snapshot x segment product")
        required = {
            "schema": EXECUTABILITY_BANK_SCHEMA,
            "real_physx_outcomes": True,
            "same_snapshot_replay_verified": True,
            "shared_environment_randomness": True,
            "label_source": "closed_loop_PhysX_outcomes_only",
            "reference_rmse_used_as_label": False,
            "raw_human_positive": False,
            "A_mix": "legacy_quarantined",
        }
        mismatches = {key: {"expected": value, "actual": self.metadata.get(key)} for key, value in required.items() if self.metadata.get(key) != value}
        if mismatches:
            raise ProtocolError(f"executability evidence guarantees differ: {mismatches}")
        _forbid_retired_name(self.metadata.get("policy_method", ""), context="diag_62 policy")


def save_executability_dataset(path: str | Path, dataset: ExecutabilityDataset) -> Path:
    dataset.validate()
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        np.savez_compressed(
            handle,
            features=np.asarray(dataset.features, dtype=np.float32),
            success=np.asarray(dataset.success, dtype=bool),
            segment_complete=np.asarray(dataset.segment_complete, dtype=bool),
            failure=np.asarray(dataset.failure, dtype=bool),
            tracking_loss=np.asarray(dataset.tracking_loss, dtype=bool),
            joint_limit_event=np.asarray(dataset.joint_limit_event, dtype=bool),
            undesired_contact_event=np.asarray(dataset.undesired_contact_event, dtype=bool),
            snapshot_ids=_strings(dataset.snapshot_ids),
            segment_ids=_strings(dataset.segment_ids),
            phase_bins=np.asarray(dataset.phase_bins, dtype=np.int64),
            phase=np.asarray(dataset.phase, dtype=np.float64),
            requested_phase_offset=np.asarray(dataset.requested_phase_offset, dtype=np.int64),
            time_scale=np.asarray(dataset.time_scale, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(dict(dataset.metadata), sort_keys=True, allow_nan=False)),
        )
    return target


def load_executability_dataset(path: str | Path) -> ExecutabilityDataset:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"real executability bank is absent: {source}")
    try:
        with np.load(source, allow_pickle=False) as data:
            forbidden = [name for name in data.files if any(token in name.lower() for token in ("rmse", "quality", "oracle", "score_label"))]
            if forbidden:
                raise ProtocolError(f"executability bank contains forbidden surrogate labels: {forbidden}")
            metadata = json.loads(str(data["metadata_json"].item()))
            dataset = ExecutabilityDataset(
                features=np.asarray(data["features"], dtype=np.float32),
                success=np.asarray(data["success"], dtype=bool),
                segment_complete=np.asarray(data["segment_complete"], dtype=bool),
                failure=np.asarray(data["failure"], dtype=bool),
                tracking_loss=np.asarray(data["tracking_loss"], dtype=bool),
                joint_limit_event=np.asarray(data["joint_limit_event"], dtype=bool),
                undesired_contact_event=np.asarray(data["undesired_contact_event"], dtype=bool),
                snapshot_ids=np.asarray(data["snapshot_ids"]),
                segment_ids=np.asarray(data["segment_ids"]),
                phase_bins=np.asarray(data["phase_bins"], dtype=np.int64),
                phase=np.asarray(data["phase"], dtype=np.float64),
                requested_phase_offset=np.asarray(data["requested_phase_offset"], dtype=np.int64),
                time_scale=np.asarray(data["time_scale"], dtype=np.float64),
                metadata=metadata,
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError(f"cannot load executability bank {source}: {exc}") from exc
    dataset.validate()
    return dataset


class BinaryMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        if tuple(int(value) for value in hidden_dims) != (256, 256):
            raise ProtocolError("diag_62 classifier hidden dimensions changed")
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def expected_calibration_error(probability: np.ndarray, label: np.ndarray, *, bins: int = 15) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    if probability.shape != label.shape or not np.isfinite(probability).all():
        raise ProtocolError("ECE inputs are not aligned and finite")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    result = 0.0
    for index in range(int(bins)):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if selected.any():
            result += float(selected.mean()) * abs(float(probability[selected].mean()) - float(label[selected].mean()))
    return float(result)


def _temperature_scale(validation_logits: np.ndarray, validation_label: np.ndarray) -> float:
    logits = torch.as_tensor(validation_logits, dtype=torch.float64)
    labels = torch.as_tensor(validation_label, dtype=torch.float64)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).clamp(0.05, 20.0))


def train_executability_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    *,
    hidden_dims: Sequence[int],
    batch_size: int,
    epochs_max: int,
    patience: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=np.float32)
    split = _strings(split)
    masks = {name: split == name for name in ("train", "validation", "test")}
    if any(mask.sum() < 8 or len(np.unique(labels[mask])) != 2 for mask in masks.values()):
        raise DependencyUnavailable("executability held-out split lacks both outcome classes")
    normalizer = FrozenStandardizer.fit(np.asarray(features)[masks["train"]])
    x = normalizer.transform(features)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = _torch_device()
    model = BinaryMLP(x.shape[1], hidden_dims).to(device)
    positive = float(labels[masks["train"]].sum())
    negative = float(masks["train"].sum() - positive)
    pos_weight = torch.tensor(negative / max(positive, 1.0), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(int(seed))
    train_indices = np.where(masks["train"])[0]
    best_auc = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(epochs_max) + 1):
        model.train()
        epoch_indices = rng.permutation(train_indices)
        for start in range(0, len(train_indices), int(batch_size)):
            indices = epoch_indices[start : start + int(batch_size)]
            batch_x = torch.as_tensor(x[indices], dtype=torch.float32, device=device)
            batch_y = torch.as_tensor(labels[indices], dtype=torch.float32, device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(batch_x), batch_y, pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(torch.as_tensor(x[masks["validation"]], dtype=torch.float32, device=device)).cpu().numpy()
        auc = float(roc_auc_score(labels[masks["validation"]], validation_logits))
        if auc > best_auc + 1.0e-12:
            best_auc = auc
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise ProtocolError("executability classifier has no selected validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        all_logits = model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    temperature = _temperature_scale(all_logits[masks["validation"]], labels[masks["validation"]])
    probability = 1.0 / (1.0 + np.exp(-np.clip(all_logits / temperature, -40.0, 40.0)))
    metrics = {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_auroc": float(best_auc),
        "temperature": temperature,
        "test_auroc": float(roc_auc_score(labels[masks["test"]], probability[masks["test"]])),
        "test_ece": expected_calibration_error(probability[masks["test"]], labels[masks["test"]]),
        "normalizer_sha256": normalizer.sha256,
    }
    return probability.astype(np.float64), metrics


def same_state_binary_pair_accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    snapshot_ids: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    snapshots = _strings(snapshot_ids)
    selected = np.ones(len(scores), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    correct = 0
    ties = 0
    count = 0
    for snapshot in sorted(set(snapshots[selected])):
        indices = np.where(selected & (snapshots == snapshot))[0]
        positive = indices[labels[indices]]
        negative = indices[~labels[indices]]
        for winner in positive:
            for loser in negative:
                delta = float(scores[winner] - scores[loser])
                correct += int(delta > 0.0)
                ties += int(delta == 0.0)
                count += 1
    if count == 0:
        raise DependencyUnavailable("no same-state success/failure candidate pairs exist")
    return {"pair_count": float(count), "accuracy": float(correct / count), "tie_fraction": float(ties / count)}


def executability_failure_cause(dataset: ExecutabilityDataset) -> dict[str, Any]:
    success = np.asarray(dataset.success, dtype=np.float64)
    snapshots = _strings(dataset.snapshot_ids)
    segments = _strings(dataset.segment_ids)
    snapshot_rates = np.asarray([success[snapshots == value].mean() for value in sorted(set(snapshots))])
    segment_rates = np.asarray([success[segments == value].mean() for value in sorted(set(segments))])
    mixed = np.asarray([
        len(np.unique(success[snapshots == value])) == 2 for value in sorted(set(snapshots))
    ])
    candidate_variance = float(np.var(segment_rates))
    snapshot_variance = float(np.var(snapshot_rates))
    mixed_fraction = float(mixed.mean())
    label = (
        "reference_executability_dominant"
        if mixed_fraction >= 0.15 and candidate_variance > snapshot_variance
        else "mixed_or_state_dominant"
    )
    return {
        "failure_cause": label,
        "between_candidate_success_rate_variance": candidate_variance,
        "between_snapshot_success_rate_variance": snapshot_variance,
        "mixed_outcome_snapshot_fraction": mixed_fraction,
        "basis": "same-physical-state closed-loop outcome variation only; no reference RMSE",
    }


__all__ = [
    "AMP_WINDOW_DIM",
    "DenoisingMLP",
    "EXECUTABILITY_BANK_SCHEMA",
    "ExecutabilityDataset",
    "FrozenDenoisingPrior",
    "FrozenStandardizer",
    "PairedAMPWindows",
    "PairedContrastiveEncoder",
    "RobotTeacherArchive",
    "binary_auc",
    "build_robot_teacher_archive",
    "cross_domain_retrieval",
    "deterministic_group_split",
    "exact_paired_amp_windows",
    "executability_failure_cause",
    "expected_calibration_error",
    "latent_distance_stability",
    "load_executability_dataset",
    "representation_preservation",
    "same_state_binary_pair_accuracy",
    "save_executability_dataset",
    "train_denoising_prior",
    "train_executability_classifier",
    "train_paired_contrastive",
]
