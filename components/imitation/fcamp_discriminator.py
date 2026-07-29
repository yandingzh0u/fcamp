from __future__ import annotations

import math

import torch
from torch import nn

from components.rollout.training_streams import CURRICULUM_STREAM, PHASE0_STREAM
from models.style_discriminator import compute_style_discriminator_loss


class FCAMPDiscriminatorMixin:
    _ENDPOINT_HISTOGRAM_BINS = 16
    _EXPERT_SAMPLING_SEED_OFFSET = 0x4643414D50

    def _sample_cpu_flat_with_end_times(
        self,
        windows_cpu: torch.Tensor,
        end_times_cpu: torch.Tensor,
        batch_size: int,
        *,
        stream_ids_cpu: torch.Tensor,
        stream_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = torch.arange(windows_cpu.shape[0], device="cpu")
        candidates = candidates[
            stream_ids_cpu.to(device="cpu", dtype=torch.int8)
            == int(stream_id)
        ]
        draw = torch.randint(
            candidates.numel(),
            (int(batch_size),),
            device="cpu",
        )
        indices = candidates.index_select(0, draw)
        raw = windows_cpu.index_select(0, indices).to(
            device=self.env.device,
            dtype=torch.float32,
            non_blocking=False,
        )
        ends = end_times_cpu.index_select(0, indices)
        return raw, ends

    def _disc_stream_quotas(
        self,
        batch_size: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        phase0 = int(
            round(
                int(batch_size)
                * float(self.cfg.streams.phase0_fraction)
            )
        )
        phase0 = min(max(1, phase0), int(batch_size) - 1)
        return (
            (PHASE0_STREAM, phase0),
            (CURRICULUM_STREAM, int(batch_size) - phase0),
        )

    def _sample_balanced_current_windows(
        self,
        windows_cpu: torch.Tensor,
        end_times_cpu: torch.Tensor,
        stream_ids_cpu: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_parts: list[torch.Tensor] = []
        end_parts: list[torch.Tensor] = []
        for stream_id, count in self._disc_stream_quotas(batch_size):
            raw, ends = self._sample_cpu_flat_with_end_times(
                windows_cpu,
                end_times_cpu,
                count,
                stream_ids_cpu=stream_ids_cpu,
                stream_id=stream_id,
            )
            raw_parts.append(raw)
            end_parts.append(ends)
        return torch.cat(raw_parts, dim=0), torch.cat(end_parts, dim=0)

    def _sample_balanced_replay_windows(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_parts: list[torch.Tensor] = []
        end_parts: list[torch.Tensor] = []
        for stream_id, count in self._disc_stream_quotas(batch_size):
            frames, ends = self.disc_window_replay.sample(
                count,
                stream_id=stream_id,
            )
            frame_parts.append(frames)
            end_parts.append(ends)
        return torch.cat(frame_parts, dim=0), torch.cat(end_parts, dim=0)

    def _expert_flat_at_end_times(self, end_times_cpu: torch.Tensor) -> torch.Tensor:
        raw = self.env.motion.get_fcamp_demo_windows_at_end_indices(
            end_times_cpu.to(device=self.env.device, dtype=torch.long),
            self.imitation_history_steps,
        )
        return self.imitation_pipeline.flatten(raw)

    def _sample_expert_end_times(self, sample_count: int) -> torch.Tensor:
        """Draw demo endpoints independently of every policy-side distribution.

        The expert measure is fixed by the configured demonstration interval.
        This method deliberately accepts only a count: current/replay endpoints,
        reset phases, curriculum state, and failure frontiers cannot be used to
        reweight discriminator positives.
        """

        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
        ):
            raise ValueError("expert sample_count must be a positive integer")
        endpoint_min = int(self.env.motion_start_phase)
        endpoint_max = int(self.env.motion_end_phase)
        motion_frames = int(self.env.motion.num_frames)
        if (
            endpoint_min < 0
            or endpoint_max < endpoint_min
            or endpoint_max >= motion_frames
        ):
            raise RuntimeError(
                "configured expert endpoint interval lies outside the motion"
            )
        generator = getattr(self, "expert_sampling_generator", None)
        if (
            not isinstance(generator, torch.Generator)
            or str(generator.device) != "cpu"
        ):
            raise RuntimeError(
                "independent expert sampling requires its dedicated CPU generator"
            )
        return torch.randint(
            endpoint_min,
            endpoint_max + 1,
            (sample_count,),
            generator=generator,
            device="cpu",
            dtype=torch.long,
        )

    def _sample_expert_flat(
        self,
        sample_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an exogenous expert batch and its independently drawn endpoints."""

        end_times = self._sample_expert_end_times(sample_count)
        return self._expert_flat_at_end_times(end_times), end_times

    def _endpoint_histogram(self, end_times: torch.Tensor) -> torch.Tensor:
        """Count endpoints in fixed bins over the configured demo interval."""

        endpoints = end_times.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        endpoint_min = int(self.env.motion_start_phase)
        endpoint_max = int(self.env.motion_end_phase)
        support_size = endpoint_max - endpoint_min + 1
        if support_size <= 0:
            raise RuntimeError("configured endpoint histogram interval is empty")
        if endpoints.numel() and (
            bool((endpoints < endpoint_min).any())
            or bool((endpoints > endpoint_max).any())
        ):
            raise RuntimeError(
                "discriminator endpoint lies outside the configured demo interval"
            )
        num_bins = min(self._ENDPOINT_HISTOGRAM_BINS, support_size)
        bin_ids = torch.div(
            (endpoints - endpoint_min) * num_bins,
            support_size,
            rounding_mode="floor",
        )
        return torch.bincount(bin_ids, minlength=num_bins)

    def _uniform_expert_endpoint_histogram(self) -> torch.Tensor:
        """Return exact bin masses for the discrete-uniform demo measure."""

        return self._endpoint_histogram(
            torch.arange(
                int(self.env.motion_start_phase),
                int(self.env.motion_end_phase) + 1,
                device="cpu",
                dtype=torch.long,
            )
        )

    @torch.no_grad()
    def _record_disc_normalizer(
        self,
        current_windows_cpu: torch.Tensor,
        current_end_times_cpu: torch.Tensor,
        current_stream_ids_cpu: torch.Tensor,
        sample_count: int,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Record policy moments plus an equal, exogenous expert sample."""

        self.disc_normalizer.clear_pending()
        batch_size = max(1, int(self.cfg.style_prior.batch_size))
        recorded = 0
        current_histogram: torch.Tensor | None = None
        expert_histogram: torch.Tensor | None = None
        while recorded < int(sample_count):
            count = min(batch_size, int(sample_count) - recorded)
            current, current_end_times = self._sample_balanced_current_windows(
                current_windows_cpu,
                current_end_times_cpu,
                current_stream_ids_cpu,
                count,
            )
            expert, expert_end_times = self._sample_expert_flat(count)
            current_batch_histogram = self._endpoint_histogram(current_end_times)
            expert_batch_histogram = self._endpoint_histogram(expert_end_times)
            if expert_histogram is None:
                current_histogram = current_batch_histogram
                expert_histogram = expert_batch_histogram
            else:
                if current_histogram is None:
                    raise RuntimeError(
                        "normalizer current endpoint histogram was not initialized"
                    )
                current_histogram += current_batch_histogram
                expert_histogram += expert_batch_histogram
            self.disc_normalizer.record(current)
            self.disc_normalizer.record(expert)
            recorded += int(current.shape[0])
        if current_histogram is None or expert_histogram is None:
            raise RuntimeError("normalizer recorded no policy/expert samples")
        return recorded, current_histogram, expert_histogram

    def _discriminator_update(
        self,
        update_idx: int,
        *,
        rollout: dict,
        commit_normalizer_before_training: bool = False,
    ) -> dict[str, float]:
        current_windows_cpu = rollout["current_disc_windows"]
        current_end_times_cpu = rollout["current_disc_end_times"]
        current_stream_ids_cpu = rollout["current_disc_stream_ids"]
        current_count = int(current_windows_cpu.shape[0])
        input_version = int(self.disc_version)
        if not (
            current_end_times_cpu.shape == current_stream_ids_cpu.shape
            and current_end_times_cpu.shape[0] == current_count
        ):
            raise RuntimeError(
                "FCAMP current discriminator windows have misaligned metadata"
            )
        current_phase0_count = int(
            (current_stream_ids_cpu == PHASE0_STREAM).sum().item()
        )
        current_curriculum_count = int(
            (current_stream_ids_cpu == CURRICULUM_STREAM).sum().item()
        )
        batch_size = int(self.cfg.style_prior.batch_size)
        first_replay_window_frames, first_replay_ends = (
            self._sample_balanced_replay_windows(batch_size)
        )
        possible_steps = math.ceil(current_count / batch_size) * int(
            self.cfg.style_prior.epochs
        )
        update_steps = min(possible_steps, int(self.cfg.style_prior.max_updates_per_iteration))
        # Record pending statistics now, but train D on the same committed
        # normalization snapshot that produced this rollout's rewards/history.
        # Commit only after D_old has been consumed, matching method/amp.py.
        self.disc_normalizer.freeze()
        (
            normalizer_samples,
            normalizer_current_histogram,
            normalizer_expert_histogram,
        ) = (
            self._record_disc_normalizer(
                current_windows_cpu,
                current_end_times_cpu,
                current_stream_ids_cpu,
                min(current_count, batch_size * update_steps),
            )
        )
        committed_before_training = False
        if commit_normalizer_before_training:
            # The first discriminator must not see raw, unscaled 3728-D input.
            # Commit matched expert/policy moments first; the rollout itself is
            # discarded, so no actor can consume its random-D reward snapshot.
            self.disc_normalizer.unfreeze()
            committed_before_training = bool(self.disc_normalizer.commit())
            self.disc_normalizer.freeze()
        totals: dict[str, float] = {}
        grad_total = 0.0
        canonical_root_xy_max = {"current": 0.0, "replay": 0.0, "expert": 0.0}
        endpoint_abs_diff_total = 0.0
        endpoint_histograms: dict[str, torch.Tensor] | None = None
        self.discriminator.train()
        for disc_step in range(update_steps):
            current_raw, current_ends = self._sample_balanced_current_windows(
                current_windows_cpu,
                current_end_times_cpu,
                current_stream_ids_cpu,
                batch_size,
            )
            if disc_step == 0:
                replay_window_frames = first_replay_window_frames
                replay_ends = first_replay_ends
            else:
                replay_window_frames, replay_ends = (
                    self._sample_balanced_replay_windows(batch_size)
                )
            replay_raw = self.imitation_pipeline.flatten(
                replay_window_frames
            ).to(
                device=self.env.device,
                dtype=torch.float32,
                non_blocking=False,
            )
            expert_raw, expert_end_times = self._sample_expert_flat(batch_size)
            expert_raw = expert_raw.to(
                device=self.env.device,
                dtype=torch.float32,
            )
            endpoint_abs_diff_total += float(
                (expert_end_times.float() - current_ends.float()).abs().mean().item()
            )
            step_histograms = {
                "current_train": self._endpoint_histogram(current_ends),
                "replay_train": self._endpoint_histogram(replay_ends),
                "expert_train": self._endpoint_histogram(expert_end_times),
            }
            if endpoint_histograms is None:
                endpoint_histograms = {
                    name: counts.clone()
                    for name, counts in step_histograms.items()
                }
            else:
                for name, counts in step_histograms.items():
                    endpoint_histograms[name] += counts
            if disc_step == 0:
                for name, raw in (
                    ("current", current_raw),
                    ("replay", replay_raw),
                    ("expert", expert_raw),
                ):
                    window = raw.view(-1, self.imitation_history_steps, self.imitation_frame_dim)
                    canonical_root_xy_max[name] = float(
                        window[:, -1, :2].abs().max().item()
                    )
            current = self.imitation_pipeline.normalize_flat(current_raw, self.disc_normalizer)
            replay = self.imitation_pipeline.normalize_flat(replay_raw, self.disc_normalizer)
            expert = self.imitation_pipeline.normalize_flat(expert_raw, self.disc_normalizer)
            output = compute_style_discriminator_loss(
                self.discriminator,
                expert_observations=expert,
                policy_observations=current,
                replay_observations=replay,
                gradient_penalty_weight=float(self.cfg.style_prior.grad_penalty),
                logit_regularization_weight=float(self.cfg.style_prior.logit_reg),
            )
            self.disc_optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            # Record the true norm without modifying gradients; standard
            # MimicKit AMP does not configure discriminator grad clipping.
            grad = nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), float("inf")
            )
            self.disc_optimizer.step()
            grad_total += float(grad)
            for key, value in output.metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
        self.disc_version += 1
        if commit_normalizer_before_training:
            committed = committed_before_training
        else:
            self.disc_normalizer.unfreeze()
            committed = self.disc_normalizer.commit()
            self.disc_normalizer.freeze()
        denom = max(update_steps, 1)
        metrics = {key: value / denom for key, value in totals.items()}
        metrics.update(
            {
                "disc/grad_norm": grad_total / denom,
                "disc/lr": float(self.cfg.style_prior.learning_rate),
                "disc/update_steps": float(update_steps),
                "disc/version": float(self.disc_version),
                "disc/input_version": float(input_version),
                "disc_norm/committed_this_update": float(committed),
                "disc_norm/committed_before_training": float(
                    commit_normalizer_before_training
                    and committed_before_training
                ),
                "disc_norm/policy_samples_update": float(normalizer_samples),
                "disc_norm/expert_samples_update": float(normalizer_samples),
                "disc/current_count": float(current_count),
                "disc/current_phase0_count": float(current_phase0_count),
                "disc/current_curriculum_count": float(
                    current_curriculum_count
                ),
                "disc/current_phase0_fraction": float(
                    current_phase0_count / max(current_count, 1)
                ),
                "disc/training_phase0_fraction": float(
                    self.cfg.streams.phase0_fraction
                ),
                "disc/replay_phase0_fraction": float(
                    self.cfg.streams.phase0_fraction
                ),
                "disc/stream_quota_available": 1.0,
                "disc/current_unique_endpoint_count": float(
                    torch.unique(current_end_times_cpu).numel()
                ),
                "disc/replay_fallback_current_count": 0.0,
                "disc/expert_sampling_contract_active": 1.0,
                "disc/expert_sampling_uniform_integer_contract_active": 1.0,
                "disc_norm/expert_sampling_contract_active": 1.0,
                "disc/expert_current_endpoint_abs_diff_mean": (
                    endpoint_abs_diff_total / denom
                ),
                "disc_window/current_latest_root_xy_abs_max": canonical_root_xy_max["current"],
                "disc_window/replay_latest_root_xy_abs_max": canonical_root_xy_max["replay"],
                "disc_window/expert_latest_root_xy_abs_max": canonical_root_xy_max["expert"],
            }
        )
        if endpoint_histograms is None:
            raise RuntimeError("discriminator update produced no endpoint samples")
        endpoint_histograms["current_pool"] = self._endpoint_histogram(
            current_end_times_cpu
        )
        histogram_bins = int(endpoint_histograms["expert_train"].numel())
        metrics["disc_endpoint/histogram_bins"] = float(histogram_bins)
        endpoint_fractions: dict[str, torch.Tensor] = {}
        for domain, counts in endpoint_histograms.items():
            total = int(counts.sum().item())
            if total <= 0:
                raise RuntimeError(
                    f"discriminator {domain} endpoint histogram is empty"
                )
            fractions = counts.to(dtype=torch.float64) / float(total)
            endpoint_fractions[domain] = fractions
            metrics[f"disc_endpoint/{domain}_sample_count"] = float(total)
            for bin_index, fraction in enumerate(fractions.tolist()):
                metrics[
                    f"disc_endpoint/{domain}_bin_{bin_index:02d}_fraction"
                ] = float(fraction)
        expert_fractions = endpoint_fractions["expert_train"]
        expected_expert_fractions = self._uniform_expert_endpoint_histogram().to(
            dtype=torch.float64
        )
        expected_expert_fractions /= expected_expert_fractions.sum()
        for bin_index, fraction in enumerate(
            expected_expert_fractions.tolist()
        ):
            metrics[
                f"disc_endpoint/expert_uniform_bin_{bin_index:02d}_fraction"
            ] = float(fraction)
        metrics["disc_endpoint/expert_train_max_abs_uniform_error"] = float(
            (
                expert_fractions
                - expected_expert_fractions
            )
            .abs()
            .max()
            .item()
        )
        metrics["disc_endpoint/expert_train_uniform_tv"] = float(
            0.5
            * (
                expert_fractions
                - expected_expert_fractions
            )
            .abs()
            .sum()
            .item()
        )
        for left, right in (
            ("current_train", "expert_train"),
            ("replay_train", "expert_train"),
            ("current_train", "replay_train"),
        ):
            metrics[f"disc_endpoint/{left}_{right}_tv"] = float(
                0.5
                * (
                    endpoint_fractions[left]
                    - endpoint_fractions[right]
                )
                .abs()
                .sum()
                .item()
            )
        normalizer_current_total = int(
            normalizer_current_histogram.sum().item()
        )
        normalizer_total = int(normalizer_expert_histogram.sum().item())
        if (
            normalizer_current_total != normalizer_samples
            or normalizer_total != normalizer_samples
        ):
            raise RuntimeError(
                "normalizer endpoint histograms do not conserve samples"
            )
        normalizer_current_fractions = normalizer_current_histogram.to(
            dtype=torch.float64
        )
        normalizer_current_fractions /= float(normalizer_current_total)
        normalizer_expert_fractions = normalizer_expert_histogram.to(
            dtype=torch.float64
        )
        normalizer_expert_fractions /= float(normalizer_total)
        metrics["disc_norm/current_endpoint_sample_count"] = float(
            normalizer_current_total
        )
        metrics["disc_norm/expert_endpoint_sample_count"] = float(
            normalizer_total
        )
        for domain, fractions in (
            ("current", normalizer_current_fractions),
            ("expert", normalizer_expert_fractions),
        ):
            for bin_index, fraction in enumerate(fractions.tolist()):
                metrics[
                    f"disc_norm/{domain}_endpoint_bin_{bin_index:02d}_fraction"
                ] = float(fraction)
        metrics["disc_norm/expert_endpoint_max_abs_uniform_error"] = float(
            (
                normalizer_expert_fractions
                - expected_expert_fractions
            )
            .abs()
            .max()
            .item()
        )
        metrics["disc_norm/expert_endpoint_uniform_tv"] = float(
            0.5
            * (
                normalizer_expert_fractions
                - expected_expert_fractions
            )
            .abs()
            .sum()
            .item()
        )
        metrics["disc_norm/current_expert_endpoint_tv"] = float(
            0.5
            * (
                normalizer_current_fractions
                - normalizer_expert_fractions
            )
            .abs()
            .sum()
            .item()
        )
        metrics.update(
            {
                key.replace("replay/", "disc_replay/"): value
                for key, value in self.disc_window_replay.statistics(
                    current_update=update_idx
                ).items()
            }
        )
        return metrics
