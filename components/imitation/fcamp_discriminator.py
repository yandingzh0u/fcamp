from __future__ import annotations

import math

import torch
from torch import nn

from components.rollout.training_streams import CURRICULUM_STREAM, PHASE0_STREAM
from models.style_discriminator import compute_style_discriminator_loss


class FCAMPDiscriminatorMixin:
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

    @torch.no_grad()
    def _record_matched_disc_normalizer(
        self,
        current_windows_cpu: torch.Tensor,
        current_end_times_cpu: torch.Tensor,
        current_stream_ids_cpu: torch.Tensor,
        sample_count: int,
    ) -> int:
        """Record an exact fixed-mixture policy/expert moment update."""

        self.disc_normalizer.clear_pending()
        batch_size = max(1, int(self.cfg.style_prior.batch_size))
        recorded = 0
        while recorded < int(sample_count):
            count = min(batch_size, int(sample_count) - recorded)
            current, end_times = self._sample_balanced_current_windows(
                current_windows_cpu,
                current_end_times_cpu,
                current_stream_ids_cpu,
                count,
            )
            expert = self._expert_flat_at_end_times(end_times)
            self.disc_normalizer.record(current)
            self.disc_normalizer.record(expert)
            recorded += int(current.shape[0])
        return recorded

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
        normalizer_samples = self._record_matched_disc_normalizer(
            current_windows_cpu,
            current_end_times_cpu,
            current_stream_ids_cpu,
            min(current_count, batch_size * update_steps),
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
            fake_ends = torch.cat((current_ends, replay_ends.to(dtype=torch.long)), dim=0)
            expert_indices = torch.randint(fake_ends.shape[0], (batch_size,), device="cpu")
            expert_end_times = fake_ends.index_select(0, expert_indices)
            expert_raw = self._expert_flat_at_end_times(expert_end_times).to(
                device=self.env.device,
                dtype=torch.float32,
            )
            endpoint_abs_diff_total += float(
                (expert_end_times.float() - current_ends.float()).abs().mean().item()
            )
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
                "disc/expert_endpoint_abs_diff_mean": endpoint_abs_diff_total / denom,
                "disc_window/current_latest_root_xy_abs_max": canonical_root_xy_max["current"],
                "disc_window/replay_latest_root_xy_abs_max": canonical_root_xy_max["replay"],
                "disc_window/expert_latest_root_xy_abs_max": canonical_root_xy_max["expert"],
            }
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
