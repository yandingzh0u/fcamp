from __future__ import annotations

import torch


class OnPolicyStateBankMixin:
    """On-policy state bank to fix the train/validation state-distribution mismatch.

    Root cause this addresses: training resets every group to a *reference* pose at a sampled
    phase (clean phase reset + tiny noise), but validation resets ONCE at phase 0 and rolls
    the policy continuously for the whole clip. So at, say, phase ~400 the validation robot
    is in the policy's own self-generated drifted state (accumulated wrist-low / contact /
    velocity / posture error), while training only ever optimizes the clean reference state at
    that phase. Phase is NOT state. The policy is never trained on the long-horizon on-policy
    state distribution it is actually scored on, which is why validation collapses around the
    phase where drift first becomes fatal.

    Fix: periodically roll the deterministic policy from phase 0 (exactly like validation),
    snapshot the REAL simulator state at a spread of phases, and let a fraction of training
    group starts restore these on-policy states instead of clean reference resets. The GRPO
    same-state contract still holds because each group's branches are replicated from the one
    restored state (handled by the caller via _replicate_group_reset_state).
    """

    def _init_onpolicy_state_bank(self) -> None:
        cfg = self.cfg
        self.onpolicy_bank_enabled = bool(getattr(cfg, "onpolicy_state_bank", False))
        # Fraction of GRPO groups per update that start from a banked on-policy state.
        self.onpolicy_state_ratio = float(getattr(cfg, "onpolicy_state_ratio", 0.5))
        # Refresh the bank (re-roll the current policy from phase 0) every N updates.
        self.onpolicy_refresh_every = max(1, int(getattr(cfg, "onpolicy_refresh_every", 25)))
        # How many env-steps to roll from phase 0 when building the bank. Should comfortably
        # exceed the phase region where the policy currently dies so the bank actually
        # contains the hard drifted states (e.g. 300-430).
        self.onpolicy_bank_rollout_steps = int(getattr(cfg, "onpolicy_bank_rollout_steps", 480))
        # Only bank states at/after this phase: early phases are already covered well by the
        # clean phase-0 start course, the value of on-policy states is in the drifted mid/late
        # region.
        self.onpolicy_bank_min_phase = int(getattr(cfg, "onpolicy_bank_min_phase", 80))
        # Max number of banked snapshots to keep (bounds memory). Each snapshot is one env's
        # full state at one phase.
        self.onpolicy_bank_capacity = int(getattr(cfg, "onpolicy_bank_capacity", 16384))
        # Minimum bank population before on-policy starts are used. Early in training the
        # policy cannot reach mid/late phases, so the bank is tiny; using a near-empty bank
        # would make many groups start from the same one or two states (degenerate). Below
        # this size we fall back to clean resets for the whole update; the bank self-fills as
        # the policy learns to reach deeper phases (natural curriculum).
        self.onpolicy_bank_min_size = int(getattr(cfg, "onpolicy_bank_min_size", 256))
        self._onpolicy_bank: dict[str, torch.Tensor] | None = None
        self._onpolicy_bank_size = 0

    # ------------------------------------------------------------------ capture / build
    def _onpolicy_capture_fields(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Capture the full per-env simulator + bookkeeping state for the given envs."""
        robot = self.env.robot
        root_state = robot.data.root_state_w.index_select(0, env_ids).clone()
        # store root position in env-local frame so it can be restored under any env origin.
        root_state[:, :3] = root_state[:, :3] - self.env.scene.env_origins.index_select(0, env_ids)
        fields = {
            "root_state_local": root_state,
            "joint_pos": robot.data.joint_pos.index_select(0, env_ids).clone(),
            "joint_vel": robot.data.joint_vel.index_select(0, env_ids).clone(),
            "phase_steps": self.env.phase_steps.index_select(0, env_ids).clone(),
            "last_action": self.env.last_action.index_select(0, env_ids).clone(),
            "prev_action": self.env.prev_action.index_select(0, env_ids).clone(),
        }
        contact = getattr(self.env, "contact_sensor", None)
        if contact is not None:
            data = contact.data
            for attr in ("net_forces_w", "net_forces_w_history"):
                buf = getattr(data, attr, None)
                if buf is not None:
                    fields[f"contact_{attr}"] = buf.index_select(0, env_ids).clone()
        return fields

    @torch.no_grad()
    def _refresh_onpolicy_state_bank(self, update_idx: int) -> None:
        if not self.onpolicy_bank_enabled:
            return
        # Snapshot the live training state so the bank rollout does not disturb training.
        snapshot = self._snapshot_env_state()
        try:
            self._build_onpolicy_state_bank()
        finally:
            self._restore_env_state(snapshot)
            # Refresh the cached training observation after restoring.
            self.current_observation = self.env.get_observation()
        if self._onpolicy_bank_size > 0:
            phases = self._onpolicy_bank["phase_steps"]
            print(
                f"[STATE_BANK] update={update_idx} refreshed size={self._onpolicy_bank_size} "
                f"phase_min={int(phases.min())} phase_mean={float(phases.float().mean()):.1f} "
                f"phase_max={int(phases.max())}",
                flush=True,
            )
        else:
            print(f"[STATE_BANK] update={update_idx} refreshed but captured 0 states", flush=True)

    @torch.no_grad()
    def _build_onpolicy_state_bank(self) -> None:
        num_envs = self.env.num_envs
        device = self.env.device
        # Start every env at phase 0, exactly like validation.
        phase0 = torch.zeros(num_envs, dtype=torch.long, device=device)
        obs = self.env.reset(phase_indices=phase0)

        captured: list[dict[str, torch.Tensor]] = []
        captured_count = 0
        alive = torch.ones(num_envs, dtype=torch.bool, device=device)
        # Capture each env at most once, when it first crosses min_phase + a per-env random
        # offset so the pool spreads across phases rather than clumping at min_phase.
        capture_phase = self.onpolicy_bank_min_phase + torch.randint(
            0,
            max(1, self.onpolicy_bank_rollout_steps - self.onpolicy_bank_min_phase),
            (num_envs,),
            device=device,
        )
        captured_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)

        cached_chunk: torch.Tensor | None = None
        chunk_index = int(self.cfg.horizon)
        for _ in range(self.onpolicy_bank_rollout_steps):
            if not self.simulation_app.is_running():
                break
            if cached_chunk is None or chunk_index >= int(self.cfg.horizon):
                cached_chunk = self._deterministic_actions(obs)
                chunk_index = 0
            action = cached_chunk[:, chunk_index, :]
            chunk_index += 1
            if bool(alive.logical_not().any()):
                action = torch.where(alive.unsqueeze(-1), action, torch.zeros_like(action))

            # Capture BEFORE stepping: envs that are alive and have reached their capture phase.
            ready = alive & (~captured_mask) & (self.env.phase_steps >= capture_phase)
            ready_ids = ready.nonzero(as_tuple=False).squeeze(-1)
            if ready_ids.numel() > 0 and captured_count < self.onpolicy_bank_capacity:
                take = min(int(ready_ids.numel()), self.onpolicy_bank_capacity - captured_count)
                ready_ids = ready_ids[:take]
                captured.append(self._onpolicy_capture_fields(ready_ids))
                captured_count += int(ready_ids.numel())
                captured_mask[ready_ids] = True

            obs, _, done, _ = self.env.step(action, auto_reset=False)
            alive = alive & ~done
            if not bool(alive.any()) or captured_count >= self.onpolicy_bank_capacity:
                break

        if captured:
            self._onpolicy_bank = {
                key: torch.cat([chunk[key] for chunk in captured], dim=0)
                for key in captured[0]
            }
            self._onpolicy_bank_size = int(self._onpolicy_bank["phase_steps"].numel())
        else:
            self._onpolicy_bank = None
            self._onpolicy_bank_size = 0

    # ------------------------------------------------------------------ apply at group reset
    @torch.no_grad()
    def _apply_onpolicy_group_starts(self, generation_count: int) -> torch.Tensor | None:
        """Restore banked on-policy states into a fraction of group anchor envs.

        Returns the group_ids that were given an on-policy start (so the caller can replicate
        the restored state across each such group's branches), or None if the bank is unused.
        """
        if not self.onpolicy_bank_enabled or self._onpolicy_bank_size == 0:
            return None
        if self._onpolicy_bank_size < self.onpolicy_bank_min_size:
            # Bank too small to draw diverse starts from; fall back to clean resets this update.
            return None
        if self.onpolicy_state_ratio <= 0.0:
            return None
        device = self.env.device
        num_groups = self.num_grpo_groups
        num_onpolicy = int(round(num_groups * min(1.0, self.onpolicy_state_ratio)))
        if num_onpolicy <= 0:
            return None

        group_ids = torch.randperm(num_groups, device=device)[:num_onpolicy]
        anchor_env_ids = group_ids * generation_count
        # Sample banked snapshots (with replacement) for these groups.
        bank_idx = torch.randint(0, self._onpolicy_bank_size, (num_onpolicy,), device=device)

        bank = self._onpolicy_bank
        root_state_local = bank["root_state_local"].index_select(0, bank_idx)
        joint_pos_full = bank["joint_pos"].index_select(0, bank_idx)
        joint_vel_full = bank["joint_vel"].index_select(0, bank_idx)
        phase_steps = bank["phase_steps"].index_select(0, bank_idx)
        last_action = bank["last_action"].index_select(0, bank_idx)
        prev_action = bank["prev_action"].index_select(0, bank_idx)

        self.env.scene.reset(env_ids=anchor_env_ids)
        self.env._write_robot_state(
            root_pos=root_state_local[:, :3],
            root_quat=root_state_local[:, 3:7],
            root_lin_vel=root_state_local[:, 7:10],
            root_ang_vel=root_state_local[:, 10:13],
            joint_pos=joint_pos_full[:, self.env.action_joint_ids],
            joint_vel=joint_vel_full[:, self.env.action_joint_ids],
            env_ids=anchor_env_ids,
        )
        self.env.phase_steps[anchor_env_ids] = phase_steps
        self.env.episode_steps[anchor_env_ids] = 0
        self.env.last_action[anchor_env_ids] = last_action
        self.env.prev_action[anchor_env_ids] = prev_action
        # Restore contact history on the anchor envs if it was banked.
        contact = getattr(self.env, "contact_sensor", None)
        if contact is not None:
            data = contact.data
            for attr in ("net_forces_w", "net_forces_w_history"):
                key = f"contact_{attr}"
                buf = getattr(data, attr, None)
                if buf is not None and key in bank:
                    buf[anchor_env_ids] = bank[key].index_select(0, bank_idx)
        self.env.scene.update(self.env.physics_dt)
        return group_ids
