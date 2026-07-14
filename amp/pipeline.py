"""Single AMP window preprocessing path shared by policy, replay and expert."""

from __future__ import annotations

import torch

from amp.features import canonicalize_amp_window


class AMPWindowPipeline:
    """Canonicalize and flatten chronological raw AMP windows.

    All discriminator domains must enter through this class:

    ``raw [B,W,F] -> canonicalize -> flatten [B,W*F] -> committed normalizer``.
    """

    def __init__(self, history_len: int, frame_dim: int) -> None:
        if history_len <= 0 or frame_dim <= 0:
            raise ValueError("history_len and frame_dim must be positive")
        self.history_len = int(history_len)
        self.frame_dim = int(frame_dim)
        self.window_dim = self.history_len * self.frame_dim

    def _check_raw(self, windows: torch.Tensor) -> torch.Tensor:
        expected = (self.history_len, self.frame_dim)
        if windows.ndim != 3 or tuple(windows.shape[1:]) != expected:
            raise ValueError(
                f"AMP raw windows must have shape [B,{expected[0]},{expected[1]}], "
                f"got {tuple(windows.shape)}"
            )
        if windows.dtype != torch.float32:
            raise TypeError(f"AMP raw windows must be float32, got {windows.dtype}")
        if windows.numel() and not bool(torch.isfinite(windows).all()):
            raise ValueError("AMP raw windows contain non-finite values")
        return windows

    def flatten(self, windows: torch.Tensor) -> torch.Tensor:
        raw = self._check_raw(windows)
        canonical = canonicalize_amp_window(raw)
        return canonical.reshape(raw.shape[0], self.window_dim)

    def normalize_flat(self, flat_windows: torch.Tensor, normalizer) -> torch.Tensor:
        if flat_windows.ndim != 2 or flat_windows.shape[1] != self.window_dim:
            raise ValueError(
                f"AMP flat windows must have shape [B,{self.window_dim}], "
                f"got {tuple(flat_windows.shape)}"
            )
        return normalizer.normalize(flat_windows)
