from __future__ import annotations

import json

import pytest

from diagnostics.common import isaac_exit
from diagnostics.common.manifest import PASS, diagnostic_result, write_json_exclusive


def test_hard_exit_is_opt_in_for_real_cli_only(monkeypatch) -> None:
    calls: list[int] = []

    def fake_exit(code: int) -> None:
        calls.append(code)
        raise RuntimeError("hard exit")

    monkeypatch.setattr(isaac_exit.os, "_exit", fake_exit)
    assert isaac_exit.finish_isaac_entrypoint(
        7, isaac_launched=True, allow_hard_exit=False
    ) == 7
    assert calls == []
    with pytest.raises(RuntimeError, match="hard exit"):
        isaac_exit.finish_isaac_entrypoint(
            0, isaac_launched=True, allow_hard_exit=True
        )
    assert calls == [0]


def test_status_is_durable_before_hard_exit_barrier(tmp_path, monkeypatch) -> None:
    target = tmp_path / "status.json"
    result = diagnostic_result("safe-exit", PASS, summary="published")
    write_json_exclusive(target, result)

    def fake_exit(code: int) -> None:
        # This models the only invariant that matters for the known Kit hang:
        # evidence is present and parseable before process termination.
        assert json.loads(target.read_text(encoding="utf-8"))["status"] == PASS
        raise RuntimeError(f"exit={code}")

    monkeypatch.setattr(isaac_exit.os, "_exit", fake_exit)
    with pytest.raises(RuntimeError, match="exit=0"):
        isaac_exit.finish_isaac_entrypoint(
            0, isaac_launched=True, allow_hard_exit=True
        )
