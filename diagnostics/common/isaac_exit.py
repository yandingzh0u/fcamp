"""Safe finalization for standalone diagnostics that launch Isaac/Kit.

On the target runner, ``AppLauncher.app.close()`` can deadlock after a
successful collection.  A diagnostic must never lose its evidence to that
cleanup hang.  Call this *after* every result/status/log artifact has been
written through the durable writers, and only from a real CLI entrypoint.
"""

from __future__ import annotations

import os
import sys


def flush_standard_streams() -> None:
    """Best-effort flush of human-readable progress before process exit."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def finish_isaac_entrypoint(
    exit_code: int,
    *,
    isaac_launched: bool,
    allow_hard_exit: bool,
) -> int:
    """Flush and, for a real Isaac CLI only, terminate without ``app.close``.

    ``allow_hard_exit`` must be a module-level ``__name__ == '__main__'``
    constant.  This deliberately keeps imported functions and unit tests
    normal-returning, while the subprocess that owns Kit safely releases all
    resources through process termination.
    """

    code = int(exit_code)
    flush_standard_streams()
    if isaac_launched and allow_hard_exit:
        os._exit(code)
    return code
