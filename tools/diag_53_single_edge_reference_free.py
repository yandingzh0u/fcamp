#!/usr/bin/env python3
"""Run the real no-reference local AMP causal gate."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.common.edge_stage5 import run_stage5_entrypoint

if __name__ == "__main__":
    raise SystemExit(run_stage5_entrypoint("53"))
