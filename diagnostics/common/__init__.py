"""Isaac-independent building blocks shared by discovery diagnostics.

Submodules are intentionally not eagerly imported: command-line probes can use
the data contracts on CPU-only hosts without importing simulator packages.
"""

from __future__ import annotations
