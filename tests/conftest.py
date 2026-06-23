"""Shared test fixtures. Puts the repo root on `sys.path` so test modules
can `from scripts.<x> import ...` regardless of how pytest is invoked.

Also stubs OpenEO credentials before any production module imports so
`utils.config` doesn't raise during test collection."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENEO_USERNAME", "test-user")
os.environ.setdefault("OPENEO_PASSWORD", "test-pass")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
