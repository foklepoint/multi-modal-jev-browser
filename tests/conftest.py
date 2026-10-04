"""Test setup: the shared fixtures live in tests/support.py and are exposed here to pytest."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import browser, browser_kwargs, site  # noqa: E402,F401