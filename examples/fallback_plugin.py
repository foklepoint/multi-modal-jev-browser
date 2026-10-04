"""A complete custom fallback. A fallback is a class with one async method.

    MMJB_FALLBACK=examples.fallback_plugin:CheapFallback uv run python examples/run_with_plugins.py ...

To have it found by name instead of by path, publish it in your own package:

    [project.entry-points."mmjb.fallbacks"]
    cheap = "mypackage.fallback:CheapFallback"

and set MMJB_FALLBACK=cheap.
"""
from __future__ import annotations

from typing import Any


class CheapFallback:
    """A fallback that presses the first control on the page. It never calls a model."""

    name = "cheap"

    def __init__(self, cost: float = 0.0) -> None:
        self.cost = cost

    async def decide(self, state: dict[str, Any], images: list[str]) -> dict[str, Any]:
        """Return one action. The agent adds its cost to the run total.

        state carries the same page and element list the decision model saw, plus
        reason_you_are_asked and allow_writes.
        """
        elements = state.get("elements") or []
        for element in elements:
            if element.get("kind") == "click":
                return {
                    "action": "click",
                    "index": int(element["index"]),
                    "reason": "the first control on the page",
                    "cost": self.cost,
                }
        return {"action": "wait", "index": 0, "reason": "nothing on the page is clickable", "cost": self.cost}