"""A complete custom safety policy: a callable allow(action, element, goal) -> bool.

    MMJB_POLICY=examples.policy:allow_except_delete uv run python examples/run_with_plugins.py ...

Or pass it straight to the agent: Agent(..., policy=allow_except_delete).
"""
from __future__ import annotations


def allow_except_delete(action: str, element: object, goal: str) -> bool:
    """Allow an action on an element unless the element's label mentions deleting.

    A custom policy replaces the built-in rules completely, so write the rules you want here rather
    than relying on mmjb.agent.refusal_reason. Return True to allow, False to refuse.
    action is "click", "type", "select" or "upload".
    """
    label = str(getattr(element, "label", "")).lower()
    if action == "click" and "delete" in label:
        return False
    if action in ("type", "select", "upload") and "form" not in goal.lower():
        return False
    return True


def log_refusals(action: str, element: object, goal: str) -> bool:
    """A policy that allows everything and writes down what the built-in rules would have refused."""
    allowed = allow_except_delete(action, element, goal)
    if not allowed:
        print(f"refused {action} on {getattr(element, 'label', '')!r}", flush=True)
    return allowed