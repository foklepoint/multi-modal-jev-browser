"""Live tests: the same local site driven by a real decision model.

They are skipped unless the keys for that model are in the environment, so the offline suite stays
offline. Run them with:

    CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... uv run pytest -m live
    OPENROUTER_API_KEY=... uv run pytest -m live
    MMJB_DECIDER=llm OPENAI_API_KEY=... uv run pytest -m live
"""
from __future__ import annotations

import os

import pytest
from support import FakeFallback

from mmjb.agent import Agent

pytestmark = pytest.mark.live

GOAL = "Open the submission form in a new tab, fill it in with the facts and send it."
FACTS = {"name": "Ada Lovelace", "message": "This tool is worth a directory listing."}


def clef_ready() -> bool:
    return bool(os.environ.get("CLOUDFLARE_API_TOKEN") and os.environ.get("CLOUDFLARE_ACCOUNT_ID"))


def jev_ready() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def llm_ready() -> bool:
    return bool(
        os.environ.get("MMJB_DECIDER") == "llm"
        and any(os.environ.get(name) for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"))
    )


def build(decider_name: str, browser, max_steps: int = 14):
    """An agent with a real decision model and a scripted fallback, so the run ends on its own."""
    return Agent(
        goal=GOAL,
        facts=FACTS,
        allow_writes=True,
        decider=build_decider_by_name(decider_name),
        fallback=FakeFallback([{"action": "done", "index": 0, "fact": "", "reason": "the goal looks met"}]),
        browser=browser,
        max_steps=max_steps,
        budget_seconds=240.0,
    )


def build_decider_by_name(name: str):
    from mmjb.deciders import build_decider

    return build_decider(name)


@pytest.mark.parametrize("name,ready", [("clef", clef_ready), ("jev", jev_ready), ("llm", llm_ready)])
async def test_a_real_decider_fills_the_form(browser, site, name, ready):
    if not ready():
        pytest.skip(f"{name} is not configured, so this live test is skipped")
    agent = build(name, browser)
    try:
        result = await agent.run(start_url=site + "/index.html")
    finally:
        await agent.close()
    print(result.summary())
    assert result.steps[0].effect.startswith("opened a new tab"), "the click on the new tab must be followed"
    operations = [record.operation for record in result.steps]
    assert "TYPE_TEXT" in operations, f"the real {name} never typed: {result.summary()}"
    assert "SELECT" in operations, f"the real {name} never filled the dropdown: {result.summary()}"
    assert result.outcome in ("verified", "unverified"), result.summary()
    if result.outcome == "verified":
        assert "confirmation" in result.steps[-1].effect


@pytest.mark.parametrize("name,ready", [("clef", clef_ready), ("jev", jev_ready), ("llm", llm_ready)])
async def test_a_real_decider_reads_a_locked_submit_button(browser, site, name, ready):
    if not ready():
        pytest.skip(f"{name} is not configured, so this live test is skipped")
    agent = Agent(
        goal="Fill both fields on the locked form with the facts and press the send button.",
        facts=FACTS,
        allow_writes=True,
        decider=build_decider_by_name(name),
        fallback=FakeFallback([{"action": "done", "index": 0, "fact": "", "reason": "the goal looks met"}]),
        browser=browser,
        max_steps=10,
        budget_seconds=180.0,
    )
    try:
        result = await agent.run(start_url=site + "/locked.html")
        after = await browser.observe()
    finally:
        await agent.close()
    print(result.summary())
    operations = [record.operation for record in result.steps]
    assert operations.count("TYPE_TEXT") == 2, f"the real {name} did not fill both fields: {result.summary()}"
    assert "Send" not in after.disabled_controls, "the submit button must be clickable once the fields are full"