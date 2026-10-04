"""The step loop end to end, driven by a decision model with no network behind it.

Every test here runs the real loop against the local site: real observation, real mouse clicks, real
typing, real tabs. Only the decision model is scripted.
"""
from __future__ import annotations

import pytest
from support import FakeFallback, ScriptedDecider

from mmjb.agent import Agent

FACTS = {"name": "Ada Lovelace", "message": "I would like to hear more."}
GOAL = "Open the submission form in a new tab, fill it in with the facts and send it."


def build_agent(browser, decider, fallback=None, **kwargs):
    settings = {
        "goal": GOAL,
        "facts": FACTS,
        "allow_writes": True,
        "decider": decider,
        "fallback": fallback,
        "browser": browser,
        "max_steps": 12,
        "budget_seconds": 120.0,
    }
    settings.update(kwargs)
    return Agent(**settings)


@pytest.mark.slow
async def test_form_is_filled_in_a_new_tab_and_the_confirmation_is_accepted(browser, site):
    decider = ScriptedDecider("form")
    agent = build_agent(browser, decider, FakeFallback())
    result = await agent.run(start_url=site + "/index.html")

    assert result.outcome == "verified", result.summary()
    assert result.fallback_calls == 0, "the cheap decision model should never need the fallback here"
    operations = [record.operation for record in result.steps]
    assert "TYPE_TEXT" in operations
    assert "SELECT" in operations
    assert "CLICK" in operations

    tab_step = result.steps[0]
    assert tab_step.operation == "CLICK"
    assert tab_step.effect.startswith("opened a new tab and moved to it")
    assert "form.html" in tab_step.effect

    final = result.steps[-1]
    assert final.operation == "DONE"
    assert "confirmation" in final.effect
    assert result.final_url.endswith("/form.html")

    typed = [record.fact for record in result.steps if record.operation == "TYPE_TEXT"]
    assert typed == ["name", "message"]


@pytest.mark.slow
async def test_a_control_clicked_three_times_with_no_effect_is_hidden(browser, site):
    decider = ScriptedDecider("dead")
    agent = build_agent(browser, decider, FakeFallback(), goal="Press the dead button", max_steps=6)
    await agent.run(start_url=site + "/dead.html")

    def dead_index(position: int) -> str | None:
        for element in decider.states[position]["elements"]:
            if element["label"] == "Dead button":
                return str(element["index"])
        return None

    assert len(decider.offered) >= 5
    for position in range(3):
        assert dead_index(position) in decider.offered[position]["click_target"]
    for position in range(3, len(decider.offered)):
        assert dead_index(position) not in decider.offered[position]["click_target"]

    counts = {record.element: record for record in agent.history if record.element == "Dead button"}
    assert counts, "the dead button was never clicked"
    assert any("nothing visible changed" in record.effect for record in agent.history)


@pytest.mark.slow
async def test_scrolling_leaves_the_menu_when_it_takes_over(browser, site):
    decider = ScriptedDecider("scroll")
    fallback = FakeFallback()
    agent = build_agent(browser, decider, fallback, goal="Read the long page", max_steps=12)
    await agent.run(start_url=site + "/tall.html")

    # The page has no controls, so once 6 of the last 8 steps were scrolls the menu holds nothing but waiting and going back.
    # No operation on an element is left, so the fallback decides those steps and the decision model is not asked.
    assert decider.picked[:6].count("SCROLL_DOWN") == 6, "the scripted decision model should have scrolled six times first"
    assert fallback.calls, "when scrolling left the menu nothing was left but waiting, so the fallback should have been asked"
    assert any("only operation left" in call["reason"] for call in fallback.calls)


@pytest.mark.slow
async def test_a_one_option_question_is_answered_without_asking(browser, site):
    decider = ScriptedDecider("single")
    agent = build_agent(browser, decider, FakeFallback(), goal="Press the only control", max_steps=1)
    result = await agent.run(start_url=site + "/single.html")

    for call in decider.calls:
        assert "click_target" not in call["keys"], "a choice with one option must not be sent to the model"
    assert result.steps[0].operation == "CLICK"
    assert "page text changed" in result.steps[0].effect
    assert "The one control was pressed" in result.steps[0].effect or "changed" in result.steps[0].effect


@pytest.mark.slow
async def test_the_fallback_steps_in_when_the_margin_is_low(browser, site):
    decider = ScriptedDecider("unsure")
    fallback = FakeFallback([{"action": "wait", "index": 0, "fact": "", "reason": "waiting"}])
    agent = build_agent(browser, decider, fallback, goal="Nothing useful", max_steps=3)
    result = await agent.run(start_url=site + "/index.html")

    assert result.fallback_calls == 3
    assert result.fallback_cost == pytest.approx(0.006, abs=1e-6)
    assert all(record.source == "fallback" for record in result.steps)
    assert "unsure" in fallback.calls[0]["reason"]


@pytest.mark.slow
async def test_a_done_claim_without_confirmation_stops_as_unverified(browser, site):
    decider = ScriptedDecider("claim-done-unproven")
    fallback = FakeFallback([{"action": "wait", "index": 0, "fact": "", "reason": "waiting"}])
    agent = build_agent(browser, decider, fallback, goal="Open the form and stop", max_steps=6)
    result = await agent.run(start_url=site + "/index.html")

    assert result.outcome == "unverified", result.summary()
    claims = [record for record in result.steps if record.operation == "DONE"]
    assert len(claims) == 2
    assert "no confirmation" in " ".join(result.notes)


@pytest.mark.slow
async def test_a_done_claim_is_accepted_when_the_decider_reads_the_new_text(browser, site):
    decider = ScriptedDecider("claim-done-proven")
    agent = build_agent(browser, decider, FakeFallback(), goal="Open the form and stop", max_steps=4)
    result = await agent.run(start_url=site + "/index.html")

    assert result.outcome == "verified", result.summary()
    assert "the decision model read the new text" in result.steps[-1].effect


@pytest.mark.slow
async def test_a_wait_that_changed_nothing_leaves_the_menu(browser, site):
    decider = ScriptedDecider("wait")
    agent = build_agent(browser, decider, FakeFallback(), goal="Wait for the page", max_steps=5)
    await agent.run(start_url=site + "/index.html")

    # Step 1 can wait. After a wait that changed nothing, WAIT is gone from the menu for a while.
    assert "WAIT" in decider.offered[0]["operation"]
    assert all("WAIT" not in (offered.get("operation") or []) for offered in decider.offered[1:4])
    assert agent.history[0].operation == "WAIT"
    assert agent.history[0].effect.startswith("nothing visible changed")


@pytest.mark.slow
async def test_a_page_with_no_controls_stops_as_blocked(browser, site):
    decider = ScriptedDecider("wait")
    agent = build_agent(browser, decider, FakeFallback(), goal="Do something", max_steps=5)
    result = await agent.run(start_url="about:blank")
    assert result.outcome == "blocked"
    assert any("no controls" in note for note in result.notes)


@pytest.mark.slow
async def test_writes_are_refused_until_allow_writes_is_on(browser, site):
    decider = ScriptedDecider("form")
    agent = build_agent(browser, decider, FakeFallback(), allow_writes=False, max_steps=6)
    result = await agent.run(start_url=site + "/form.html")

    for record in result.steps:
        assert record.operation not in ("TYPE_TEXT", "SELECT")
        assert record.fact is None
    observation = await browser.observe()
    for element in observation.elements:
        if element.label == "Your name":
            assert element.value == ""
        if element.label == "Send the form":
            assert element.label not in [step.element for step in result.steps]


@pytest.mark.slow
async def test_a_closed_tab_does_not_stop_the_run(browser, site):
    decider = ScriptedDecider("form")
    agent = build_agent(browser, decider, FakeFallback(), max_steps=2)
    result = await agent.run(start_url=site + "/index.html")
    assert result.steps[0].effect.startswith("opened a new tab")
    assert browser.page.url.endswith("/form.html")

    # Close the tab the run moved into. The agent must fall back to a tab it still has.
    await browser.page.close()
    observation = await browser.observe()
    assert browser.page.is_closed() is False
    assert observation.url.endswith("/index.html")


async def test_the_numbered_screenshot_carries_the_element_indexes(browser, site):
    agent = build_agent(browser, ScriptedDecider("single"), FakeFallback(), max_steps=1)
    await agent.run(start_url=site + "/single.html")
    assert agent.last_screenshot is not None
    assert agent.last_screenshot[:2] == b"\xff\xd8", "the numbered screenshot must be a JPEG"

    page_state = await browser.page.evaluate("() => document.getElementById('__mmjb_marks')")
    assert page_state is None, "the overlay must be removed after the screenshot"


@pytest.mark.slow
async def test_a_recorded_run_keeps_a_numbered_frame_and_a_line_for_every_step(browser, site, tmp_path):
    import json

    decider = ScriptedDecider("form")
    agent = build_agent(browser, decider, FakeFallback(), record_dir=str(tmp_path))
    result = await agent.run(start_url=site + "/index.html")

    assert result.outcome == "verified", result.summary()
    lines = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert len(lines) == len(result.steps)
    for line in lines:
        assert (tmp_path / line["frame"]).stat().st_size > 1000
    assert [line["n"] for line in lines] == sorted(line["n"] for line in lines)
    summary = json.loads((tmp_path / "result.json").read_text())
    assert summary["outcome"] == "verified"
    assert summary["steps"] == len(result.steps)

