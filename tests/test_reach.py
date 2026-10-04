"""Things the element list cannot name or that live in another document: embedded frames, points, drags and keys."""
from __future__ import annotations

import pytest
from support import FakeFallback, choose, noul, options_of

from mmjb.agent import Agent
from mmjb.fallback import normalise_action

pytestmark = pytest.mark.usefixtures("browser")


class UnsureDecider:
    """A decision model that is never sure, so every step goes to the fallback."""

    name = "unsure"

    async def decide(self, state, questions, images):
        answers = {}
        for key in questions:
            if key in ("goal_done", "no_move"):
                answers[key] = noul(0.0)
                continue
            options = options_of(questions, key)
            if not options:
                continue
            answers[key] = choose(options[0], 1.0, {option: 1.0 for option in options[1:]})
        return answers


def point_agent(browser, site, page, actions, allow_writes=True):
    fallback = FakeFallback(actions=actions)
    agent = Agent(
        goal="Do what the page needs.",
        start_url=site + page,
        decider=UnsureDecider(),
        fallback=fallback,
        browser=browser,
        allow_writes=allow_writes,
        max_steps=1,
        budget_seconds=60.0,
    )
    return agent, fallback


async def test_controls_inside_a_frame_are_listed_numbered_and_usable(browser, site):
    await browser.goto(site + "/frame.html")
    observation = await browser.observe()

    inside = [element for element in observation.elements if element.frame]
    labels = [element.label for element in inside]
    assert "Inner name" in labels
    assert "Save inner" in labels
    assert [element.index for element in observation.elements] == list(range(1, len(observation.elements) + 1))
    assert "Not saved" in observation.text

    field = next(element for element in inside if element.label == "Inner name")
    button = next(element for element in inside if element.label == "Save inner")
    await browser.type_text(field.index, "Ada")
    await browser.click(button.index)
    after = await browser.observe()
    assert "Saved: Ada" in after.text

    shot = await browser.marked_screenshot([element.index for element in after.elements])
    assert shot is not None and len(shot) > 1000


async def test_click_at_lands_on_a_canvas(browser, site):
    await browser.goto(site + "/canvas.html")
    await browser.click_at(220, 180)
    observation = await browser.observe()
    assert "Canvas clicked at 220,180" in observation.text


async def test_click_at_refuses_a_point_outside_the_viewport(browser, site):
    from mmjb.browser import BrowserError

    await browser.goto(site + "/canvas.html")
    with pytest.raises(BrowserError):
        await browser.click_at(5000, 10)


async def test_drag_moves_a_box_onto_a_target(browser, site):
    await browser.goto(site + "/drag.html")
    assert "Not dropped" in (await browser.observe()).text
    await browser.drag(90, 175, 500, 180)
    assert "Dropped in the zone" in (await browser.observe()).text


async def test_keys_drive_a_dropdown_that_only_opens_from_the_keyboard(browser, site):
    await browser.goto(site + "/keys.html")
    observation = await browser.observe()
    menu = next(element for element in observation.elements if element.label == "Choose a plan")
    await browser.click(menu.index)
    await browser.press("ArrowDown")
    await browser.press("ArrowDown")
    await browser.press("Enter")
    assert "Plan: Team" in (await browser.observe()).text


async def test_the_fallback_can_click_at_a_point(browser, site):
    agent, fallback = point_agent(browser, site, "/canvas.html", [{"action": "click_at", "x": 220, "y": 180, "reason": "the canvas"}])
    result = await agent.run()
    first = result.steps[0]
    assert first.source == "fallback"
    assert first.operation == "CLICK_AT"
    assert "point (220, 180)" in first.element
    assert "page text changed" in first.effect
    assert fallback.calls[0]["images"] >= 1, "the fallback should at least get the plain screenshot"


async def test_the_fallback_can_drag(browser, site):
    agent, _fallback = point_agent(
        browser, site, "/drag.html", [{"action": "drag", "x": 90, "y": 175, "x2": 500, "y2": 180, "reason": "drop it"}]
    )
    result = await agent.run()
    assert result.steps[0].operation == "DRAG"
    assert "page text changed" in result.steps[0].effect


async def test_the_fallback_can_press_a_key(browser, site):
    agent, _fallback = point_agent(browser, site, "/keys.html", [{"action": "press", "key": "Tab", "reason": "move on"}])
    result = await agent.run()
    assert result.steps[0].operation == "PRESS"
    assert result.steps[0].element == "key Tab"
    assert _fallback.calls[0]["images"] == 2, "a page with controls gives the fallback the numbered screenshot and the plain one"


async def test_a_key_the_fallback_names_badly_is_not_pressed(browser, site):
    agent, _fallback = point_agent(browser, site, "/keys.html", [{"action": "press", "key": "Enter; rm -rf", "reason": "bad"}])
    result = await agent.run()
    assert result.outcome == "blocked"


async def test_points_drags_and_keys_are_refused_while_writes_are_off(browser, site):
    agent, _fallback = point_agent(
        browser, site, "/canvas.html", [{"action": "click_at", "x": 220, "y": 180, "reason": "the canvas"}], allow_writes=False
    )
    result = await agent.run()
    assert result.steps[0].effect.startswith("refused:")
    assert "Canvas clicked" not in (await browser.observe()).text


def test_the_fallback_reply_keeps_coordinates_and_the_key():
    action = normalise_action({"action": "click_at", "x": "10", "y": 20.5})
    assert action["x"] == 10.0 and action["y"] == 20.5
    assert normalise_action({"action": "press", "key": "ArrowDown"})["key"] == "ArrowDown"
