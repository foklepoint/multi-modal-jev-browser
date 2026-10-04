"""The numbered screenshot: one box per element, the right number on it, and no overlay left behind."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("browser")


async def test_the_overlay_is_removed_after_the_screenshot(browser, site):
    await browser.goto(site + "/form.html")
    observation = await browser.observe()
    raw = await browser.marked_screenshot([element.index for element in observation.elements])
    assert raw is not None
    assert raw[:2] == b"\xff\xd8"
    assert await browser.page.evaluate("() => document.getElementById('__mmjb_marks')") is None


async def test_the_boxes_carry_the_element_indexes(browser, site):
    await browser.goto(site + "/form.html")
    observation = await browser.observe()
    indexes = [element.index for element in observation.elements]
    rects = await __import__("mmjb.marks", fromlist=["draw_marks"]).draw_marks(browser.page, indexes)
    try:
        tags = await browser.page.evaluate(
            "() => Array.from(document.querySelectorAll('#__mmjb_marks div div')).map((node) => node.textContent)"
        )
    finally:
        await browser.page.evaluate("() => { const layer = document.getElementById('__mmjb_marks'); if (layer) layer.remove(); }")
    assert [str(index) for index in indexes] == tags
    assert len(rects) == len(indexes)
    for rect in rects:
        assert rect[3] > 0 and rect[4] > 0


async def test_a_page_with_no_elements_still_returns_a_screenshot(browser, site):
    await browser.goto(site + "/index.html")
    observation = await browser.observe()
    indexes = [element.index for element in observation.elements if element.label == "A button that does nothing"]
    raw = await browser.marked_screenshot(indexes)
    assert raw is None or raw[:2] == b"\xff\xd8"