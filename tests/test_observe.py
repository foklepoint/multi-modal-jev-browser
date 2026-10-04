"""Reading a page: the elements in the viewport, the page text and the disabled controls."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("browser")


async def test_index_page_lists_links_with_indexes(browser, site):
    await browser.goto(site + "/index.html")
    observation = await browser.observe()

    labels = [element.label for element in observation.elements]
    assert any("new tab" in label for label in labels)
    assert any("one control" in label for label in labels)
    assert [element.index for element in observation.elements] == list(range(1, len(observation.elements) + 1))
    assert observation.title == "Test site home"
    assert "Test site" in observation.text
    assert observation.url.endswith("/index.html")


async def test_form_fields_carry_value_and_state(browser, site):
    await browser.goto(site + "/form.html")
    observation = await browser.observe()

    by_label = {}
    for element in observation.elements:
        by_label[element.label] = element

    name = by_label["Your name"]
    assert name.kind == "type"
    assert name.role == "textbox"
    assert name.required is True
    assert name.value == ""

    message = by_label["Your message"]
    assert message.kind == "type"

    category = by_label["Category"]
    assert category.kind == "select"
    assert "Tools" in category.options
    assert category.required is True

    terms = None
    for element in observation.elements:
        if "accept the terms" in element.label.lower():
            terms = element
    assert terms is not None
    assert terms.checked is False
    assert terms.kind == "click"

    submit = by_label["Send the form"]
    assert submit.kind == "click"
    assert submit.disabled is False


async def test_disabled_submit_is_listed_and_named(browser, site):
    await browser.goto(site + "/locked.html")
    observation = await browser.observe()

    assert "Send" in observation.disabled_controls
    send = None
    for element in observation.elements:
        if element.label == "Send":
            send = element
    assert send is not None
    assert send.disabled is True

    await browser.type_text(1, "a value")
    await browser.type_text(2, "another value")
    await browser.settle()
    after = await browser.observe()
    assert "Send" not in after.disabled_controls


async def test_tall_page_lists_only_the_viewport_and_names_the_rest(browser, site):
    await browser.goto(site + "/tall.html")
    observation = await browser.observe()

    labels = [element.label for element in observation.elements]
    assert not any("bottom" in label for label in labels)
    below = [item for item in observation.offscreen_controls if item["direction"] == "below"]
    assert any("bottom" in item["label"] for item in below)
    assert observation.scroll["page_height"] > observation.scroll["viewport_height"]

    await browser.scroll("bottom")
    await browser.settle()
    at_bottom = await browser.observe()
    assert any("bottom" in element.label for element in at_bottom.elements)
    assert at_bottom.scroll["y"] > 0


async def test_element_state_changes_when_a_field_is_filled(browser, site):
    await browser.goto(site + "/form.html")
    before = await browser.observe()
    await browser.type_text(1, "Ada Lovelace")
    await browser.settle()
    after = await browser.observe()

    assert before.fingerprint() != after.fingerprint()
    for element in after.elements:
        if element.label == "Your name":
            assert element.value == "Ada Lovelace"


async def test_listing_shows_the_numbers_and_the_offscreen_controls(browser, site):
    await browser.goto(site + "/index.html")
    observation = await browser.observe()
    listing = observation.listing()
    assert "[1]" in listing
    assert "URL: " in listing
    assert "Off screen:" not in listing or "Off screen:" in listing