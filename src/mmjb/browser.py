"""The Playwright wrapper: open a page, look at it, click it, type in it, and follow tabs.

Everything the agent does to a browser goes through this module, so there is one place that knows how
a tab is followed, how a click is aimed and what an action actually changed.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from . import marks
from .observe import Observation, observe

DEFAULT_VIEWPORT = {"width": 1120, "height": 780}
SETTLE_MS = 700
LOAD_TIMEOUT_MS = 8000
SCROLL_STEP = 700

SCROLL_JS = r"""
((delta) => {
  window.scrollBy({ top: delta, left: 0, behavior: 'instant' });
  return Math.round(window.scrollY);
})
"""

SCROLL_TO_JS = r"""
((top) => {
  window.scrollTo({ top: top, left: 0, behavior: 'instant' });
  return Math.round(window.scrollY);
})
"""


class BrowserError(RuntimeError):
    """Something about the browser stopped the agent. The message says what to do next."""


@dataclass
class Snapshot:
    """The parts of a page that tell you whether the last action changed anything."""

    url: str
    text: str
    control_state: str
    fingerprint: str

    @classmethod
    def of(cls, observation: Observation) -> "Snapshot":
        control_state = json.dumps(observation.control_state, sort_keys=True)
        return cls(
            url=observation.url,
            text=observation.text,
            control_state=control_state,
            fingerprint=observation.fingerprint(),
        )


def describe_effect(before: Snapshot, after: Snapshot, new_tab_url: str | None) -> str:
    """What the last action did, in words, for the next decision and the fallback model."""
    if new_tab_url:
        return "opened a new tab and moved to it: " + new_tab_url[:120]
    if after.url != before.url:
        return "url changed to " + after.url[:120]
    if after.text != before.text:
        change = len(after.text) - len(before.text)
        if change >= 0:
            return f"same url; the page text changed by +{change} characters"
        return f"same url; the page text changed by {change} characters"
    if after.control_state != before.control_state:
        return "same url and text; a field value or a control state changed"
    return "nothing visible changed: same url, same text, no new tab"


def nothing_changed(effect: str) -> bool:
    """True when an action left no trace at all."""
    return effect.startswith("nothing visible changed")


class Browser:
    """One browser context and the tab the agent works in."""

    def __init__(
        self,
        cdp_url: str | None = None,
        chrome: bool = False,
        headless: bool = True,
        viewport: dict[str, int] | None = None,
    ) -> None:
        self.cdp_url = cdp_url or None
        self.chrome = chrome
        self.headless = headless
        self.viewport = dict(viewport or DEFAULT_VIEWPORT)
        self.playwright: Any = None
        self.browser: Any = None
        self.context: Any = None
        self.page: Any = None
        self.tracked: list[Any] = []
        self.seen: set[int] = set()
        self._started = False

    # --- lifecycle ---------------------------------------------------------------------------

    async def start(self) -> "Browser":
        """Open a browser, or attach to the one MMJB_CDP_URL points at."""
        from playwright.async_api import async_playwright

        self.playwright = await async_playwright().start()
        if self.cdp_url:
            self.browser = await self.playwright.chromium.connect_over_cdp(self.cdp_url)
            contexts = self.browser.contexts
            if contexts:
                self.context = contexts[0]
            else:
                self.context = await self.browser.new_context(viewport=self.viewport)
        else:
            self.browser = await self._launch(self.playwright.chromium)
            self.context = await self.browser.new_context(viewport=self.viewport)
        self.context.on("page", self._on_page)
        pages = [page for page in self.context.pages if not page.is_closed()]
        if pages:
            self.page = pages[-1]
        else:
            self.page = await self.context.new_page()
        self.tracked = [self.page]
        self.seen = {id(page) for page in self.context.pages}
        self._started = True
        return self

    async def _launch(self, chromium: Any) -> Any:
        if self.chrome:
            try:
                return await chromium.launch(headless=self.headless, channel="chrome")
            except Exception as error:
                raise BrowserError(
                    "Chrome could not start (" + str(error)[:160] + "). Unset MMJB_CHROME to use Playwright's own Chromium, "
                    "or install it with: playwright install chromium"
                ) from error
        try:
            return await chromium.launch(headless=self.headless)
        except Exception as error:
            raise BrowserError(
                "Playwright's Chromium could not start (" + str(error)[:160] + "). Install it with: "
                "playwright install chromium, or set MMJB_CHROME=1 to use your installed Google Chrome"
            ) from error

    async def stop(self) -> None:
        """Close everything this object opened."""
        try:
            if self.browser is not None and not self.cdp_url:
                await self.browser.close()
        except Exception:
            pass
        try:
            if self.playwright is not None:
                await self.playwright.stop()
        except Exception:
            pass
        self._started = False

    def _on_page(self, page: Any) -> None:
        # A new tab is recorded here and moved to later, once the click that opened it has settled.
        self.seen.add(id(page))

    async def __aenter__(self) -> "Browser":
        return await self.start()

    async def __aexit__(self, kind: Any, value: Any, traceback: Any) -> None:
        await self.stop()

    # --- tabs ---------------------------------------------------------------------------------

    def _open_pages(self) -> list[Any]:
        pages = []
        for page in self.context.pages:
            if not page.is_closed():
                pages.append(page)
        return pages

    async def adopt_new_tab(self) -> str | None:
        """Move to a tab that one of this run's tabs just opened, or report None."""
        try:
            ours = set(id(page) for page in self.tracked)
            fresh = []
            for page in self._open_pages():
                if id(page) in ours:
                    continue
                try:
                    opener = await page.opener()
                except Exception:
                    opener = None
                if opener is None or id(opener) in ours:
                    fresh.append(page)
            if not fresh:
                return None
            target = fresh[-1]
            self.page = target
            self.tracked.append(target)
            try:
                await target.wait_for_load_state("load", timeout=LOAD_TIMEOUT_MS)
            except Exception:
                pass
            return target.url
        except Exception:
            return None

    async def recover_closed_tab(self) -> bool:
        """A tab that closed leaves the agent nowhere; go back to the newest tab of the run that is open."""
        if self.page is not None and not self.page.is_closed():
            return False
        alive = self._open_pages()
        alive_ids = set(id(page) for page in alive)
        for page in reversed(self.tracked):
            if not page.is_closed() and id(page) in alive_ids:
                self.page = page
                return True
        if alive:
            self.page = alive[-1]
            self.tracked.append(self.page)
            return True
        self.page = await self.context.new_page()
        self.tracked.append(self.page)
        return True

    async def back_to_first_tab(self) -> str | None:
        """The main tab of the run, for an agent that has wandered into a popup."""
        if self.tracked and not self.tracked[0].is_closed():
            self.page = self.tracked[0]
            return self.page.url
        return None

    async def settle(self, pause_ms: int = SETTLE_MS) -> None:
        """Wait for the page to finish loading, then a short pause for scripts and animations."""
        try:
            await self.page.wait_for_load_state("load", timeout=LOAD_TIMEOUT_MS)
        except Exception:
            pass
        try:
            await self.page.wait_for_timeout(pause_ms)
        except Exception:
            await asyncio.sleep(pause_ms / 1000.0)

    # --- reading ------------------------------------------------------------------------------

    async def goto(self, url: str) -> None:
        """Open a URL in the working tab."""
        if not url:
            return
        try:
            await self.page.goto(url, wait_until="load", timeout=30000)
        except Exception as error:
            raise BrowserError(f"{url} did not load ({str(error)[:160]}). Check the URL and that the site is up.") from error
        await self.settle()

    async def observe(self) -> Observation:
        """Read the working tab right now."""
        await self.recover_closed_tab()
        return await observe(self.page)

    async def marked_screenshot(self, indices: list[int]) -> bytes | None:
        """A JPEG of the viewport with a numbered box on each of these elements."""
        raw = await marks.marked_screenshot(self.page, indices)
        if raw is None:
            return None
        return raw

    async def current_url(self) -> str:
        try:
            return self.page.url
        except Exception:
            return ""

    async def title(self) -> str:
        try:
            return await self.page.title()
        except Exception:
            return ""

    # --- acting -------------------------------------------------------------------------------

    async def _element(self, index: int) -> Any:
        handle = await self.page.evaluate_handle("(index) => (window.__mmjb && window.__mmjb.nodes) ? window.__mmjb.nodes[index] : null", index)
        element = handle.as_element()
        if element is None:
            raise BrowserError(
                f"element {index} is not on the page any more. Call look() again for the current element numbers."
            )
        return element

    async def click(self, index: int) -> None:
        """A real mouse click at the centre of the element, after scrolling it into view."""
        element = await self._element(index)
        try:
            await element.scroll_into_view_if_needed(timeout=4000)
        except Exception:
            pass
        await self.page.wait_for_timeout(120)
        box = await element.bounding_box()
        if not box:
            raise BrowserError(f"element {index} has no size on screen, so it cannot be clicked. Call look() again.")
        x = box["x"] + box["width"] / 2.0
        y = box["y"] + box["height"] / 2.0
        await self.page.mouse.click(x, y)

    FOCUS_JS = r"""
    ((index) => {
      const element = (window.__mmjb && window.__mmjb.nodes) ? window.__mmjb.nodes[index] : null;
      if (!element) return false;
      element.scrollIntoView({ block: 'center', inline: 'center' });
      element.focus();
      if (typeof element.select === 'function') {
        element.select();
      } else {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(element);
        selection.removeAllRanges();
        selection.addRange(range);
      }
      return true;
    })
    """

    async def type_text(self, index: int, text: str) -> None:
        """Replace whatever is in the field with this text: focus, select all, insert."""
        focused = await self.page.evaluate(self.FOCUS_JS, index)
        if not focused:
            raise BrowserError(f"element {index} is not on the page any more. Call look() again for the current element numbers.")
        await self.page.keyboard.insert_text(text)

    async def select_option(self, index: int, option: str) -> None:
        """Pick a dropdown option by its visible text, or by its value when the text does not match."""
        element = await self._element(index)
        try:
            await element.select_option(label=option)
        except Exception:
            try:
                await element.select_option(value=option)
            except Exception as error:
                raise BrowserError(
                    f"'{option}' is not an option of element {index} ({str(error)[:140]}). Call look() to read the options."
                ) from error

    async def upload(self, index: int, path: str) -> None:
        """Hand a file to a file input without opening the operating system's dialog."""
        element = await self._element(index)
        try:
            await element.set_input_files(path)
        except Exception as error:
            raise BrowserError(f"could not upload {path} to element {index} ({str(error)[:140]}).") from error

    async def scroll(self, direction: str, amount: int = SCROLL_STEP) -> int:
        """Scroll the page. Direction is down, up, top or bottom."""
        step = direction.lower()
        if step == "top":
            return await self.page.evaluate(SCROLL_TO_JS, 0)
        if step == "bottom":
            height = await self.page.evaluate("() => Math.round(document.documentElement.scrollHeight)")
            return await self.page.evaluate(SCROLL_TO_JS, height)
        if step == "up":
            return await self.page.evaluate(SCROLL_JS, -amount)
        if step == "down":
            return await self.page.evaluate(SCROLL_JS, amount)
        raise BrowserError(f"'{direction}' is not a scroll direction. Use down, up, top or bottom.")

    async def scroll_to_label(self, label: str) -> bool:
        """Scroll a control the observer listed but could not see, matched by its label."""
        found = await self.page.evaluate(
            r"""
            ((label) => {
              const clean = (value) => (value || '').replace(/\s+/g, ' ').trim();
              const text = (element) => {
                let out = clean(element.getAttribute('aria-label'));
                if (out) return out;
                if (element.labels && element.labels.length) {
                  out = clean(Array.from(element.labels).map((node) => clean(node.innerText)).join(' '));
                  if (out) return out;
                }
                return clean(element.innerText);
              };
              const candidates = Array.from(document.querySelectorAll('a[href], button, input, textarea, select, summary'));
              for (const element of candidates) {
                if (text(element) === label) {
                  element.scrollIntoView({ block: 'center', inline: 'center' });
                  return true;
                }
              }
              return false;
            })
            """,
            label,
        )
        return bool(found)

    async def press(self, key: str) -> None:
        """Press a key such as Enter, Tab or Escape."""
        await self.page.keyboard.press(key)

    async def go_back(self) -> bool:
        """Go back one entry in this tab's history."""
        try:
            response = await self.page.go_back(wait_until="load", timeout=15000)
        except Exception:
            return False
        return response is not None

    async def can_scroll(self) -> dict[str, bool]:
        """Whether the page has anywhere to scroll up or down from here."""
        return await self.page.evaluate(
            r"""
            (() => {
              const top = Math.round(window.scrollY);
              const bottom = Math.round(window.scrollY + window.innerHeight);
              const height = Math.round(document.documentElement.scrollHeight);
              return { up: top > 4, down: bottom < height - 4 };
            })()
            """
        )