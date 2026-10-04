"""The MCP server. One browser per session, tools that an AI agent can drive.

Nothing here ever writes to stdout: stdout is the protocol channel.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import io
import json
import logging
import os
import sys
from typing import Any

from .agent import Agent
from .browser import Browser

LOG = logging.getLogger("mmjb")

INSTRUCTIONS = (
    "browse runs a whole goal on its own: give it a goal, a start_url and the facts to type, and it "
    "fills forms, clicks buttons and follows popups. Use look, click, type, select, scroll, goto, back "
    "and press when you want to drive the page yourself. Every element number comes from look()."
)

# mcp 2.x renamed FastMCP to MCPServer. Both work; the tool decorators are the same.
try:  # pragma: no cover - depends on the installed mcp major version
    from mcp.server.mcpserver import Image, MCPServer as _ServerClass
except ImportError:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP as _ServerClass
    from mcp.server.fastmcp import Image


def _version() -> str:
    from . import __version__

    return __version__


server = _ServerClass("mmjb", version=_version(), instructions=INSTRUCTIONS)

# These tools return text and images, not JSON data, so structured output is turned off where the
# installed mcp supports the switch. Without it the SDK tries to serialise the Image helper.
_TOOL_OPTIONS = set(inspect.signature(server.tool).parameters)

SESSION: dict[str, Any] = {"browser": None, "goal": "", "allow_writes": False}
LOCK = asyncio.Lock()


def tool() -> Any:
    """Register a tool that returns content blocks rather than JSON data."""
    options: dict[str, Any] = {}
    if "structured_output" in _TOOL_OPTIONS:
        options["structured_output"] = False
    return server.tool(**options)


def safe(fn: Any) -> Any:
    """Keep stdout clean for the protocol, and hand the agent a message instead of a stack trace."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                return await fn(*args, **kwargs)
            except Exception as error:
                LOG.warning("%s failed: %s", fn.__name__, error)
                return "error: " + str(error)[:400]

    return wrapper


async def _browser() -> Browser:
    """Open the session's browser on first use and keep it open for every later call."""
    if SESSION.get("browser") is None:
        cdp = os.environ.get("MMJB_CDP_URL") or ""
        chrome = os.environ.get("MMJB_CHROME") == "1"
        headless = os.environ.get("MMJB_HEADFUL") != "1"
        browser = Browser(cdp_url=cdp, chrome=chrome, headless=headless)
        await browser.start()
        SESSION["browser"] = browser
    return SESSION["browser"]


def _agent(goal: str = "") -> Agent:
    """A fresh agent that shares this session's browser, so one goal does not inherit another's steps."""
    browser = SESSION.get("browser")
    return Agent(
        goal=goal or SESSION.get("goal", ""),
        browser=browser,
        allow_writes=bool(SESSION.get("allow_writes", False)),
        log=lambda line: LOG.info(line),
    )


def _text_and_image(text: str, screenshot: bytes | None) -> list[Any]:
    """A text block and, when there is one, the numbered screenshot."""
    parts: list[Any] = [text]
    if screenshot:
        parts.append(Image(data=screenshot, format="jpeg"))
    return parts


@tool()
@safe
async def browse(
    goal: str,
    start_url: str = "",
    facts: dict[str, str] | None = None,
    allow_writes: bool = False,
    max_steps: int = 40,
) -> list[Any]:
    """Run the whole loop until the goal is met or a limit is reached, and return the result with the last numbered screenshot."""
    async with LOCK:
        SESSION["browser"] = await _browser()
        SESSION["goal"] = goal
        SESSION["allow_writes"] = allow_writes
        agent = _agent(goal)
        result = await agent.run(
            start_url=start_url,
            facts=dict(facts or {}),
            allow_writes=allow_writes,
            max_steps=max_steps,
        )
        return _text_and_image(json.dumps(result.as_dict(), indent=2), result.screenshot)


@tool()
@safe
async def look() -> list[Any]:
    """Show the current page as a numbered screenshot and a list of the numbered elements on it."""
    async with LOCK:
        SESSION["browser"] = await _browser()
        agent = _agent()
        observation, screenshot = await agent.look()
        return _text_and_image(observation.listing(), screenshot)


@tool()
@safe
async def click(index: int, allow_writes: bool = False) -> str:
    """Click the element with this index, using the numbers from the last look()."""
    async with LOCK:
        SESSION["browser"] = await _browser()
        SESSION["allow_writes"] = SESSION.get("allow_writes", False) or allow_writes
        agent = _agent()
        effect, _observation = await agent.simple_action("click", index=index)
        return effect


@tool()
@safe
async def type(index: int, text: str, allow_writes: bool = False) -> str:
    """Type text into the element with this index, using the numbers from the last look()."""
    async with LOCK:
        SESSION["browser"] = await _browser()
        SESSION["allow_writes"] = SESSION.get("allow_writes", False) or allow_writes
        agent = _agent()
        effect, _observation = await agent.simple_action("type", index=index, text=text)
        return effect


@tool()
@safe
async def select(index: int, option: str, allow_writes: bool = False) -> str:
    """Choose an option in the dropdown with this index, using the numbers from the last look()."""
    async with LOCK:
        SESSION["browser"] = await _browser()
        SESSION["allow_writes"] = SESSION.get("allow_writes", False) or allow_writes
        agent = _agent()
        effect, _observation = await agent.simple_action("select", index=index, text=option)
        return effect


@tool()
@safe
async def scroll(direction: str = "down") -> str:
    """Scroll the page; direction is down, up, top or bottom."""
    async with LOCK:
        SESSION["browser"] = await _browser()
        agent = _agent()
        effect, _observation = await agent.simple_action("scroll", text=direction)
        return effect


@tool()
@safe
async def goto(url: str) -> str:
    """Open a URL in the browser tab the agent is using."""
    async with LOCK:
        SESSION["browser"] = await _browser()
        agent = _agent()
        effect, _observation = await agent.simple_action("goto", url=url)
        return effect


@tool()
@safe
async def back() -> str:
    """Go back to the previous page in the current tab."""
    async with LOCK:
        SESSION["browser"] = await _browser()
        agent = _agent()
        effect, _observation = await agent.simple_action("back")
        return effect


@tool()
@safe
async def click_at(x: float, y: float, allow_writes: bool = False) -> str:
    """Click a point of the viewport, in the pixels of the screenshot from look(). For canvases, challenge widgets and anything with no numbered element."""
    async with LOCK:
        SESSION["browser"] = await _browser()
        SESSION["allow_writes"] = SESSION.get("allow_writes", False) or allow_writes
        agent = _agent()
        effect, _observation = await agent.simple_action("click_at", x=x, y=y)
        return effect


@tool()
@safe
async def drag(x: float, y: float, to_x: float, to_y: float, allow_writes: bool = False) -> str:
    """Press at one point of the viewport, move to another and release: sliders, sortable lists, drag and drop."""
    async with LOCK:
        SESSION["browser"] = await _browser()
        SESSION["allow_writes"] = SESSION.get("allow_writes", False) or allow_writes
        agent = _agent()
        effect, _observation = await agent.simple_action("drag", x=x, y=y, x2=to_x, y2=to_y)
        return effect


@tool()
@safe
async def press(key: str, allow_writes: bool = False) -> str:
    """Press a key such as Enter, ArrowDown, Tab or Escape on the current page, for menus and dropdowns that open from the keyboard."""
    async with LOCK:
        SESSION["browser"] = await _browser()
        SESSION["allow_writes"] = SESSION.get("allow_writes", False) or allow_writes
        agent = _agent()
        effect, _observation = await agent.simple_action("press", text=key)
        return effect


@tool()
@safe
async def status() -> dict[str, Any]:
    """Report the current URL, the configured decision and fallback models, and how much work has been done."""
    async with LOCK:
        if SESSION.get("browser") is not None:
            SESSION["browser"] = await _browser()
        agent = _agent()
        report = await agent.status()
        report["allow_writes"] = SESSION.get("allow_writes", False)
        return report


@tool()
@safe
async def doctor() -> dict[str, Any]:
    """Check Python, the browser, the decision models and the fallback, and say what to fix."""
    from .cli import collect_doctor_checks, run_doctor_text

    checks, healthy = await collect_doctor_checks()
    buffer = io.StringIO()
    await run_doctor_text(buffer, checks, healthy)
    return {
        "healthy": healthy,
        "report": buffer.getvalue().splitlines(),
        "checks": [{"name": check.name, "state": check.state, "detail": check.detail, "fix": check.fix} for check in checks],
    }


def main() -> None:
    """Entry point for the mmjb-mcp console script. Speaks MCP over stdio."""
    logging.basicConfig(level=os.environ.get("MMJB_LOG_LEVEL", "WARNING"), stream=sys.stderr)
    server.run()


if __name__ == "__main__":
    main()