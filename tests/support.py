"""Shared test helpers: the local site, a browser fixture and a deterministic decision model.

Nothing here touches the network. The ScriptedDecider is rule based so the real step loop can be
driven end to end without an API key.
"""
from __future__ import annotations

import functools
import http.server
import os
import socketserver
import threading
from typing import Any

import pytest

SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
VIEWPORT = {"width": 1100, "height": 720}


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the test site without logging every request."""

    def log_message(self, format: str, *args: Any) -> None:
        return


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


@pytest.fixture(scope="session")
def site() -> str:
    """Serve tests/site on a free port and yield its base URL."""
    handler = functools.partial(_QuietHandler, directory=SITE_DIR)
    server = _Server(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _chromium_kind() -> str:
    """Playwright's own Chromium when it is installed, else the system Chrome."""
    if os.environ.get("MMJB_CHROME") == "1":
        return "chrome"
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as play:
            if os.path.exists(play.chromium.executable_path):
                return "chromium"
    except Exception:
        pass
    return "chrome"


@pytest.fixture(scope="session")
def browser_kwargs() -> dict[str, Any]:
    """Launch arguments for the browser, whichever build is available on this machine."""
    return {"chrome": _chromium_kind() == "chrome", "headless": True, "viewport": VIEWPORT}


@pytest.fixture
async def browser(browser_kwargs: dict[str, Any]):
    """A real browser for one test, closed afterwards."""
    from mmjb.browser import Browser

    instance = Browser(**browser_kwargs)
    await instance.start()
    try:
        yield instance
    finally:
        await instance.stop()


def choose(option: str, score: float, rest: dict[str, float] | None = None) -> dict[str, Any]:
    """A choice answer with a probability for every option."""
    probabilities = {option: score}
    if rest:
        for key, value in rest.items():
            probabilities[key] = value
    total = sum(probabilities.values())
    if total <= 0:
        total = 1.0
    scaled = {}
    for key, value in probabilities.items():
        scaled[key] = round(value / total, 4)
    return {"type": "choice", "choice": option, "probabilities": scaled, "confidence": round(max(scaled.values()), 4)}


def noul(probability: float) -> dict[str, Any]:
    """A yes/no answer."""
    return {"type": "noul", "noul": probability}


def options_of(questions: dict[str, Any], key: str) -> list[str]:
    """The option keys of one question, whether it is a Question object or a plain dict."""
    question = questions.get(key)
    if question is None:
        return []
    options = getattr(question, "options", None)
    if options is None and isinstance(question, dict):
        options = question.get("options") or question.get("criteria")
    return list(options or [])


def element_of(state: dict[str, Any], kind: str, label_contains: str = "") -> dict[str, Any] | None:
    """The first element of a kind whose label contains this text."""
    for element in state.get("elements") or []:
        if element.get("kind") != kind:
            continue
        if label_contains and label_contains.lower() not in str(element.get("label", "")).lower():
            continue
        return element
    return None


class ScriptedDecider:
    """A decision model with no network behind it, driven by simple rules on the page state.

    It records what it was asked and what it was offered, so a test can check both.
    """

    name = "scripted"

    def __init__(self, mode: str = "form") -> None:
        self.mode = mode
        self.calls: list[dict[str, Any]] = []
        self.offered: list[dict[str, list[str]]] = []
        self.states: list[dict[str, Any]] = []
        self.picked: list[str] = []

    def offered_operation_menus(self) -> list[list[str] | None]:
        """The operation menu of every call, in order. One entry per call, so it lines up with picks.

        A call whose operation question had a single option never sees it, because the agent answers
        that locally. The entry is None in that case.
        """
        menus: list[list[str] | None] = []
        for offered in self.offered:
            menus.append(offered.get("operation"))
        return menus

    # --- helpers used by the rules below ----------------------------------------------------

    def _operation(self, questions: dict[str, Any], pick: str, score: float = 0.9) -> dict[str, Any]:
        options = options_of(questions, "operation")
        if pick not in options:
            pick = options[0] if options else pick
        rest = {}
        for option in options:
            if option != pick:
                rest[option] = 0.02
        return choose(pick, score, rest)

    def _target(self, questions: dict[str, Any], key: str, index: int) -> dict[str, Any]:
        options = options_of(questions, key)
        pick = str(index)
        if pick not in options:
            pick = options[0] if options else pick
        rest = {}
        for option in options:
            if option != pick:
                rest[option] = 0.01
        return choose(pick, 0.95, rest)

    # --- the protocol --------------------------------------------------------------------------

    async def decide(self, state: dict[str, Any], questions: dict[str, Any], images: list[str]) -> dict[str, Any]:
        """Answer this step's questions from the state, with no network call."""
        self.calls.append({"keys": list(questions.keys()), "images": len(images)})
        self.offered.append({key: options_of(questions, key) for key in questions})
        self.states.append({"elements": state.get("elements") or [], "url": (state.get("page") or {}).get("url", "")})
        if "confirm_new_text" in questions:
            probability = 0.2 if self.mode == "claim-done-unproven" else 0.95
            return {"confirm_new_text": noul(probability)}
        text = str((state.get("page") or {}).get("text") or "")
        if "Thanks" in text and "review" in text:
            self.picked.append("DONE")
            return {
                "goal_done": noul(0.99),
                "no_move": noul(0.02),
                "operation": self._operation(questions, "WAIT", 0.6),
                "fact_for": choose("none", 1.0, {}),
            }
        answers: dict[str, Any] = {"no_move": noul(0.03)}
        if self.mode == "form":
            answers.update(self._fill_form(state, questions))
        elif self.mode == "dead":
            answers.update(self._click_dead(state, questions))
        elif self.mode == "scroll":
            answers.update(self._scroll(questions))
        elif self.mode == "wait":
            answers.update(self._wait(questions))
        elif self.mode == "unsure":
            answers.update(self._unsure(state, questions))
        elif self.mode == "single":
            answers.update(self._single(state, questions))
        elif self.mode in ("claim-done-proven", "claim-done-unproven"):
            answers.update(self._claim_done(state, questions))
        else:
            answers["goal_done"] = noul(0.01)
            answers["operation"] = self._operation(questions, "WAIT", 0.6)
        if "fact_for" in questions:
            answers.setdefault("fact_for", choose("none", 0.5, self._other_facts(questions)))
        if "option_for" in questions:
            answers.setdefault("option_for", choose("none", 1.0, {}))
        # offered and picked stay the same length: a call with no operation question (the
        # confirmation check, or a one-option choice answered locally) records an empty pick.
        picked = answers.get("operation")
        if "operation" in questions and isinstance(picked, dict):
            self.picked.append(str(picked.get("choice") or ""))
        else:
            self.picked.append("")
        return answers

    def _other_facts(self, questions: dict[str, Any]) -> dict[str, float]:
        options = options_of(questions, "fact_for")
        return {option: 0.05 for option in options if option != "none"}

    def _other_options(self, questions: dict[str, Any]) -> dict[str, float]:
        options = options_of(questions, "option_for")
        return {option: 0.05 for option in options if option != "none"}

    def _fill_form(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Open the form, fill it one field at a time, tick the box, then submit."""
        elements = state.get("elements") or []
        answers: dict[str, Any] = {"goal_done": noul(0.02)}

        for element in elements:
            label = str(element.get("label", "")).lower()
            if element.get("kind") == "click" and "new tab" in label:
                answers["operation"] = self._operation(questions, "CLICK", 0.95)
                answers["click_target"] = self._target(questions, "click_target", int(element["index"]))
                answers["fact_for"] = choose("none", 1.0, {})
                return answers

        for element in elements:
            if element.get("kind") != "type" or element.get("value"):
                continue
            label = str(element.get("label", "")).lower()
            if "message" in label:
                fact = "message"
            elif "name" in label:
                fact = "name"
            else:
                continue
            answers["operation"] = self._operation(questions, "TYPE_TEXT", 0.9)
            answers["type_text_target"] = self._target(questions, "type_text_target", int(element["index"]))
            answers["fact_for"] = choose(fact, 0.99, self._other_facts(questions))
            return answers

        for element in elements:
            if element.get("kind") == "select" and not element.get("value"):
                answers["operation"] = self._operation(questions, "SELECT", 0.9)
                answers["select_target"] = self._target(questions, "select_target", int(element["index"]))
                answers["option_for"] = choose("Tools", 0.99, self._other_options(questions))
                answers["fact_for"] = choose("none", 1.0, {})
                return answers

        for element in elements:
            if element.get("kind") != "click":
                continue
            if "accept the terms" in str(element.get("label", "")).lower() and not element.get("checked"):
                answers["operation"] = self._operation(questions, "CLICK", 0.9)
                answers["click_target"] = self._target(questions, "click_target", int(element["index"]))
                answers["fact_for"] = choose("none", 1.0, {})
                return answers

        for element in elements:
            if element.get("kind") == "click" and "send" in str(element.get("label", "")).lower():
                answers["operation"] = self._operation(questions, "CLICK", 0.95)
                answers["click_target"] = self._target(questions, "click_target", int(element["index"]))
                answers["fact_for"] = choose("none", 1.0, {})
                return answers

        answers["operation"] = self._operation(questions, "WAIT", 0.6)
        return answers

    def _claim_done(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Open the form, then claim the goal is done with nothing on the page to prove it."""
        url = str((state.get("page") or {}).get("url") or "")
        elements = state.get("elements") or []
        answers: dict[str, Any] = {"no_move": noul(0.02), "fact_for": choose("none", 1.0, {})}
        if "form.html" not in url:
            answers["goal_done"] = noul(0.02)
            for element in elements:
                if element.get("kind") == "click" and "new tab" in str(element.get("label", "")).lower():
                    answers["operation"] = self._operation(questions, "CLICK", 0.95)
                    answers["click_target"] = self._target(questions, "click_target", int(element["index"]))
                    return answers
            answers["operation"] = self._operation(questions, "WAIT", 0.6)
            return answers
        answers["goal_done"] = noul(0.99)
        answers["operation"] = self._operation(questions, "WAIT", 0.6)
        return answers

    def _click_dead(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Always click the first control, whatever it is."""
        answers: dict[str, Any] = {"goal_done": noul(0.02)}
        target = element_of(state, "click")
        if target is None:
            answers["operation"] = self._operation(questions, "WAIT", 0.8)
            return answers
        answers["operation"] = self._operation(questions, "CLICK", 0.9)
        answers["click_target"] = self._target(questions, "click_target", int(target["index"]))
        answers["fact_for"] = choose("none", 1.0, {})
        return answers

    def _scroll(self, questions: dict[str, Any]) -> dict[str, Any]:
        """Always scroll down when the page offers it."""
        return {
            "goal_done": noul(0.02),
            "no_move": noul(0.02),
            "operation": self._operation(questions, "SCROLL_DOWN", 0.95),
            "fact_for": choose("none", 1.0, {}),
        }

    def _wait(self, questions: dict[str, Any]) -> dict[str, Any]:
        """Always wait, which is what a decision model does when nothing looks worth clicking."""
        return {
            "goal_done": noul(0.02),
            "no_move": noul(0.02),
            "operation": self._operation(questions, "WAIT", 0.95),
            "fact_for": choose("none", 1.0, {}),
        }

    def _unsure(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Split the operation question almost evenly so the margin is under 0.1."""
        options = options_of(questions, "operation")
        if len(options) < 2:
            return self._click_dead(state, questions)
        first = options[0]
        second = options[1]
        rest = [option for option in options if option not in (first, second)]
        probabilities = {first: 0.5, second: 0.45}
        for option in rest:
            probabilities[option] = 0.05 / max(1, len(rest))
        return {
            "goal_done": noul(0.02),
            "no_move": noul(0.02),
            "operation": {"type": "choice", "choice": first, "probabilities": probabilities, "confidence": 0.5},
            "fact_for": choose("none", 1.0, {}),
        }

    def _single(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Click the only control on the page, if there is one."""
        answers: dict[str, Any] = {"goal_done": noul(0.02), "no_move": noul(0.02)}
        answers["operation"] = self._operation(questions, "CLICK", 0.9)
        target = element_of(state, "click")
        if target is not None and "click_target" in questions:
            answers["click_target"] = self._target(questions, "click_target", int(target["index"]))
        answers["fact_for"] = choose("none", 1.0, {})
        return answers


class FakeFallback:
    """A fallback with no network behind it, so the loop-guard path can be tested."""

    name = "fake"

    def __init__(self, actions: list[dict[str, Any]] | None = None, cost: float = 0.002) -> None:
        self.actions = list(actions or [])
        self.calls: list[dict[str, Any]] = []
        self.cost = cost

    async def decide(self, state: dict[str, Any], images: list[str]) -> dict[str, Any]:
        """Return the next scripted action, repeating the last one when the list runs out."""
        self.calls.append({"reason": state.get("reason_you_are_asked", ""), "images": len(images)})
        if not self.actions:
            action: dict[str, Any] = {"action": "wait", "index": 0, "fact": "", "reason": "nothing scripted"}
        elif len(self.calls) > len(self.actions):
            action = self.actions[-1]
        else:
            action = self.actions[len(self.calls) - 1]
        record = dict(action)
        record["cost"] = self.cost
        return record