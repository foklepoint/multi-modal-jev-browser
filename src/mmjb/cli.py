"""The command line: `mmjb run`, `mmjb look` and `mmjb doctor`.

`mmjb run --json` prints exactly one JSON object on stdout and everything else on stderr, so an
agent can pipe it straight into another tool.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from .agent import Agent
from .browser import Browser
from .deciders import DeciderError, build_decider, cheap_probe, clef_configured, jev_configured, plugin_deciders
from .fallback import fallback_model, plugin_fallbacks

EXIT_CODES = {
    "verified": 0,
    "blocked": 2,
    "unverified": 3,
    "step_limit": 4,
    "error": 1,
}

DOCTOR_CLEF_FIX = "set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID, or set OPENROUTER_API_KEY for Jev, or MMJB_DECIDER=llm"
DOCTOR_JEV_FIX = "set OPENROUTER_API_KEY, or set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID for Clef, or MMJB_DECIDER=llm"
DOCTOR_LLM_FIX = "set MMJB_DECIDER=llm and MMJB_DECIDER_MODEL to a LiteLLM model your key can reach (for example openrouter/deepseek/deepseek-v4.1-flash)"
DOCTOR_BROWSER_FIX = "run: playwright install chromium, or set MMJB_CHROME=1 to use your installed Google Chrome"


def parse_facts(pairs: list[str] | None) -> dict[str, str]:
    """Turn --fact name=value arguments into a dictionary."""
    facts: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--fact needs key=value, got '{pair}'. For example --fact name=Ada Lovelace")
        key, value = pair.split("=", 1)
        facts[key.strip()] = value
    return facts


@dataclass
class Check:
    """One line of the doctor report."""

    name: str
    state: str
    detail: str
    fix: str = ""

    def line(self) -> str:
        text = f"{self.state:<5} {self.name}: {self.detail}"
        if self.fix and self.state != "PASS":
            text += f"\n      fix: {self.fix}"
        return text


async def check_browser() -> Check:
    """Launch the browser and load about:blank."""
    cdp = os.environ.get("MMJB_CDP_URL") or ""
    chrome = os.environ.get("MMJB_CHROME") == "1"
    headless = os.environ.get("MMJB_HEADFUL") != "1"
    browser = Browser(cdp_url=cdp, chrome=chrome, headless=headless)
    try:
        await browser.start()
        await browser.page.goto("about:blank")
        version = browser.browser.version if hasattr(browser.browser, "version") else "unknown"
        where = f"attached to {cdp}" if cdp else ("Google Chrome" if chrome else "Playwright Chromium")
        return Check("browser", "PASS", f"{where} {version} launched and loaded about:blank")
    except Exception as error:
        return Check("browser", "FAIL", f"could not start a browser: {str(error)[:200]}", DOCTOR_BROWSER_FIX)
    finally:
        await browser.stop()


async def check_decider(label: str, factory: Any, configured: bool, fix: str) -> Check:
    """Check one decision model's settings and make one small call with it."""
    if not configured:
        return Check(f"decider {label}", "SKIP", "not configured, so it was not called", fix)
    try:
        decider = factory()
    except Exception as error:
        return Check(f"decider {label}", "FAIL", f"cannot be built: {str(error)[:200]}", fix)
    try:
        detail = await cheap_probe(decider)
        return Check(f"decider {label}", "PASS", detail)
    except Exception as error:
        return Check(f"decider {label}", "FAIL", f"the call failed: {str(error)[:200]}", fix)
    finally:
        closer = getattr(decider, "close", None)
        if closer is not None:
            try:
                await closer()
            except Exception:
                pass


async def collect_doctor_checks() -> tuple[list[Check], bool]:
    """Run every check and return them with a yes/no for whether the setup is usable."""
    checks: list[Check] = []
    version = sys.version_info
    if (version.major, version.minor) >= (3, 12):
        checks.append(Check("python", "PASS", f"{version.major}.{version.minor}.{version.micro}"))
    else:
        checks.append(
            Check("python", "FAIL", f"{version.major}.{version.minor}.{version.micro} is too old", "install Python 3.12 or newer")
        )
    try:
        import playwright  # noqa: F401

        checks.append(Check("playwright", "PASS", "the playwright package is importable"))
    except Exception as error:
        checks.append(Check("playwright", "FAIL", f"cannot import playwright: {str(error)[:160]}", "pip install playwright"))
    checks.append(await check_browser())

    wanted = (os.environ.get("MMJB_DECIDER") or "").strip()
    clef_on = clef_configured() and wanted in ("", "clef", "clef-flash")
    jev_on = jev_configured() and wanted in ("", "jev")
    llm_on = wanted in ("", "llm")
    checks.append(await check_decider("clef", lambda: build_decider("clef"), clef_on, DOCTOR_CLEF_FIX))
    checks.append(await check_decider("jev", lambda: build_decider("jev"), jev_on, DOCTOR_JEV_FIX))
    checks.append(await check_decider("llm", lambda: build_decider("llm"), llm_on, DOCTOR_LLM_FIX))

    model = fallback_model()
    providers = [
        name
        for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")
        if os.environ.get(name)
    ]
    if providers:
        checks.append(Check("fallback model", "PASS", f"{model} (keys found: {', '.join(providers)})"))
    else:
        checks.append(
            Check(
                "fallback model",
                "FAIL",
                f"{model} has no API key, so the fallback cannot be called",
                "set one of OPENAI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY or GEMINI_API_KEY",
            )
        )
    plugins = sorted(list(plugin_deciders().keys()))
    if plugins:
        checks.append(Check("decider plugins", "PASS", "found: " + ", ".join(plugins)))
    fallbacks = sorted(list(plugin_fallbacks().keys()))
    if fallbacks:
        checks.append(Check("fallback plugins", "PASS", "found: " + ", ".join(fallbacks)))

    deciders_ok = sum(1 for check in checks if check.name.startswith("decider ") and check.state == "PASS")
    browser_ok = any(check.name == "browser" and check.state == "PASS" for check in checks)
    return (checks, deciders_ok >= 1 and browser_ok)


async def run_doctor(stream: Any = None) -> int:
    """Check Python, the browser, every decision model and the fallback. Returns the exit code."""
    out = stream or sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        checks, healthy = await collect_doctor_checks()
    await run_doctor_text(out, checks, healthy)
    return 0 if healthy else 1


async def run_doctor_text(stream: Any, checks: list[Check], healthy: bool) -> None:
    """Print the doctor report. Shared by the CLI and the MCP doctor tool."""
    deciders_ok = sum(1 for check in checks if check.name.startswith("decider ") and check.state == "PASS")
    for check in checks:
        print(check.line(), file=stream)
    print(
        (f"ready: the browser works and {deciders_ok} decision model(s) answered")
        if healthy
        else "not ready: the browser and at least one working decision model are needed",
        file=stream,
    )


async def command_run(args: argparse.Namespace) -> int:
    """`mmjb run --goal ... --url ... --fact key=value`"""
    facts = parse_facts(args.fact)
    result_out = sys.stdout

    def step_logger(record: Any) -> None:
        print("  " + record.line(), file=sys.stderr)

    agent = Agent(
        goal=args.goal,
        start_url=args.url,
        facts=facts,
        allow_writes=args.allow_writes,
        max_steps=args.max_steps,
        budget_seconds=args.budget,
        on_step=step_logger,
        record_dir=args.record or None,
    )
    # A third-party library writing to stdout would corrupt the JSON, so stdout is closed to everything but the result.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            result = await agent.run()
        finally:
            await agent.close()
    if args.out and result.screenshot:
        with open(args.out, "wb") as handle:
            handle.write(result.screenshot)
        print(f"saved the last numbered screenshot to {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(result.as_dict()), file=result_out)
    else:
        print(result.summary(), file=result_out)
    for note in result.notes:
        print("note: " + note, file=sys.stderr)
    return EXIT_CODES.get(result.outcome, 1)


async def command_look(args: argparse.Namespace) -> int:
    """`mmjb look --url ... --out shot.jpg` saves the numbered screenshot and prints the element list."""
    agent = Agent()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            browser = await agent.ensure_browser()
            await browser.goto(args.url)
            observation, screenshot = await agent.look()
        finally:
            await agent.close()
    if args.out and screenshot:
        with open(args.out, "wb") as handle:
            handle.write(screenshot)
        print(f"saved the numbered screenshot to {args.out}", file=sys.stderr)
    print(observation.listing())
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The whole command line."""
    parser = argparse.ArgumentParser(prog="mmjb", description="A browser agent driven by a small decision model.")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="run the loop until the goal is met or a limit is reached")
    run.add_argument("--goal", required=True, help="what the agent should achieve, in one sentence")
    run.add_argument("--url", default="", help="the URL to start at")
    run.add_argument("--fact", action="append", metavar="KEY=VALUE", help="a value the agent can type, repeat for several")
    run.add_argument("--allow-writes", action="store_true", help="allow typing, selecting, uploading and submit buttons")
    run.add_argument("--max-steps", type=int, default=40, help="give up after this many steps (default 40)")
    run.add_argument("--budget", type=float, default=300.0, help="give up after this many seconds (default 300)")
    run.add_argument("--json", action="store_true", help="print one JSON object on stdout and everything else on stderr")
    run.add_argument("--out", default="", help="save the last numbered screenshot to this file")
    run.add_argument("--record", default="", help="save every numbered screenshot, a steps.jsonl and a result.json in this directory (for the demo GIF)")

    look = commands.add_parser("look", help="save one numbered screenshot and print the element list")
    look.add_argument("--url", required=True, help="the URL to open")
    look.add_argument("--out", default="shot.jpg", help="where to save the numbered screenshot (default shot.jpg)")

    commands.add_parser("doctor", help="check Python, the browser, the decision models and the fallback")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the mmjb console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__

        print(__version__)
        return 0
    if not args.command:
        parser.print_help()
        return 0
    try:
        if args.command == "run":
            return asyncio.run(command_run(args))
        if args.command == "look":
            return asyncio.run(command_look(args))
        return asyncio.run(run_doctor())
    except DeciderError as error:
        print("error: " + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())