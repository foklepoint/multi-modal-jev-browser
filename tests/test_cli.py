"""The command line and the MCP tools: exit codes, JSON on stdout, and the tool list."""
from __future__ import annotations

import json

import pytest
from support import FakeFallback, ScriptedDecider

from mmjb import cli, mcp_server
from mmjb.agent import Agent


def test_exit_codes_are_the_documented_ones():
    assert cli.EXIT_CODES == {
        "verified": 0,
        "blocked": 2,
        "unverified": 3,
        "step_limit": 4,
        "error": 1,
    }


def test_facts_are_parsed_from_key_value_arguments():
    assert cli.parse_facts(["name=Ada", "message=Hello there"]) == {"name": "Ada", "message": "Hello there"}
    assert cli.parse_facts(["url=https://example.test/a?b=c"]) == {"url": "https://example.test/a?b=c"}
    assert cli.parse_facts(None) == {}
    with pytest.raises(SystemExit):
        cli.parse_facts(["nope"])


def test_the_parser_documents_every_option():
    parser = cli.build_parser()
    subparsers = parser._subparsers._group_actions[0].choices
    assert set(subparsers) == {"run", "look", "doctor"}
    for name in ("run", "look", "doctor"):
        text = subparsers[name].format_help()
        for action in subparsers[name]._actions:
            if not action.option_strings:
                continue
            assert action.help, f"{name} {action.option_strings} has no help text"
        assert text

    run_help = parser.parse_args(["run", "--goal", "g", "--url", "u", "--json", "--allow-writes", "--fact", "a=b"])
    assert run_help.command == "run"
    assert run_help.json is True
    assert run_help.allow_writes is True
    look = parser.parse_args(["look", "--url", "u", "--out", "shot.jpg"])
    assert look.command == "look"
    assert look.out == "shot.jpg"
    assert parser.parse_args(["doctor"]).command == "doctor"


@pytest.mark.slow
async def test_run_with_json_prints_one_object_on_stdout(browser, site, capsys, monkeypatch):
    decider = ScriptedDecider("form")
    original = Agent.__init__

    def patched(self, *args, **kwargs):
        kwargs["decider"] = decider
        kwargs["fallback"] = FakeFallback()
        kwargs["browser"] = browser
        original(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "__init__", patched)
    args = cli.build_parser().parse_args(
        [
            "run",
            "--goal",
            "Open the submission form in a new tab, fill it in with the facts and send it.",
            "--url",
            site + "/index.html",
            "--fact",
            "name=Ada Lovelace",
            "--fact",
            "message=I would like to hear more.",
            "--allow-writes",
            "--json",
        ]
    )
    code = await cli.command_run(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert code == 0
    assert payload["outcome"] == "verified"
    assert set(payload) >= {"goal", "outcome", "steps", "decider_calls", "fallback_calls", "fallback_cost", "seconds", "final_url"}
    assert payload["steps"][0]["effect"].startswith("opened a new tab")
    assert "opened a new tab" in captured.err
    assert "DONE" in captured.err
    assert captured.out.strip().count("\n") == 0, "stdout must hold one JSON object and nothing else"


@pytest.mark.slow
async def test_look_writes_the_screenshot_and_prints_the_element_list(browser, site, capsys, tmp_path):
    out = tmp_path / "shot.jpg"
    args = cli.build_parser().parse_args(["look", "--url", site + "/form.html", "--out", str(out)])
    code = await cli.command_look(args)
    captured = capsys.readouterr()
    assert code == 0
    assert out.exists()
    assert out.read_bytes()[:2] == b"\xff\xd8"
    assert "[1]" in captured.out
    assert "Send the form" in captured.out


@pytest.mark.slow
async def test_doctor_prints_a_pass_and_fail_line_for_each_check(browser_kwargs, monkeypatch, capsys):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("MMJB_DECIDER", "llm")
    original = Agent.ensure_browser

    async def no_browser(self):
        raise RuntimeError("no browser in this test")

    monkeypatch.setattr(Agent, "ensure_browser", no_browser)
    code = await cli.run_doctor()
    captured = capsys.readouterr()
    assert code == 1
    assert "PASS  python:" in captured.out
    assert "FAIL  fallback model:" in captured.out
    assert "fix:" in captured.out
    assert "not ready" in captured.out
    assert "PASS  browser:" in captured.out
    assert original is not None


async def test_the_mcp_server_exposes_every_documented_tool():
    tools = await mcp_server.server.list_tools()
    names = set()
    for tool in tools:
        names.add(tool.name)
        assert tool.description, f"{tool.name} needs a description an agent can read"
        assert tool.description.endswith("."), f"{tool.name}: {tool.description}"
    expected = {"browse", "look", "click", "type", "select", "scroll", "goto", "back", "press", "status", "doctor"}
    assert expected <= names


def test_the_mcp_module_never_writes_to_stdout():
    source = open(mcp_server.__file__).read()
    body = "\n".join(line for line in source.split("\n") if not line.strip().startswith("#"))
    assert "print(" not in body, "the MCP server must not print to stdout"


async def test_status_reports_the_configured_models(browser):
    agent = Agent(goal="g", browser=browser, decider=ScriptedDecider(), fallback=FakeFallback())
    report = await agent.status()
    assert report["decider"] == "scripted"
    assert report["fallback"] == "fake"
    assert report["goal"] == "g"
    assert report["steps_taken"] == 0


@pytest.mark.slow
async def test_the_browse_tool_opens_the_start_url_and_returns_the_screenshot(browser, site, monkeypatch):
    decider = ScriptedDecider("form")
    original = mcp_server._agent

    def patched(goal: str = ""):
        agent = original(goal)
        agent.decider = decider
        agent.fallback = FakeFallback()
        return agent

    monkeypatch.setattr(mcp_server, "_agent", patched)
    monkeypatch.setitem(mcp_server.SESSION, "browser", browser)
    parts = await mcp_server.browse(
        goal="Open the submission form in a new tab, fill it in with the facts and send it.",
        start_url=site + "/index.html",
        facts={"name": "Ada Lovelace", "message": "I would like to hear more."},
        allow_writes=True,
        max_steps=10,
    )
    assert isinstance(parts[0], str)
    payload = json.loads(parts[0])
    assert payload["outcome"] == "verified", payload
    assert payload["final_url"].endswith("/form.html")
    assert len(parts) == 2
    assert hasattr(parts[1], "to_image_content")


@pytest.mark.slow
async def test_the_look_tool_returns_text_and_an_image(browser, site, monkeypatch):
    monkeypatch.setitem(mcp_server.SESSION, "browser", browser)
    await mcp_server.goto(site + "/form.html")
    parts = await mcp_server.look()
    assert "Send the form" in parts[0]
    assert len(parts) == 2
    content = parts[1].to_image_content()
    assert content.mime_type == "image/jpeg"


@pytest.mark.slow
async def test_a_browser_error_comes_back_as_a_message_not_a_crash(browser, site, monkeypatch):
    monkeypatch.setitem(mcp_server.SESSION, "browser", browser)
    await mcp_server.goto(site + "/form.html")
    answer = await mcp_server.click(index=99)
    assert answer.startswith("error: ")
    assert "look()" in answer