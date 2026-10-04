# AGENTS.md

You have just been pointed at this repository. This file is for you.

## What this is

`multi-modal-jev-browser` (package `mmjb`) is a browser agent. A small, cheap decision model looks
at a page and picks the next action. An ordinary vision LLM steps in only when the decision model is
unsure or stuck. One decision-model call per step reads a numbered screenshot, the element list, the
page text, the task facts and what the last actions changed.

It is a CLI (`mmjb`), a Python library (`from mmjb import Agent`) and an MCP server (`mmjb-mcp`).

## Install and run

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run playwright install chromium
```

MCP server, no clone needed:

```bash
uvx --from git+https://github.com/foklepoint/multi-modal-jev-browser mmjb-mcp
```

Before anything else, check the setup. It prints one pass or fail line per check with the exact fix
for each failure, and exits 0 only when the browser works and at least one decision model answers:

```bash
uv run mmjb doctor
```

## Environment variables

| Variable | What it does |
| --- | --- |
| `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` | Clef, the default decision model |
| `OPENROUTER_API_KEY` | Jev, the cheapest decision model |
| `MMJB_DECIDER` | `clef`, `clef-flash`, `jev`, `llm`, a plugin name, or `package.module:ClassName` |
| `MMJB_DECIDER_MODEL` | the LiteLLM model behind `MMJB_DECIDER=llm` |
| `MMJB_CLEF_MODEL` | `clef` (27B, reads page text) or `clef-flash` (9B) |
| `MMJB_FALLBACK_MODEL` | the vision LLM that takes one step when the decision model is unsure |
| `MMJB_CDP_URL` | attach to a Chrome you started with `--remote-debugging-port=9222` |
| `MMJB_CHROME` | `1` to use your installed Google Chrome |
| `MMJB_HEADFUL` | `1` to watch the browser window |
| `MMJB_POLICY` | `package.module:function` for a custom safety policy |
| `MMJB_FALLBACK` | `package.module:ClassName` for a custom fallback |
| `MMJB_LOG_LEVEL` | logging for the MCP server, always on stderr |

Keys are read only from the environment. Never write one into a file. `.env.example` documents them all.

## MCP tools

One example call each. `index` always comes from `look()`.

```text
browse(goal="Fill in the contact form and send it",
       start_url="https://example.com/contact",
       facts={"name": "Ada Lovelace", "message": "I would like to hear more."},
       allow_writes=true,
       max_steps=20)

look()

click(index=5, allow_writes=true)

type(index=2, text="Ada Lovelace", allow_writes=true)

select(index=3, option="Tools", allow_writes=true)

scroll(direction="down")

goto(url="https://example.com/pricing")

back()

press(key="Enter")
click_at(x=320, y=180)
drag(x=90, y=175, to_x=500, to_y=180)

status()

doctor()
```

`allow_writes` is false by default on every tool. Without it the agent will not type, select, upload
or click a submit button. `browse` sets it for the rest of the session too.

## The CLI

```bash
mmjb run --goal "..." --url "..." --fact key=value --allow-writes --json
mmjb look --url "..." --out shot.jpg
mmjb doctor
```

`mmjb run --json` prints one JSON object on stdout and everything else on stderr. Exit codes:
0 verified, 2 blocked, 3 unverified, 4 step limit, 1 error.

## Reading the result of `browse`

The JSON has `outcome` and it is the field that matters:

- `verified`: the goal was met and the page showed a confirmation for it
- `unverified`: the model claimed the goal was met twice and the page never showed proof
- `blocked`: nothing on the page can move the goal forward
- `step_limit`: it ran out of steps or seconds
- `error`: something went wrong, see `notes`

Also `steps` (each with `operation`, `element`, `effect`, `url`), `decider_calls`, `fallback_calls`,
`fallback_cost`, `seconds` and `final_url`. A low `fallback_calls` compared with `decider_calls` means
the cheap model was doing the work.

## Running the tests

```bash
uv run pytest -q                     # offline, no keys needed
uv run pytest -m live -q -s          # needs a real decision model
```

The offline suite serves `tests/site` from a `http.server` fixture and drives the real loop with a
scripted decision model, so nothing touches the network.

The browser fixture picks Playwright's own Chromium when it is installed and falls back to
`channel="chrome"`. If neither works, `uv run playwright install chromium` fixes it.

## Where to add a decision model

`src/mmjb/deciders.py` holds the three built-in ones. To add your own, write a class with one async
method and name it in an environment variable:

```python
from mmjb.deciders import Answer


class MyDecider:
    name = "mine"

    async def decide(self, state: dict, questions: dict, images: list[str]) -> dict[str, Answer]:
        ...
```

```bash
MMJB_DECIDER=mypackage.decider:MyDecider mmjb run --goal "..." --url "..."
```

Or register it in your package under the `mmjb.deciders` entry point group so `MMJB_DECIDER=mydecider`
works. `docs/extending.md` has the worked examples, and `examples/decider_plugin.py` is a complete
one you can copy.

To add a built-in decision model instead: write the class in `deciders.py`, give it a `name`, add it
to `SHORT_NAMES`, and add a case to `build_decider`. Then add tests that check the request it builds
and how it reads an answer, the way `tests/test_deciders.py` does.

## The three failures you will hit

**`no decision model is configured` on the first call.** No keys in the environment of the process that
started the server. Fix: set `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`, or
`OPENROUTER_API_KEY`, or `MMJB_DECIDER=llm` with `MMJB_DECIDER_MODEL` set to a model your key can
reach. Run `mmjb doctor` to see which check failed.

**`Playwright's Chromium could not start` or `Chrome could not start`.** No browser is installed.
Fix: `uv run playwright install chromium`, or set `MMJB_CHROME=1` to use your installed Google
Chrome.

**`element N is not on the page any more. Call look() again`.** The element numbers came from an
older `look()`. The page has changed since. Fix: call `look()` again and use the new numbers. Do not
guess an index.

## Layout

```
src/mmjb/observe.py     the JavaScript that reads a page, and the typed objects it returns
src/mmjb/marks.py       the numbered overlay and the screenshot
src/mmjb/browser.py     Playwright: actions, tab following, effect detection
src/mmjb/deciders.py    the decision models and how one is chosen
src/mmjb/fallback.py    the vision LLM fallback
src/mmjb/agent.py       the step loop, the guards and the safety rules
src/mmjb/cli.py         mmjb run, mmjb look, mmjb doctor
src/mmjb/mcp_server.py  the MCP tools
examples/bench.py       the five task benchmark and the cost tables in the README
examples/compare_deciders.py  one form, each decision model, a markdown table
examples/make_demo_gif.py     side by side GIF from `mmjb run --record DIR` runs
docs/design.md          why each rule exists
docs/extending.md       the four extension points with worked examples
```

## House style

Plain constructs. `1024 * 1024`, not a shift. Explicit loops, no walrus, no clever one-liners. Short
comments only where the reason is not obvious. No emojis. In README and docs, plain declarative
sentences and no em dashes.