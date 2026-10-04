# Extending mmjb

Four extension points. No framework, no base classes you must inherit, no registry to register with
at import time. Each one is a name the environment can point at.

- a **decision model**: a class with `async decide(state, questions, images)`
- a **fallback**: a class with `async decide(state, images)`
- a **safety policy**: a callable `allow(action, element, goal) -> bool`
- a **hook**: a callable `on_step(record)` that runs after each step

The runnable version of all four is in `examples/`:

```
examples/decider_plugin.py   a decision model with no network behind it
examples/fallback_plugin.py  a fallback with no network behind it
examples/policy.py           a safety policy and a logging policy
examples/run_with_plugins.py runs the agent with all four at once
```

---

## 1. A new decision model

A decision model answers the questions of one step. Twenty lines is enough.

```python
from mmjb.deciders import Answer


class FirstLetterDecider:
    """Scores each option by its first letter. No network, no API key."""

    name = "first-letter"

    def __init__(self, model: str = "no model") -> None:
        self.model = model

    async def decide(self, state: dict, questions: dict, images: list[str]) -> dict[str, Answer]:
        answers = {}
        for key, question in questions.items():
            if question.kind == "noul":
                answers[key] = Answer(key=key, kind="noul", noul=0.5)
                continue
            probabilities = {}
            best = ""
            best_probability = 0.0
            for option in question.options:
                label = str(option)
                probability = 0.9 if label[:1].upper() in ("C", "T", "S") else 0.4
                probabilities[label] = probability
                if probability > best_probability:
                    best = label
                    best_probability = probability
            answers[key] = Answer(
                key=key,
                kind="choice",
                choice=best,
                probabilities=probabilities,
                confidence=best_probability,
            )
        return answers
```

What you get:

- `state` is a plain dict: `goal`, `page` (url, title, text), `elements`, `facts`,
  `offscreen_controls`, `disabled_controls`, `recent_actions`, `allow_writes`, and `screenshot_note`
  when there was a screenshot.
- `questions` maps a key to a `Question` with `.kind` (`"choice"` or `"noul"`), `.instructions` and
  `.options`.
- `images` is a list of data URI strings, empty when there was no screenshot.
- Return `Answer` objects, or plain dicts of the form
  `{"type": "choice", "choice": ..., "probabilities": {...}, "confidence": ...}` and
  `{"type": "noul", "noul": 0.0}`. Both are read the same way.
- The margin the agent uses is your top probability minus the runner-up. If you return only a
  `choice` and no probabilities the confidence is used as the probability.

Run it with the path:

```bash
MMJB_DECIDER=examples.decider_plugin:FirstLetterDecider uv run mmjb run --goal "..." --url ...
```

Or from Python:

```python
from mmjb import Agent, Result

agent = Agent(goal="...", start_url="...", decider=FirstLetterDecider(), allow_writes=True)
result = await agent.run()
print(result.outcome, result.decider_calls, result.fallback_cost)
```

### Register it so it has a name

Put this in your own package's `pyproject.toml`:

```toml
[project.entry-points."mmjb.deciders"]
first-letter = "mypackage.decider:FirstLetterDecider"
```

Install it and `MMJB_DECIDER=first-letter` works. `pip install mmjb-decider` is all it takes for
somebody else to gain a decision model with no change to this repository.

If your decider holds an HTTP client, give it an `async def close(self)` method and the agent will
call it when the run ends.

---

## 2. A new fallback

The fallback takes one step when the decision model is unsure, says no move exists, or a click has
changed nothing three times running.

```python
class CheapFallback:
    """Presses the first control on the page. Never calls a model."""

    name = "cheap"

    def __init__(self, cost: float = 0.0) -> None:
        self.cost = cost

    async def decide(self, state: dict, images: list[str]) -> dict:
        for element in state.get("elements") or []:
            if element.get("kind") == "click":
                return {"action": "click", "index": int(element["index"]), "reason": "the first control", "cost": self.cost}
        return {"action": "wait", "index": 0, "reason": "nothing clickable", "cost": self.cost}
```

The action is one of:

| action | what the agent does |
| --- | --- |
| `click` | clicks element `index` |
| `type` | types the fact named by `fact` into element `index` |
| `select` | picks the option named by `fact` in the dropdown at `index` |
| `scroll_down`, `scroll_up` | scrolls |
| `goto` | opens `url` (or `fact` if there is no `url`) |
| `wait` | waits about a second |
| `back` | goes back in the tab's history |
| `press` | presses the key named by `key` (a key name such as `ArrowDown`, `Enter` or `Tab`) |
| `click_at` | clicks the point `x`, `y`, in pixels of the plain screenshot (the second image) |
| `drag` | presses at `x`, `y`, moves to `x2`, `y2` and releases |
| `done` | checks the page for a confirmation and stops if it finds one |
| `blocked` | stops the run with outcome `blocked` |

`images` holds the numbered screenshot and, when the agent has a browser, the same page with no boxes drawn on it. `state["scroll"]` has
`viewport_width` and `viewport_height`, the size of both images. Points, drags and key presses count as writes: they are refused while
`allow_writes` is off.

`state` also carries `reason_you_are_asked`, so the fallback knows why it was woken up, and
`unconfirmed_done_claim` when a DONE claim could not be proved. `cost` is added to the run total and
reported as `fallback_cost`.

Run it with `MMJB_FALLBACK=examples.fallback_plugin:CheapFallback`, or register it:

```toml
[project.entry-points."mmjb.fallbacks"]
cheap = "mypackage.fallback:CheapFallback"
```

---

## 3. Custom safety rules

A policy is a plain function. Return True to allow, False to refuse.

```python
def allow_except_delete(action: str, element: object, goal: str) -> bool:
    label = str(getattr(element, "label", "")).lower()
    if action == "click" and "delete" in label:
        return False
    if action in ("type", "select", "upload") and "form" not in goal.lower():
        return False
    return True
```

- `action` is `"click"`, `"type"`, `"select"` or `"upload"`.
- `element` is an `Element`, with `.index`, `.kind`, `.role`, `.label`, `.value`, `.checked`,
  `.disabled` and `.required`. It is also a string when there is no element behind the action.
- `goal` is the goal text.

The policy is asked twice: once when the menu of operations is built, so a refused control is never
offered to the decision model, and once before the action runs. **A custom policy replaces the
built-in rules.** `mmjb.agent.refusal_reason` holds them if you want to build on them instead.

```python
from mmjb.agent import refusal_reason


def allow_unless_account(action: str, element: object, goal: str) -> bool:
    if "account" in str(getattr(element, "label", "")).lower() and "account" not in goal.lower():
        return False
    return refusal_reason(action, getattr(element, "label", ""), goal, allow_writes=True) == ""
```

Pass it directly:

```python
agent = Agent(goal="...", policy=allow_except_delete)
```

or by name:

```bash
MMJB_POLICY=examples.policy:allow_except_delete
```

---

## 4. A step hook

Any callable that takes a `StepRecord`. Useful for logging, a cost ledger, or a screenshot per step.

```python
import json
import sys


def on_step(record) -> None:
    print(json.dumps(record.as_dict()), file=sys.stderr)


agent = Agent(goal="...", on_step=on_step)
```

The record has `step`, `source` (`decider` or `fallback`), `operation`, `element`, `margin`, `effect`,
`url`, `index`, `fact` and `reason`. An exception in the hook is caught and added to the run's notes,
so a broken logger cannot end a run.

---

## Reading the result

```python
result = await agent.run()
result.outcome       # verified, blocked, unverified, step_limit or error
result.steps         # list of StepRecord
result.decider_calls # one per step, plus one per confirmation question
result.fallback_calls
result.fallback_cost
result.seconds
result.final_url
result.notes
result.screenshot    # the last numbered screenshot, as JPEG bytes
result.as_dict()     # the same thing as a plain dict, which is what --json prints
```

`verified` means the goal was met and the page showed a confirmation for it. `unverified` means the
model claimed the goal was met twice and the page never showed proof. `blocked` means nothing on the
page can move the goal. `step_limit` means it ran out of steps or seconds.

---

## Checklist for a new decision model

- Name the class attribute `name`. It shows up in `Agent.status()`.
- Read credentials from the environment in `__init__`, never from a file.
- Answer every question key you are given. A key you leave out is treated as no answer.
- Return probabilities that add up to roughly 1.0, so the margin means something.
- Return an `async def close(self)` if you hold a client.
- Raise `DeciderError` with a message that says what to fix. The agent turns it into a fallback call
  and puts the message in `result.notes`, so the run survives.