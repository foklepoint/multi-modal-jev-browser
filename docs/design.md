# Design

Why the project is built the way it is. Each section records a decision and the failure behind it.

## The shape of a step

One step is one decision-model call. The call carries:

- the state: goal, page url, title and text, the element list, the facts, the offscreen controls,
  the disabled controls and the last twelve step records including what each step changed
- the questions: `operation`, one target question per operation that needs an element
  (`click_target`, `type_text_target`, `select_target`), `scroll_to_target` when there are offscreen
  controls, `fact_for`, `option_for`, and the two yes/no questions `goal_done` and `no_move`
- the numbered screenshot as a data URI

The numbered screenshot is a DOM overlay: one fixed-position container with `pointer-events: none`
and the maximum z-index, one coloured box per element and a small tag carrying the element index.
It is removed straight after the screenshot. The tag number is the element index in the element list,
so one screenshot and one list describe the same thing.

## DONE and BLOCKED are not menu options

They used to be two items in the `operation` menu. As a menu item, BLOCKED was picked in many steps
where a move existed, and a stronger model found a move in nearly all of those steps. Both are now
yes/no questions: `goal_done` is acted on at 0.8 or above, `no_move` is acted on
at 0.9 or above and only when the operation margin is under 0.2. A decision model that is confident
that no move exists and unsure what to do is exactly the case the fallback is for.

## A choice needs two options

`prepare_questions` answers any choice with fewer than two options locally, at full confidence,
instead of sending it. Questions are capped at 250 options each and batched 64 at a time, which is
the limit the providers publish. A question with no options is answered with an empty choice, and the
agent turns that into a WAIT.

## Which value goes in which field

`TYPE_TEXT` needs a fact, and the field is not known until the target question is answered. Asking
`fact_for` as a second call would double the cost of every typing step, so it goes in the same call
as a choice over the fact keys plus `none`, with the whole element list in the state and the
instruction to answer `none` when the next operation is not a typing operation. When the answer is
`none` the agent falls back to a local match between the field label and the fact keys, and if that
also fails the step is refused rather than guessed.

`SELECT` has the same shape, so `option_for` is a choice over the visible option texts of every
dropdown on the page, each described with the dropdown it belongs to. If the answer is not an option
of the chosen dropdown the agent picks the first option the goal names, then the first non-empty one.

This is the one place where the design gives the model a slightly weaker question than the ideal one
(a question about the field it has just been told about). It buys one call per step, which is the
whole point of the project. It is recorded here so a future change can revisit it knowingly.

## Following tabs

A click that opens a new tab leaves the agent on the old one, where nothing changes until the loop
guards fire. The browser records every page the context creates and, after each action, looks for a
page that one of this run's tabs opened. If there is one the agent moves to it and the step record
says `opened a new tab and moved to it: <url>`. If the tab the agent is in has closed, it goes back
to the newest tab of the run that is still open.

## What an action did

Every action is compared with the page before it and the page after it, in this order:

1. a new tab was opened: `opened a new tab and moved to it: <url>`
2. the url changed: `url changed to <url>`
3. the page text changed: `same url; the page text changed by +N characters`
4. a field value or a control state changed: `same url and text; a field value or a control state changed`
5. nothing changed: `nothing visible changed: same url, same text, no new tab`

The page text is `body.innerText`, which does not include the value of an input, so a typed value
shows up in case 4 through the control state, which is the list of `[label, value, checked]` for every
field and toggle on the page.

Three clicks on a control with no visible change hide that control from the options for six steps.
The count is keyed on the control's label, the only thing that survives re-observation.

## The guards

- **Low margin.** An operation margin under 0.1 hands the step to the fallback.
- **Stall.** Six steps with nothing changing hands the step to the fallback.
- **Scrolling.** Six of the last eight steps were scrolls removes every scroll operation from the menu.
  The window is self-clearing: any other kind of step ends the run of scrolls.
- **Waiting.** A WAIT that changed nothing hides WAIT from the menu for three steps. A real Jev run
  picked WAIT six times in a row on a page with six links on it, and the stall guard only caught it on
  the seventh step. A WAIT is only worth an operation slot when the page is doing something itself.
- **Empty page.** A page with no controls at all and nothing offscreen ends the run as blocked rather
  than waiting six times on `about:blank`.
- **Type targets.** A text field that already holds one of the facts is not offered as a typing
  target, so the agent cannot retype the same value into the same field.

## The done check

A form that answers with an inline confirmation loses it on reload, so the agent never reloads to
prove a goal. When `goal_done` is at 0.8 or above the agent looks at text that was not on the first
page of the run:

- if it matches the confirmation word list, the run is `verified`
- otherwise it asks one yes/no question about that new text, and a score of 0.9 or above is enough
- if neither holds the claim is unconfirmed, the fallback is told why, and the loop goes on. Two
  unconfirmed claims end the run as `unverified` rather than claiming success that was not shown.

`unverified` is a real outcome and the agent reports it rather than guessing.

## Safety

`allow_writes` is false by default. When it is false the agent never types, selects or uploads, and
never clicks a control whose label contains submit, send, save, post, publish, pay, buy, delete,
remove, subscribe, upgrade, log out or checkout. When it is true the second list (delete, remove, log
out, pay, buy, upgrade, subscribe, checkout) is still refused unless the goal itself contains that
word. Matching is on whole words, so `resend` does not trip `send`.

The policy is consulted twice: once when the menu is built, so a refused control is never offered, and
once before the action runs, so nothing can slip through. `MMJB_POLICY` or `Agent(policy=...)`
replaces the rules entirely.

Page text is untrusted data. The rules text handed to every decision model says so, and so does the
fallback prompt.

## Environment and defaults

| Variable | Default | What it does |
| --- | --- | --- |
| `MMJB_DECIDER` | clef, then jev, then llm | which decision model: `clef`, `jev`, `llm` or `package.module:ClassName` |
| `MMJB_DECIDER_MODEL` | `MMJB_FALLBACK_MODEL`, then `openrouter/deepseek/deepseek-v4.1-flash` | the model behind `LLMDecider` |
| `MMJB_CLEF_MODEL` | `clef` | `clef` is 27B and reads page text, `clef-flash` is 9B and does not |
| `MMJB_FALLBACK_MODEL` | `openrouter/deepseek/deepseek-v4.1-flash` | the fallback model |
| `MMJB_CDP_URL` | unset | attach to a Chrome instead of launching one |
| `MMJB_CHROME` | unset | use the installed Google Chrome |
| `MMJB_HEADFUL` | unset | show the browser window |
| `MMJB_FALLBACK` | unset | `package.module:ClassName` or a plugin name |
| `MMJB_POLICY` | unset | `package.module:function` |
| `MMJB_LOG_LEVEL` | `WARNING` | logging for the MCP server, always to stderr |

Keys are only ever read from the environment. `.env.example` documents them; the library never opens
that file, because a file of keys is a file of secrets.

## Decisions taken without asking

- **Playwright, not CDP directly.** The brief's stack says Playwright, and it keeps the tab and
  popup handling in one library instead of a hand-rolled CDP client.
- **mcp 2.x with a fallback to 1.x.** mcp 2 renamed FastMCP to MCPServer. The server imports whichever
  is installed.
- **A `<select>` reports an empty value while its placeholder option is selected.** A model that sees
  "Choose one" reads a choice as made. A selected option whose value attribute is empty is not a
  choice.
- **An element made invisible with `opacity: 0` is skipped.** Those are usually a native checkbox
  hidden behind a styled box, and listing both the input and the box makes the model click twice.
- **`Result.outcome` never repeats.** `step_limit` is set before the loop and only replaced when a
  step returns a real outcome, so a run that uses all its steps reports `step_limit` and not "".
- **A custom policy replaces the built-in rules** rather than adding to them. Adding to them would
  make it impossible to allow something the built-in rules refuse.
- **Recording is opt-in.** `mmjb run --record DIR` (or `Agent(record_dir=DIR)`) keeps each numbered
  screenshot, a `steps.jsonl` and a `result.json`; `examples/make_demo_gif.py` turns runs into a GIF.