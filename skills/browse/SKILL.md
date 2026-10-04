---
name: browse
description: Drive a web page with the mmjb MCP server. Use it to fill in and send a form, sign up, request a demo, log in to a page the session already has open, or carry out any goal on a website that needs typing and clicking. Reach for it instead of guessing at HTML or writing a scraper. Call browse for a whole goal; call look, click, type and select when you want to drive the page yourself.
---

# Driving a web page

The `mmjb` MCP server puts a real browser in front of you. `browse` runs a whole goal on its own, one
cheap decision-model step at a time. You read the result and decide what to do next.

## When to use browse

Use `browse` when the task is a sequence of steps on a page: fill a form, sign up, request a demo,
apply for an account, log in, get to a page behind a menu. The decision model is fast and cheap, and a
form takes six to ten steps without any LLM call.

Do not use `browse` for reading a page (fetch it), for anything behind a login the session does not
have, or for anything that needs a CAPTCHA, a canvas, drag and drop, or a custom dropdown that only
opens with the keyboard.

## Calling browse

```text
browse(goal="Fill in the contact form with the facts and send it",
       start_url="https://example.com/contact",
       facts={"name": "Ada Lovelace",
              "message": "I would like to hear more."},
       allow_writes=true,
       max_steps=20)
```

- `goal` is one sentence in plain language, naming the end state, not the steps. "Fill in the contact
  form and send it" beats "click the name box, type, then scroll".
- `start_url` is where the run begins. Leave it empty to use the tab that is already open, which is how
  you continue in the same session.
- `facts` are the values the agent may type. Name each one after what it is: `name`, `email`, `message`,
  `company`. The agent matches fact keys to field labels, so good names are what makes typing work.
  A fact whose value is a path, prefixed `file:`, is uploaded rather than typed.
- `allow_writes` is false by default. Without it the agent will not type, select, upload or click a
  submit button, and the run ends as blocked. Turn it on when you mean it.
- `max_steps` defaults to 40.

The response is a JSON object plus the last numbered screenshot.

## Reading the result

The `outcome` field is the one that matters.

| outcome | what happened | what to do |
| --- | --- | --- |
| `verified` | the page showed a confirmation for the goal | nothing, it is done |
| `unverified` | the model claimed it was done twice and the page never confirmed | look, decide whether it worked, and say so |
| `blocked` | nothing on the page could move the goal forward | read `notes`, then either fix the goal or drive the page yourself |
| `step_limit` | it ran out of steps or seconds | raise `max_steps`, or hand over to the single tools |
| `error` | something broke | `notes` has the message; `doctor()` usually explains it |

`steps` is the trail. Each step has `operation`, `element`, `effect` and `url`. The `effect` field is
the most useful part: "opened a new tab and moved to it", "url changed to", "the page text changed by
+27 characters", or "nothing visible changed". A run full of "nothing visible changed" means the goal
was written for a page that was not there.

`fallback_calls` above `decider_calls` means the cheap model was unsure a lot and the expensive vision
LLM took over. That is a signal that the goal text is vague.

## When to drop to the single tools

Use these when `browse` did not finish the job, or when you want precise control:

1. `look()` returns a numbered screenshot and the numbered element list. Read it.
2. `click(index=5, allow_writes=true)`, `type(index=2, text="...", allow_writes=true)`,
   `select(index=3, option="Tools", allow_writes=true)`.
3. `scroll(direction="down")` for content below the fold, `goto(url)`, `back()`, `press(key="Enter")`.
4. `browse` again for whatever is left, with the same facts.

Every index comes from the last `look()`. After any page change the numbers move, so call `look()`
again. If you get `element N is not on the page any more`, that is the reason: look again.

## Good habits

- Name the goal as an end state, and name the facts after the fields.
- Use `look()` before `browse` when you are not sure what is on the page.
- Use `status()` to see the current URL and whether `allow_writes` is on.
- Use `doctor()` when a run errors immediately. It says exactly which variable to set.
- If a run ends `unverified`, do not report success to the user. Say the page never confirmed it.