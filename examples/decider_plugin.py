"""A complete custom decision model. A decider is a class with one async method.

    MMJB_DECIDER=examples.decider_plugin:FirstLetterDecider uv run python examples/run_with_plugins.py ...

To have it found by name instead of by path, publish it in your own package:

    [project.entry-points."mmjb.deciders"]
    first-letter = "mypackage.decider:FirstLetterDecider"

and set MMJB_DECIDER=first-letter.
"""
from __future__ import annotations

from typing import Any

from mmjb.deciders import Answer


class FirstLetterDecider:
    """A decision model with no network behind it: it scores each option by its first letter.

    It runs the whole loop with no API key at all, which is useful for tests and for checking that
    the plumbing works on a site.
    """

    name = "first-letter"

    def __init__(self, model: str = "no model") -> None:
        # Real deciders read their credentials from the environment here, never from a file.
        self.model = model

    async def decide(self, state: dict[str, Any], questions: dict[str, Any], images: list[str]) -> dict[str, Answer]:
        """Answer every question of this step in one call.

        state is a plain dict: goal, page (url, title, text), elements, facts, offscreen_controls,
        disabled_controls, recent_actions and sometimes screenshot_note.
        questions maps a key to a Question with .kind ("choice" or "noul"), .instructions and .options.
        images is a list of data URI strings, empty when there was no screenshot.
        """
        answers: dict[str, Answer] = {}
        for key, question in questions.items():
            if question.kind == "noul":
                # A yes/no question. The agent acts on it only above a threshold, so 0.5 changes nothing.
                answers[key] = Answer(key=key, kind="noul", noul=0.5)
                continue
            probabilities: dict[str, float] = {}
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