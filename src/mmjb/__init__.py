"""multi-modal-jev-browser: a browser agent driven by a small decision model.

The decision model answers typed multiple-choice and yes/no questions about one page. An ordinary
vision LLM steps in only when the decision model is unsure or stuck.

    from mmjb import Agent
    agent = Agent(goal="Send the contact form", start_url="https://example.com/contact")
    result = await agent.run(facts={"name": "Ada", "message": "Hello"})
    print(result.outcome)
"""
from __future__ import annotations

__version__ = "0.1.0"

from .agent import Agent, Result, StepRecord
from .browser import Browser
from .deciders import CloudflareClef, Decider, DeciderError, LLMDecider, Question, TypesafeJev, build_decider
from .fallback import Fallback, FallbackError, LiteLLMFallback, build_fallback
from .observe import Element, Observation

__all__ = [
    "Agent",
    "Browser",
    "CloudflareClef",
    "Decider",
    "DeciderError",
    "Element",
    "Fallback",
    "FallbackError",
    "LLMDecider",
    "LiteLLMFallback",
    "Observation",
    "Question",
    "Result",
    "StepRecord",
    "TypesafeJev",
    "build_decider",
    "build_fallback",
    "__version__",
]