"""Run the agent with all four extension points at once: a decider, a fallback, a policy and a hook.

    uv run python examples/run_with_plugins.py --url https://example.com/contact \\
        --goal "Fill in the contact form and send it." \\
        --fact name="Ada Lovelace" --fact message="I would like to hear more."

Everything is resolved from the environment, so the same script works with a real model:

    MMJB_DECIDER=examples.decider_plugin:FirstLetterDecider \\
    MMJB_FALLBACK=examples.fallback_plugin:CheapFallback \\
    MMJB_POLICY=examples.policy:allow_except_delete \\
    uv run python examples/run_with_plugins.py --url ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# The example modules live next to this file, so the repository root has to be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmjb import Agent
from mmjb.deciders import load_object


def on_step(record: object) -> None:
    """The on_step hook. Called once per step with the step record."""
    print(json.dumps(record.as_dict()), file=sys.stderr)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent with the example extension points.")
    parser.add_argument("--url", required=True, help="the page to start at")
    parser.add_argument("--goal", default="Fill in the contact form and send it.")
    parser.add_argument("--fact", action="append", default=[], metavar="KEY=VALUE", help="a value the agent can type")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    facts = {}
    for pair in args.fact:
        key, value = pair.split("=", 1)
        facts[key] = value

    decider_path = os.environ.get("MMJB_DECIDER", "examples.decider_plugin:FirstLetterDecider")
    fallback_path = os.environ.get("MMJB_FALLBACK", "examples.fallback_plugin:CheapFallback")
    policy_path = os.environ.get("MMJB_POLICY", "examples.policy:allow_except_delete")
    print("decider: " + decider_path, file=sys.stderr)
    print("fallback: " + fallback_path, file=sys.stderr)
    print("policy: " + policy_path, file=sys.stderr)

    agent = Agent(
        goal=args.goal,
        start_url=args.url,
        facts=facts,
        allow_writes=True,
        decider=load_object(decider_path)(),
        fallback=load_object(fallback_path)(),
        policy=load_object(policy_path),
        on_step=on_step,
        max_steps=args.max_steps,
    )
    try:
        result = await agent.run()
    finally:
        await agent.close()
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))