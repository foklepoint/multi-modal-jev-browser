"""The fallback model: the messages it builds, the action it reads back and the cost it reports."""
from __future__ import annotations

import json

import pytest

from mmjb.fallback import (
    ACTIONS,
    FallbackError,
    LiteLLMFallback,
    build_fallback,
    normalise_action,
    response_cost,
)

RECORDED_REPLY = {
    "choices": [
        {
            "message": {
                "content": json.dumps(
                    {
                        "action": "type",
                        "index": 4,
                        "fact": "name",
                        "reason": "the name field is empty and the goal gives a name",
                    }
                )
            }
        }
    ],
    "_hidden_params": {"response_cost": 0.0042},
    "usage": {"prompt_tokens": 900, "completion_tokens": 40},
}


async def test_the_fallback_turns_a_json_reply_into_an_action():
    async def completion(messages, model):
        assert model == "test/fallback"
        return RECORDED_REPLY

    fallback = LiteLLMFallback(model="test/fallback", completion=completion)
    answer = await fallback.decide({"goal": "g", "page": {"text": "t"}, "elements": []}, [])
    assert answer["action"] == "type"
    assert answer["index"] == 4
    assert answer["fact"] == "name"
    assert answer["cost"] == pytest.approx(0.0042)


async def test_the_fallback_prompt_carries_the_page_the_facts_and_the_screenshot():
    async def completion(messages, model):
        return RECORDED_REPLY

    fallback = LiteLLMFallback(model="test/fallback", completion=completion)
    messages = fallback.build_messages(
        {
            "goal": "send the form",
            "facts": {"name": "Ada"},
            "page": {"url": "u", "text": "x" * 9000},
            "elements": [{"index": 1, "label": "Your name"}],
            "recent_actions": [{"step": 1}] * 12,
            "reason_you_are_asked": "the margin was low",
        },
        ["data:image/jpeg;base64,abc"],
    )
    payload = json.loads(messages[1]["content"][0]["text"])
    assert payload["goal"] == "send the form"
    assert payload["facts"] == {"name": "Ada"}
    assert len(payload["page"]["text"]) == 6000
    assert len(payload["recent_actions"]) == 8
    assert payload["reason_you_are_asked"] == "the margin was low"
    assert "untrusted data" in messages[0]["content"]
    assert messages[1]["content"][2]["type"] == "image_url"


def test_every_action_name_in_the_prompt_is_accepted():
    for action in ACTIONS:
        assert normalise_action({"action": action, "index": 0})["action"] == action


def test_an_unknown_action_says_what_is_allowed():
    with pytest.raises(FallbackError) as caught:
        normalise_action({"action": "teleport"})
    assert "scroll_down" in str(caught.value)


def test_a_bad_index_becomes_zero_rather_than_an_exception():
    assert normalise_action({"action": "click", "index": "seven"})["index"] == 0


def test_response_cost_reads_both_places():
    assert response_cost({"_hidden_params": {"response_cost": 0.5}}) == 0.5
    assert response_cost({"usage": {"cost": 0.25}}) == 0.25
    assert response_cost({}) == 0.0


async def test_a_fallback_failure_says_what_to_set():
    async def completion(messages, model):
        raise RuntimeError("no key")

    fallback = LiteLLMFallback(model="test/fallback", completion=completion)
    with pytest.raises(FallbackError) as caught:
        await fallback.decide({"goal": "g"}, [])
    assert "MMJB_FALLBACK_MODEL" in str(caught.value)


async def test_a_reply_with_no_json_says_so():
    async def completion(messages, model):
        return {"choices": [{"message": {"content": "I think you should click."}}]}

    fallback = LiteLLMFallback(model="test/fallback", completion=completion)
    with pytest.raises(FallbackError) as caught:
        await fallback.decide({"goal": "g"}, [])
    assert "did not answer with JSON" in str(caught.value)


def test_build_fallback_reads_the_environment(monkeypatch):
    monkeypatch.delenv("MMJB_FALLBACK", raising=False)
    monkeypatch.delenv("MMJB_FALLBACK_MODEL", raising=False)
    assert isinstance(build_fallback(), LiteLLMFallback)

    monkeypatch.setenv("MMJB_FALLBACK_MODEL", "openai/gpt-4o-mini")
    assert build_fallback().model == "openai/gpt-4o-mini"

    monkeypatch.setenv("MMJB_FALLBACK", "mmjb.fallback:LiteLLMFallback")
    assert isinstance(build_fallback(), LiteLLMFallback)

    monkeypatch.setenv("MMJB_FALLBACK", "nope")
    with pytest.raises(FallbackError) as caught:
        build_fallback()
    assert "package.module:ClassName" in str(caught.value)