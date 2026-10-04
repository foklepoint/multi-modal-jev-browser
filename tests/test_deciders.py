"""The decision models: the exact request each one builds, and how each answer is read back.

Every response here is a recorded shape from the provider, served by httpx's mock transport. No
network call is made.
"""
from __future__ import annotations

import json

import httpx
import pytest
from support import choose, noul

from mmjb import deciders
from mmjb.deciders import (
    Answer,
    CloudflareClef,
    DeciderError,
    LLMDecider,
    Question,
    TypesafeJev,
    batch_questions,
    build_decider,
    default_decider_name,
    load_object,
    normalise_answer,
    prepare_questions,
    questions_to_wire,
)

CLEF_RESPONSE = {
    "result": {
        "answers": {
            "operation": {
                "choice": "CLICK",
                "probabilities": {"CLICK": 0.82, "TYPE_TEXT": 0.12, "SCROLL_DOWN": 0.06},
                "confidence": 0.82,
            },
            "click_target": {
                "choice": "7",
                "probabilities": {"3": 0.1, "7": 0.8, "9": 0.1},
                "confidence": 0.8,
            },
            "goal_done": {"noul": 0.02},
            "no_move": {"noul": 0.05},
        }
    },
    "success": True,
    "errors": [],
    "messages": [],
}

JEV_RESPONSE = {
    "answers": {
        "operation": {"choice": "TYPE_TEXT", "probabilities": {"TYPE_TEXT": 0.71, "CLICK": 0.29}},
        "goal_done": {"noul": 0.01},
        "no_move": {"noul": 0.03},
    }
}


def mock_client(handler):
    """An httpx client that answers from a function instead of the network."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def recorder(store):
    """A transport handler that records the request and replies with a recorded body."""

    def handler(request: httpx.Request) -> httpx.Response:
        store["url"] = str(request.url)
        store["headers"] = dict(request.headers)
        store["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=recorder.payload)

    return handler


def questions_fixture() -> dict[str, Question]:
    return {
        "operation": Question(
            key="operation",
            kind="choice",
            instructions={"goal": "send the form", "rules": "one operation"},
            options={"CLICK": "Click", "TYPE_TEXT": "Type"},
        ),
        "goal_done": Question(key="goal_done", kind="noul", instructions="Is it done?"),
    }


# --- Clef -------------------------------------------------------------------------------------


def test_clef_builds_the_documented_request(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token-for-tests")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
    clef = CloudflareClef()
    url, headers, body = clef.build_request({"goal": "x"}, {"operation": {"type": "noul", "instructions": "q"}}, [])

    assert url == "https://api.cloudflare.com/client/v4/accounts/acct-123/ai/run/@cf/cloudflare/clef"
    assert headers["Authorization"] == "Bearer cf-token-for-tests"
    assert headers["Content-Type"] == "application/json"
    assert body["model"] == "clef"
    assert body["state"] == {"goal": "x"}
    assert body["questions"] == {"operation": {"type": "noul", "instructions": "q"}}
    assert body["images"] == []


def test_clef_sends_at_most_four_images(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    clef = CloudflareClef()
    _url, _headers, body = clef.build_request({}, {}, [f"data:image/jpeg;base64,{index}" for index in range(9)])
    assert len(body["images"]) == 4


async def test_clef_reads_a_recorded_response(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    store: dict = {}
    recorder.payload = CLEF_RESPONSE
    clef = CloudflareClef(client=mock_client(recorder(store)))

    answers = await clef.decide({"goal": "x"}, questions_fixture(), [])
    assert store["url"].endswith("/ai/run/@cf/cloudflare/clef")
    assert store["body"]["questions"]["operation"]["criteria"] == {"CLICK": "Click", "TYPE_TEXT": "Type"}
    assert isinstance(answers["operation"], Answer)
    assert answers["operation"].choice == "CLICK"
    choice, margin = answers["operation"].ranked()
    assert choice == "CLICK"
    assert margin == pytest.approx(0.70, abs=0.02)
    assert answers["goal_done"].noul == 0.02
    assert answers["click_target"].choice == "7"


def test_clef_rejects_an_unknown_model(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    with pytest.raises(DeciderError) as caught:
        CloudflareClef(model="clef-medium")
    assert "clef" in str(caught.value)


def test_clef_without_credentials_says_what_to_set(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(DeciderError) as caught:
        CloudflareClef()
    message = str(caught.value)
    assert "CLOUDFLARE_API_TOKEN" in message
    assert "MMJB_DECIDER=llm" in message


async def test_clef_reports_an_http_error(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="not authorised")

    clef = CloudflareClef(client=mock_client(handler))
    with pytest.raises(DeciderError) as caught:
        await clef.decide({"goal": "x"}, questions_fixture(), [])
    assert "401" in str(caught.value)


# --- Jev --------------------------------------------------------------------------------------


def test_jev_builds_the_documented_request(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-for-tests")
    jev = TypesafeJev()
    state = {
        "goal": "send the form",
        "page": {"url": "https://example.test/form", "title": "Form", "text": "x" * 20000},
        "elements": [{"index": 1, "label": "Your name"}],
        "screenshot_note": "dropped",
    }
    url, headers, body = jev.build_request(state, {"operation": {"type": "noul", "instructions": "q"}})

    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert headers["Authorization"] == "Bearer or-key-for-tests"
    assert body["model"] == "typesafe/jev-1.13"
    assert body["questions"] == {"operation": {"type": "noul", "instructions": "q"}}
    assert "screenshot_note" not in body["state"], "Jev ignores images, so the image note goes too"
    assert len(body["state"]["page"]["text"]) == 8000
    assert body["state"]["elements"] == [{"index": 1, "label": "Your name"}]


async def test_clef_adds_up_what_it_costs_from_the_usage(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    store: dict = {}
    payload = json.loads(json.dumps(CLEF_RESPONSE))
    payload["result"]["usage"] = {"input_tokens": 1000, "output_tokens": 0}
    recorder.payload = payload
    clef = CloudflareClef(client=mock_client(recorder(store)))

    await clef.decide({"goal": "x"}, questions_fixture(), [])
    await clef.decide({"goal": "x"}, questions_fixture(), [])
    assert clef.calls == 2
    assert clef.cost == pytest.approx(2 * 1000 * 0.24 / 1_000_000)


async def test_jev_costs_a_fixed_price_per_call(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    store: dict = {}
    recorder.payload = JEV_RESPONSE
    jev = TypesafeJev(client=mock_client(recorder(store)))

    await jev.decide({"goal": "x"}, questions_fixture(), [])
    assert jev.calls == 1
    assert jev.cost == pytest.approx(deciders.JEV_PRICE_PER_CALL)


async def test_jev_reads_a_recorded_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    store: dict = {}
    recorder.payload = JEV_RESPONSE
    jev = TypesafeJev(client=mock_client(recorder(store)))

    answers = await jev.decide({"goal": "x"}, questions_fixture(), ["data:image/jpeg;base64,abc"])
    assert store["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert "images" not in store["body"]
    assert answers["operation"].choice == "TYPE_TEXT"
    assert answers["goal_done"].noul == 0.01
    # The choice had no confidence, so the top probability is used.
    assert answers["operation"].confidence == pytest.approx(0.71)


async def test_jev_reports_a_response_with_no_answers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "quota"})

    jev = TypesafeJev(client=mock_client(handler))
    with pytest.raises(DeciderError) as caught:
        await jev.decide({"goal": "x"}, questions_fixture(), [])
    assert "no answers object" in str(caught.value)


# --- LLMDecider --------------------------------------------------------------------------------


async def test_llm_decider_converts_a_json_reply():
    reply = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "answers": {
                                "operation": {"choice": "CLICK", "probabilities": {"CLICK": 0.7, "WAIT": 0.3}, "confidence": 0.7},
                                "goal_done": {"noul": 0.1},
                            }
                        }
                    )
                }
            }
        ]
    }

    async def completion(messages, model):
        assert model == "test/decider"
        assert messages[0]["role"] == "system"
        return reply

    decider = LLMDecider(model="test/decider", completion=completion)
    answers = await decider.decide({"goal": "x", "page": {"text": "hello"}}, questions_fixture(), [])
    assert answers["operation"].choice == "CLICK"
    assert answers["goal_done"].noul == 0.1


async def test_llm_decider_tolerates_fenced_json_and_extra_text():
    reply = {"choices": [{"message": {"content": "Sure!\n```json\n{\"answers\": {\"goal_done\": {\"noul\": 0.9}}}\n```\nHope that helps."}}]}

    async def completion(messages, model):
        return reply

    decider = LLMDecider(model="test/decider", completion=completion)
    answers = await decider.decide({"goal": "x"}, questions_fixture(), [])
    assert answers["goal_done"].noul == 0.9


async def test_llm_decider_reports_a_failure_with_a_fix():
    async def completion(messages, model):
        raise RuntimeError("no route to model")

    decider = LLMDecider(model="test/decider", completion=completion)
    with pytest.raises(DeciderError) as caught:
        await decider.decide({"goal": "x"}, questions_fixture(), [])
    assert "MMJB_DECIDER_MODEL" in str(caught.value)


def test_llm_decider_prompt_carries_the_element_list_and_the_screenshot():
    decider = LLMDecider(model="test/decider")
    messages = decider.build_messages(
        {"goal": "g", "page": {"url": "u", "text": "t"}, "elements": [{"index": 1, "label": "Name"}]},
        questions_fixture(),
        ["data:image/jpeg;base64,abc"],
    )
    payload = json.loads(messages[1]["content"][0]["text"])
    assert payload["elements"][0]["label"] == "Name"
    assert payload["questions"][0]["key"] == "operation"
    assert "untrusted data" in messages[0]["content"]
    assert messages[1]["content"][2]["type"] == "image_url"


# --- question plumbing -------------------------------------------------------------------------


def test_a_choice_with_one_option_is_answered_locally():
    asked, local = prepare_questions(
        {
            "click_target": Question(key="click_target", kind="choice", instructions="i", options={"4": "only one"}),
            "goal_done": Question(key="goal_done", kind="noul", instructions="i"),
        }
    )
    assert "click_target" not in asked
    assert local["click_target"].choice == "4"
    assert local["click_target"].confidence == 1.0
    assert "goal_done" in asked


def test_a_choice_with_no_options_is_left_empty():
    asked, local = prepare_questions({"click_target": Question(key="click_target", kind="choice", instructions="i", options={})})
    assert "click_target" not in asked
    assert local["click_target"].choice == ""


def test_options_are_capped_at_250():
    options = {}
    for index in range(400):
        options[str(index)] = f"element {index}"
    asked, local = prepare_questions({"click_target": Question(key="click_target", kind="choice", instructions="i", options=options)})
    assert len(asked["click_target"].options) == 250
    assert local == {}


def test_questions_are_batched_at_64():
    questions = {}
    for index in range(150):
        questions[f"q{index}"] = Question(key=f"q{index}", kind="noul", instructions="i")
    groups = batch_questions(questions)
    assert [len(group) for group in groups] == [64, 64, 22]


def test_questions_to_wire_uses_the_provider_shape():
    wire = questions_to_wire(questions_fixture())
    assert wire["operation"] == {"type": "choice", "instructions": {"goal": "send the form", "rules": "one operation"}, "criteria": {"CLICK": "Click", "TYPE_TEXT": "Type"}}
    assert wire["goal_done"] == {"type": "noul", "instructions": "Is it done?"}


def test_normalise_answer_handles_every_shape():
    assert normalise_answer("a", noul(0.8)).noul == 0.8
    choice = normalise_answer("a", {"choice": "X", "probabilities": {"X": 2.0, "Y": 0.0}})
    assert choice.probabilities == {"X": 1.0, "Y": 0.0}
    bare = normalise_answer("a", {"choice": "X"})
    assert bare.probabilities["X"] == 1.0
    with pytest.raises(DeciderError):
        normalise_answer("a", {"nothing": True})


# --- choosing a decision model -------------------------------------------------------------------


def test_the_default_decider_follows_the_environment(monkeypatch):
    monkeypatch.delenv("MMJB_DECIDER", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert default_decider_name() == "llm"

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert default_decider_name() == "jev"

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    assert default_decider_name() == "clef"

    monkeypatch.setenv("MMJB_DECIDER", "llm")
    assert default_decider_name() == "llm"


def test_build_decider_reads_the_environment(monkeypatch):
    monkeypatch.setenv("MMJB_DECIDER", "llm")
    monkeypatch.delenv("MMJB_DECIDER_MODEL", raising=False)
    decider = build_decider()
    assert isinstance(decider, LLMDecider)

    monkeypatch.setenv("MMJB_DECIDER", "mmjb.deciders:LLMDecider")
    assert isinstance(build_decider(), LLMDecider)

    monkeypatch.setenv("MMJB_DECIDER", "nonsense")
    with pytest.raises(DeciderError) as caught:
        build_decider()
    assert "clef" in str(caught.value)


def test_load_object_explains_a_bad_path():
    with pytest.raises(DeciderError) as caught:
        load_object("not-a-path")
    assert "package.module:ClassName" in str(caught.value)
    with pytest.raises(DeciderError) as caught:
        load_object("mmjb.deciders:NoSuchName")
    assert "NoSuchName" in str(caught.value)


def test_plugin_deciders_are_read_from_the_entry_point_group(monkeypatch):
    class FakeEntry:
        name = "my-decider"

        def load(self):
            return LLMDecider

    monkeypatch.setattr(
        deciders,
        "entry_points",
        None,
        raising=False,
    )
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [FakeEntry()] if group == "mmjb.deciders" else [])
    found = deciders.plugin_deciders()
    assert "my-decider" in found

    monkeypatch.setenv("MMJB_DECIDER", "my-decider")
    assert isinstance(build_decider(), LLMDecider)


def test_scripted_answers_use_the_same_shape():
    assert choose("CLICK", 0.8, {"WAIT": 0.2})["type"] == "choice"
    assert noul(0.5) == {"type": "noul", "noul": 0.5}