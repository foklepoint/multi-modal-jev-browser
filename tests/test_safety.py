"""Safety: what the agent refuses to do, and what a custom policy can change."""
from __future__ import annotations

from mmjb.agent import default_allowed, mentions, refusal_reason
from mmjb.observe import Element


def element(label: str, kind: str = "click") -> Element:
    return Element(index=1, kind=kind, role="button", label=label)


def test_writes_are_off_by_default():
    for label in ("Submit", "Send the form", "Save changes", "Post", "Publish"):
        assert refusal_reason("click", label, "do the thing", False) != ""


def test_typing_needs_allow_writes():
    assert refusal_reason("type", "Your name", "fill the form", False) != ""
    assert refusal_reason("select", "Category", "fill the form", False) != ""
    assert refusal_reason("upload", "Logo", "fill the form", False) != ""
    assert refusal_reason("type", "Your name", "fill the form", True) == ""
    assert refusal_reason("select", "Category", "fill the form", True) == ""
    assert refusal_reason("upload", "Logo", "fill the form", True) == ""


def test_destructive_clicks_need_the_goal_to_ask_for_them():
    for word in ("Delete account", "Remove row", "Log out", "Pay now", "Buy credits", "Upgrade plan", "Subscribe"):
        assert refusal_reason("click", word, "fill in the profile form", True) != "", word
        assert refusal_reason("click", word, word.lower(), True) == "", word


def test_the_goal_word_must_match_a_whole_word():
    assert mentions("resend the file", "send") is False
    assert mentions("send the file", "send") is True
    assert mentions("Save and continue", "save") is True
    assert mentions("Bookkeeping", "book") is False


def test_a_submit_button_is_allowed_once_the_goal_asks_for_it():
    assert refusal_reason("click", "Send the form", "fill the form and send it", True) == ""
    assert refusal_reason("click", "Send the form", "fill the form", False) != ""


def test_default_allowed_uses_the_allow_writes_flag():
    assert default_allowed("click", element("Save"), "save the draft", False) is False
    assert default_allowed("click", element("Save"), "save the draft", True) is True
    assert default_allowed("type", element("Your name", "type"), "fill the form", False) is False


def test_a_custom_policy_replaces_the_builtin_rules(monkeypatch):
    from mmjb.agent import Agent

    seen = []

    def policy(action, item, goal):
        seen.append((action, item.label, goal))
        return action != "click"

    agent = Agent(goal="anything", policy=policy)
    assert agent.policy_for()("click", element("Delete"), "anything") is False
    assert agent.policy_for()("type", element("Your name", "type"), "anything") is True
    assert seen[0] == ("click", "Delete", "anything")


def allow_everything(action, item, goal):
    """A policy that permits every action. Named by MMJB_POLICY in the test below."""
    return True


def test_a_policy_can_be_named_in_the_environment(monkeypatch):
    from mmjb.agent import Agent

    monkeypatch.setenv("MMJB_POLICY", "test_safety:allow_everything")
    agent = Agent(goal="anything")
    agent.load_policy()
    assert agent.custom_policy is allow_everything
    assert agent.policy_for()("click", element("Delete"), "anything") is True