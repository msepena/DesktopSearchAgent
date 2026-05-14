from pathlib import Path

import pytest

from desktop_search.config import Settings
from desktop_search.llm import answer, answer_from_web
from desktop_search.retriever import Hit
from desktop_search.web_search import WebResult


class _FakeBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMessagesAPI:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self._text)


class FakeAnthropic:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessagesAPI(text)


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_model="claude-sonnet-4-6")


def test_returns_text_and_extracts_citations(settings: Settings) -> None:
    fake = FakeAnthropic(
        "The answer is X [source: /x/a.md p.1]. More detail [source: /x/b.md]."
    )
    hits = [
        Hit(text="snippet A", source=Path("/x/a.md"), section="p.1", score=0.9),
        Hit(text="snippet B", source=Path("/x/b.md"), section=None, score=0.7),
    ]
    ans = answer(settings, "what is X?", hits, client=fake)
    assert "answer is X" in ans.text
    assert "[source: /x/a.md p.1]" in ans.citations
    assert "[source: /x/b.md]" in ans.citations
    assert len(ans.citations) == 2


def test_system_block_carries_cache_control(settings: Settings) -> None:
    fake = FakeAnthropic("ok")
    answer(settings, "q", [], client=fake)
    system = fake.messages.last_kwargs["system"]
    assert isinstance(system, list)
    assert len(system) == 1
    assert system[0]["type"] == "text"
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_uses_configured_model(settings: Settings) -> None:
    fake = FakeAnthropic("ok")
    answer(settings, "q", [], client=fake)
    assert fake.messages.last_kwargs["model"] == "claude-sonnet-4-6"


def test_context_snippets_appear_in_user_message(settings: Settings) -> None:
    fake = FakeAnthropic("ok")
    hits = [
        Hit(text="distinct_marker_xyz", source=Path("/x/notes.md"), section="slide 1", score=0.9),
    ]
    answer(settings, "q", hits, client=fake)
    user_msg = fake.messages.last_kwargs["messages"][0]["content"]
    assert "distinct_marker_xyz" in user_msg
    assert "/x/notes.md" in user_msg
    assert "slide 1" in user_msg


def test_empty_context_still_calls_model(settings: Settings) -> None:
    fake = FakeAnthropic("I cannot answer from the provided files.")
    ans = answer(settings, "q", [], client=fake)
    assert "cannot answer" in ans.text
    assert ans.citations == []


def test_citations_are_deduplicated(settings: Settings) -> None:
    fake = FakeAnthropic(
        "X [source: /x/a.md p.1] and again X [source: /x/a.md p.1]."
    )
    ans = answer(settings, "q", [], client=fake)
    assert ans.citations == ["[source: /x/a.md p.1]"]


def test_no_citations_returns_empty_list(settings: Settings) -> None:
    fake = FakeAnthropic("Plain text without any source tags.")
    ans = answer(settings, "q", [], client=fake)
    assert ans.citations == []


# --- answer_from_web -----------------------------------------------------


def test_answer_from_web_returns_url_citations(settings: Settings) -> None:
    fake = FakeAnthropic(
        "Foo is good [source: https://foo.example]. So is bar [source: https://bar.example]."
    )
    results = [
        WebResult(title="Foo Docs", url="https://foo.example", snippet="Foo intro."),
        WebResult(title="Bar Site", url="https://bar.example", snippet="Bar intro."),
    ]
    ans = answer_from_web(settings, "what?", results, client=fake)
    assert "[source: https://foo.example]" in ans.citations
    assert "[source: https://bar.example]" in ans.citations
    assert len(ans.citations) == 2


def test_answer_from_web_uses_distinct_system_prompt(settings: Settings) -> None:
    fake = FakeAnthropic("ok")
    answer_from_web(settings, "q", [], client=fake)
    system_text = fake.messages.last_kwargs["system"][0]["text"]
    assert "web search results" in system_text.lower()
    assert fake.messages.last_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_answer_from_web_includes_urls_in_user_message(settings: Settings) -> None:
    fake = FakeAnthropic("ok")
    results = [
        WebResult(title="T", url="https://distinct-url-marker.example", snippet="body"),
    ]
    answer_from_web(settings, "q", results, client=fake)
    user_msg = fake.messages.last_kwargs["messages"][0]["content"]
    assert "https://distinct-url-marker.example" in user_msg
