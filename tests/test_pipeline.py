from pathlib import Path

import pytest

from desktop_search.config import Settings
from desktop_search.indexer import get_collection
from desktop_search.pipeline import ask
from desktop_search.web_search import WebResult


class StubEmbedder:
    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


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
        self.call_count = 0
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        return _FakeResponse(self._text)


class FakeAnthropic:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessagesAPI(text)


class FakeWebProvider:
    def __init__(self, results: list[WebResult] | Exception) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 5) -> list[WebResult]:
        self.calls.append((query, max_results))
        if isinstance(self._results, Exception):
            raise self._results
        return list(self._results)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        folders=[],
        chroma_path=tmp_path / "chroma",
        top_k=3,
        confidence_threshold=0.5,
    )


def _seed(coll, rows: list[tuple[str, list[float], str, str, str]]) -> None:
    coll.upsert(
        ids=[r[0] for r in rows],
        embeddings=[r[1] for r in rows],
        documents=[r[2] for r in rows],
        metadatas=[
            {"source": r[3], "section": r[4], "mtime": 1.0, "chunk_index": 0}
            for r in rows
        ],
    )


# --- 'none' branch -------------------------------------------------------


def test_empty_index_returns_none_source(settings: Settings) -> None:
    fake = FakeAnthropic("should-not-be-called")
    res = ask(
        settings,
        "anything",
        embedder=StubEmbedder([1.0, 0.0, 0.0, 0.0]),
        client=fake,
        web_provider=FakeWebProvider([]),
    )
    assert res.source == "none"
    assert res.hits == []
    assert res.citations == []
    assert fake.messages.call_count == 0


# --- 'local' branch ------------------------------------------------------


def test_high_confidence_returns_local_answer(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [("a", [1.0, 0.0, 0.0, 0.0], "doc text", "/x/a.md", "p.1")])
    fake = FakeAnthropic("From the file [source: /x/a.md p.1].")
    res = ask(
        settings,
        "q",
        embedder=StubEmbedder([1.0, 0.0, 0.0, 0.0]),
        client=fake,
        web_provider=FakeWebProvider([]),
    )
    assert res.source == "local"
    assert res.citations == ["[source: /x/a.md p.1]"]
    assert res.hits[0].score >= settings.confidence_threshold
    assert fake.messages.call_count == 1


# --- 'web' branch --------------------------------------------------------


def test_low_confidence_runs_web_search_and_cites_urls(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [("a", [0.0, 1.0, 0.0, 0.0], "unrelated", "/x/a.md", "")])

    web_provider = FakeWebProvider([
        WebResult(title="Foo", url="https://example.com/foo", snippet="Foo bar."),
        WebResult(title="Bar", url="https://example.com/bar", snippet="More foo."),
    ])
    fake = FakeAnthropic(
        "Per the web [source: https://example.com/foo] and [source: https://example.com/bar]."
    )
    res = ask(
        settings,
        "q",
        embedder=StubEmbedder([1.0, 0.0, 0.0, 0.0]),
        client=fake,
        web_provider=web_provider,
    )
    assert res.source == "web"
    assert "[source: https://example.com/foo]" in res.citations
    assert "[source: https://example.com/bar]" in res.citations
    assert web_provider.calls == [("q", settings.top_k)]
    # Web system prompt was used (still has cache_control)
    system = fake.messages.last_kwargs["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_low_confidence_with_empty_web_results(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [("a", [0.0, 1.0, 0.0, 0.0], "unrelated", "/x/a.md", "")])

    fake = FakeAnthropic("should-not-be-called")
    res = ask(
        settings,
        "q",
        embedder=StubEmbedder([1.0, 0.0, 0.0, 0.0]),
        client=fake,
        web_provider=FakeWebProvider([]),
    )
    assert res.source == "web"
    assert "no results" in res.answer.lower()
    assert fake.messages.call_count == 0


def test_low_confidence_with_web_provider_failure(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [("a", [0.0, 1.0, 0.0, 0.0], "unrelated", "/x/a.md", "")])

    fake = FakeAnthropic("should-not-be-called")
    res = ask(
        settings,
        "q",
        embedder=StubEmbedder([1.0, 0.0, 0.0, 0.0]),
        client=fake,
        web_provider=FakeWebProvider(RuntimeError("boom")),
    )
    assert res.source == "web"
    assert "boom" in res.answer
    assert fake.messages.call_count == 0


def test_response_carries_all_retrieved_hits(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [
        ("a", [1.0, 0.0, 0.0, 0.0], "A text", "/x/a.md", ""),
        ("b", [0.9, 0.1, 0.0, 0.0], "B text", "/x/b.md", ""),
        ("c", [0.8, 0.2, 0.0, 0.0], "C text", "/x/c.md", ""),
    ])
    fake = FakeAnthropic("answer")
    res = ask(
        settings,
        "q",
        embedder=StubEmbedder([1.0, 0.0, 0.0, 0.0]),
        client=fake,
        web_provider=FakeWebProvider([]),
    )
    assert res.source == "local"
    assert len(res.hits) == settings.top_k
