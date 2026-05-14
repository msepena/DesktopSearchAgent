from pathlib import Path

import pytest

from desktop_search.config import Settings
from desktop_search.indexer import get_collection
from desktop_search.pipeline import ask


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

    def create(self, **kwargs):
        self.call_count += 1
        return _FakeResponse(self._text)


class FakeAnthropic:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessagesAPI(text)


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


def test_empty_index_returns_none_source(settings: Settings) -> None:
    fake = FakeAnthropic("should-not-be-called")
    res = ask(
        settings,
        "anything",
        embedder=StubEmbedder([1.0, 0.0, 0.0, 0.0]),
        client=fake,
    )
    assert res.source == "none"
    assert res.hits == []
    assert res.citations == []
    assert fake.messages.call_count == 0


def test_high_confidence_returns_local_answer(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [
        ("a", [1.0, 0.0, 0.0, 0.0], "doc text", "/x/a.md", "p.1"),
    ])
    fake = FakeAnthropic("From the file [source: /x/a.md p.1].")
    res = ask(
        settings,
        "q",
        embedder=StubEmbedder([1.0, 0.0, 0.0, 0.0]),
        client=fake,
    )
    assert res.source == "local"
    assert "From the file" in res.answer
    assert res.citations == ["[source: /x/a.md p.1]"]
    assert len(res.hits) == 1
    assert res.hits[0].score >= settings.confidence_threshold
    assert fake.messages.call_count == 1


def test_low_confidence_returns_web_stub(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    # Orthogonal vector — query [1,0,0,0] vs stored [0,1,0,0] gives score ~0
    _seed(coll, [
        ("a", [0.0, 1.0, 0.0, 0.0], "unrelated text", "/x/a.md", ""),
    ])
    fake = FakeAnthropic("should-not-be-called")
    res = ask(
        settings,
        "q",
        embedder=StubEmbedder([1.0, 0.0, 0.0, 0.0]),
        client=fake,
    )
    assert res.source == "web"
    assert "Web search fallback" in res.answer
    assert res.citations == []
    assert len(res.hits) == 1
    assert res.hits[0].score < settings.confidence_threshold
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
    )
    assert res.source == "local"
    assert len(res.hits) == settings.top_k
