from pathlib import Path

import pytest

from desktop_search.config import Settings
from desktop_search.indexer import get_collection
from desktop_search.retriever import search


class StubEmbedder:
    """Returns a fixed vector for every text. Used to control the query side."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(folders=[], chroma_path=tmp_path / "chroma", top_k=3)


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


def test_empty_collection_returns_empty(settings: Settings) -> None:
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    assert search(settings, "anything", embedder=embedder) == []


def test_hits_ordered_by_descending_score(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [
        ("a", [1.0, 0.0, 0.0, 0.0], "doc A text", "/x/a.md", ""),
        ("b", [0.0, 1.0, 0.0, 0.0], "doc B text", "/x/b.md", "p.2"),
        ("c", [0.7, 0.7, 0.0, 0.0], "doc C text", "/x/c.md", "slide 1"),
    ])
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    hits = search(settings, "query", k=3, embedder=embedder)
    assert len(hits) == 3
    assert [h.text for h in hits] == ["doc A text", "doc C text", "doc B text"]
    for i in range(len(hits) - 1):
        assert hits[i].score >= hits[i + 1].score


def test_hit_carries_source_path_and_section(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [
        ("a", [1.0, 0.0, 0.0, 0.0], "doc text", "/x/notes/a.md", "p.3"),
    ])
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    hits = search(settings, "q", k=1, embedder=embedder)
    assert hits[0].source == Path("/x/notes/a.md")
    assert hits[0].section == "p.3"


def test_scores_clamped_to_unit_range(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [
        ("a", [1.0, 0.0, 0.0, 0.0], "match", "/x/a.md", ""),
        ("b", [0.0, 1.0, 0.0, 0.0], "different", "/x/b.md", ""),
    ])
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    hits = search(settings, "q", k=2, embedder=embedder)
    for h in hits:
        assert 0.0 <= h.score <= 1.0
    assert hits[0].text == "match"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].score > hits[1].score


def test_k_overrides_settings_top_k(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [
        ("a", [1.0, 0.0, 0.0, 0.0], "A", "/x/a.md", ""),
        ("b", [0.9, 0.1, 0.0, 0.0], "B", "/x/b.md", ""),
        ("c", [0.0, 1.0, 0.0, 0.0], "C", "/x/c.md", ""),
    ])
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])

    assert len(search(settings, "q", embedder=embedder)) == settings.top_k
    assert len(search(settings, "q", k=1, embedder=embedder)) == 1


def test_empty_section_metadata_becomes_none(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [("a", [1.0, 0.0, 0.0, 0.0], "text", "/x/a.md", "")])
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    hits = search(settings, "q", k=1, embedder=embedder)
    assert hits[0].section is None


def test_k_larger_than_collection_is_safe(settings: Settings) -> None:
    coll = get_collection(settings.chroma_path)
    _seed(coll, [("a", [1.0, 0.0, 0.0, 0.0], "only doc", "/x/a.md", "")])
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    hits = search(settings, "q", k=10, embedder=embedder)
    assert len(hits) == 1
