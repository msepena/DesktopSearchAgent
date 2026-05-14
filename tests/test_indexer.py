import os
import time
from pathlib import Path

import pytest

from desktop_search.config import Settings
from desktop_search.indexer import (
    _split_tokens,
    build_index,
    chunk_documents,
    get_collection,
)
from desktop_search.loaders import Document


class StubEmbedder:
    """Deterministic 4-dim vectors derived from text length. No model download."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t) % 7), float(len(t) % 11), float(len(t) % 13), 1.0] for t in texts]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        folders=[],
        chunk_size=50,
        chunk_overlap=10,
        chroma_path=tmp_path / "chroma",
    )


def test_split_tokens_empty_returns_empty() -> None:
    assert _split_tokens("", 100, 10) == []


def test_split_tokens_produces_multiple_chunks_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    chunks = _split_tokens(text, size=20, overlap=5)
    assert len(chunks) > 1


def test_split_tokens_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        _split_tokens("hello", size=10, overlap=10)


def test_chunk_documents_assigns_global_indices(tmp_path: Path) -> None:
    long_text = " ".join(f"token{i}" for i in range(300))
    d1 = Document(text=long_text, source=tmp_path / "a.txt", section="p.1", mtime=1.0)
    d2 = Document(text=long_text, source=tmp_path / "a.txt", section="p.2", mtime=1.0)
    chunks = chunk_documents([d1, d2], chunk_size=50, overlap=10)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
    # Section labels propagate to chunks
    assert {c.section for c in chunks} == {"p.1", "p.2"}


def test_first_run_adds_chunks(tmp_path: Path, settings: Settings) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("hello world " * 100)
    (folder / "b.md").write_text("another file " * 100)

    stats = build_index(settings, paths=[folder], embedder=StubEmbedder())
    assert stats.files_indexed == 2
    assert stats.chunks_added > 0

    coll = get_collection(settings.chroma_path)
    assert coll.count() == stats.chunks_added


def test_second_run_skips_unchanged(tmp_path: Path, settings: Settings) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("hello " * 200)

    stats1 = build_index(settings, paths=[folder], embedder=StubEmbedder())
    stats2 = build_index(settings, paths=[folder], embedder=StubEmbedder())

    assert stats1.chunks_added > 0
    assert stats2.chunks_added == 0
    assert stats2.files_skipped == 1
    assert stats2.files_indexed == 0


def test_modified_file_replaces_old_chunks(tmp_path: Path, settings: Settings) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    f = folder / "a.md"
    f.write_text("original content " * 500)

    stats1 = build_index(settings, paths=[folder], embedder=StubEmbedder())
    coll = get_collection(settings.chroma_path)
    count1 = coll.count()
    assert count1 == stats1.chunks_added
    assert count1 > 0

    # Shrink the file and bump mtime so the run is treated as modified.
    f.write_text("short content")
    new_time = time.time() + 5
    os.utime(f, (new_time, new_time))

    stats2 = build_index(settings, paths=[folder], embedder=StubEmbedder())
    count2 = coll.count()

    assert stats2.files_indexed == 1
    assert stats2.chunks_added > 0
    assert count2 == stats2.chunks_added
    assert count2 < count1  # old chunks removed


def test_walker_skips_git_and_node_modules(tmp_path: Path, settings: Settings) -> None:
    folder = tmp_path / "proj"
    folder.mkdir()
    (folder / "README.md").write_text("project " * 100)
    (folder / ".git").mkdir()
    (folder / ".git" / "config").write_text("git config " * 50)
    (folder / "node_modules").mkdir()
    (folder / "node_modules" / "lib.js").write_text("module " * 50)

    stats = build_index(settings, paths=[folder], embedder=StubEmbedder())
    assert stats.files_indexed == 1


def test_unsupported_extension_is_skipped(tmp_path: Path, settings: Settings) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "blob.bin").write_bytes(b"\x00\x01\x02" * 100)
    (folder / "ok.md").write_text("hello world " * 50)
    stats = build_index(settings, paths=[folder], embedder=StubEmbedder())
    assert stats.files_indexed == 1


def test_empty_folders_returns_zero_stats(settings: Settings) -> None:
    stats = build_index(settings, paths=[], embedder=StubEmbedder())
    assert stats.files_scanned == 0
    assert stats.chunks_added == 0
