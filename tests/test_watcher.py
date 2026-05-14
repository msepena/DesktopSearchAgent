import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from desktop_search.config import Settings
from desktop_search.indexer import build_index, get_collection
from desktop_search.watcher import DebouncedReindexer, should_index, watch_folders


def _event(src: str, is_dir: bool = False, dest: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(is_directory=is_dir, src_path=src, dest_path=dest)


class StubEmbedder:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


# --- DebouncedReindexer unit tests ---------------------------------------


def test_modified_fires_on_change() -> None:
    seen: list = []
    h = DebouncedReindexer(
        on_change=lambda p: seen.append(("change", p)),
        on_delete=lambda p: seen.append(("delete", p)),
        debounce_seconds=0.02,
    )
    h.on_modified(_event("/tmp/a.md"))
    h.flush()
    assert seen == [("change", Path("/tmp/a.md"))]


def test_deleted_fires_on_delete() -> None:
    seen: list = []
    h = DebouncedReindexer(
        on_change=lambda p: seen.append(("change", p)),
        on_delete=lambda p: seen.append(("delete", p)),
        debounce_seconds=0.02,
    )
    h.on_deleted(_event("/tmp/a.md"))
    h.flush()
    assert seen == [("delete", Path("/tmp/a.md"))]


def test_rapid_modifies_collapse_to_one_callback() -> None:
    seen: list = []
    h = DebouncedReindexer(
        on_change=lambda p: seen.append(("change", p)),
        on_delete=lambda p: seen.append(("delete", p)),
        debounce_seconds=0.05,
    )
    for _ in range(5):
        h.on_modified(_event("/tmp/a.md"))
    h.flush()
    assert seen == [("change", Path("/tmp/a.md"))]


def test_directory_events_are_ignored() -> None:
    seen: list = []
    h = DebouncedReindexer(
        on_change=lambda p: seen.append(p),
        on_delete=lambda p: seen.append(p),
        debounce_seconds=0.02,
    )
    h.on_modified(_event("/tmp/dir", is_dir=True))
    h.on_created(_event("/tmp/dir", is_dir=True))
    h.on_deleted(_event("/tmp/dir", is_dir=True))
    h.flush()
    assert seen == []


def test_moved_emits_delete_then_change() -> None:
    seen: list = []
    h = DebouncedReindexer(
        on_change=lambda p: seen.append(("change", p)),
        on_delete=lambda p: seen.append(("delete", p)),
        debounce_seconds=0.02,
    )
    h.on_moved(_event("/tmp/old.md", dest="/tmp/new.md"))
    h.flush()
    assert ("delete", Path("/tmp/old.md")) in seen
    assert ("change", Path("/tmp/new.md")) in seen


# --- path filter ---------------------------------------------------------


def test_should_index_filters_hidden_and_ignored() -> None:
    ignore = [".git", "node_modules"]
    assert should_index(Path("/x/notes/a.md"), ignore)
    assert not should_index(Path("/x/.git/config"), ignore)
    assert not should_index(Path("/x/node_modules/lib.js"), ignore)
    assert not should_index(Path("/x/.hidden.md"), ignore)


# --- integration: real watchdog Observer + stub embedder -----------------


def test_modified_file_triggers_reindex(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("desktop_search.watcher.FastEmbedEmbedder", StubEmbedder)
    monkeypatch.setattr("desktop_search.indexer.FastEmbedEmbedder", StubEmbedder)

    folder = tmp_path / "notes"
    folder.mkdir()
    f = folder / "a.md"
    f.write_text("initial content " * 200)

    settings = Settings(
        folders=[folder],
        chroma_path=tmp_path / "chroma",
        chunk_size=50,
        chunk_overlap=10,
    )
    build_index(settings, paths=[folder], embedder=StubEmbedder())
    coll = get_collection(settings.chroma_path)
    initial_count = coll.count()
    assert initial_count > 0

    observer = watch_folders(settings, paths=[folder], debounce_seconds=0.1)
    try:
        time.sleep(0.2)
        f.write_text("brand new shorter content")
        future = time.time() + 5
        os.utime(f, (future, future))
        # Poll up to 3s for the re-index to land.
        deadline = time.time() + 3.0
        while time.time() < deadline and coll.count() == initial_count:
            time.sleep(0.1)
    finally:
        observer.stop()
        observer.join()

    assert coll.count() != initial_count


def test_deleted_file_removes_chunks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("desktop_search.watcher.FastEmbedEmbedder", StubEmbedder)
    monkeypatch.setattr("desktop_search.indexer.FastEmbedEmbedder", StubEmbedder)

    folder = tmp_path / "notes"
    folder.mkdir()
    f = folder / "a.md"
    f.write_text("hello world " * 100)

    settings = Settings(
        folders=[folder],
        chroma_path=tmp_path / "chroma",
        chunk_size=50,
        chunk_overlap=10,
    )
    build_index(settings, paths=[folder], embedder=StubEmbedder())
    coll = get_collection(settings.chroma_path)
    assert coll.count() > 0

    observer = watch_folders(settings, paths=[folder], debounce_seconds=0.1)
    try:
        time.sleep(0.2)
        f.unlink()
        deadline = time.time() + 3.0
        while time.time() < deadline and coll.count() > 0:
            time.sleep(0.1)
    finally:
        observer.stop()
        observer.join()

    assert coll.count() == 0


def test_watch_folders_rejects_empty_folder_list(tmp_path: Path) -> None:
    settings = Settings(folders=[], chroma_path=tmp_path / "chroma")
    with pytest.raises(ValueError):
        watch_folders(settings, paths=[])
