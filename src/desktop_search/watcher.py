"""File watcher that re-indexes individual files as they change.

Uses ``watchdog`` to observe the configured folders. Events are debounced so
a flurry of writes (typical when an editor saves) collapses into a single
re-index call. Deletions drop the file's chunks from Chroma.
"""

import threading
from pathlib import Path
from typing import Callable, Literal

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import Settings
from .indexer import (
    Embedder,
    FastEmbedEmbedder,
    get_collection,
    index_file,
    is_ignored_segment,
)


Action = Literal["change", "delete"]


class DebouncedReindexer(FileSystemEventHandler):
    """Coalesces rapid filesystem events per (path, action) pair."""

    def __init__(
        self,
        on_change: Callable[[Path], None],
        on_delete: Callable[[Path], None],
        debounce_seconds: float = 0.5,
    ) -> None:
        self._on_change = on_change
        self._on_delete = on_delete
        self._debounce_seconds = debounce_seconds
        self._lock = threading.Lock()
        self._pending: dict[tuple[Path, str], threading.Timer] = {}

    def _schedule(self, path: Path, action: Action) -> None:
        key = (path, action)
        with self._lock:
            existing = self._pending.pop(key, None)
        if existing is not None:
            existing.cancel()

        def fire() -> None:
            with self._lock:
                self._pending.pop(key, None)
            if action == "delete":
                self._on_delete(path)
            else:
                self._on_change(path)

        timer = threading.Timer(self._debounce_seconds, fire)
        with self._lock:
            self._pending[key] = timer
        timer.start()

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._schedule(Path(event.src_path), "change")

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self._schedule(Path(event.src_path), "change")

    def on_deleted(self, event) -> None:
        if not event.is_directory:
            self._schedule(Path(event.src_path), "delete")

    def on_moved(self, event) -> None:
        if not event.is_directory:
            self._schedule(Path(event.src_path), "delete")
            self._schedule(Path(event.dest_path), "change")

    def flush(self) -> None:
        """Block until any pending debounced callbacks have fired."""
        with self._lock:
            timers = list(self._pending.values())
        for t in timers:
            t.join()


def should_index(path: Path, ignore_patterns: list[str]) -> bool:
    ignored = set(ignore_patterns)
    return not any(is_ignored_segment(p, ignored) for p in path.parts)


def watch_folders(
    settings: Settings,
    paths: list[Path] | None = None,
    embedder: Embedder | None = None,
    debounce_seconds: float = 0.5,
) -> Observer:
    """Start a recursive watchdog Observer over the given folders.

    Returns the live ``Observer``. Callers should ``observer.stop()`` and
    ``observer.join()`` to shut down cleanly.
    """
    folders = list(paths) if paths else list(settings.folders)
    if not folders:
        raise ValueError("No folders to watch — set folders in config.yaml or pass --path.")

    collection = get_collection(settings.chroma_path)
    if embedder is None:
        embedder = FastEmbedEmbedder(settings.embedding_model)

    def on_change(p: Path) -> None:
        if not p.exists() or not should_index(p, settings.ignore_patterns):
            return
        index_file(p, collection, embedder, settings)

    def on_delete(p: Path) -> None:
        collection.delete(where={"source": str(p)})

    handler = DebouncedReindexer(on_change, on_delete, debounce_seconds)
    observer = Observer()
    for folder in folders:
        f = Path(folder)
        if f.is_dir():
            observer.schedule(handler, str(f), recursive=True)
    observer.start()
    return observer
