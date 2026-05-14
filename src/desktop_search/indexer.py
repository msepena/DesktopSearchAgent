"""Indexer — implemented in M3."""

from pathlib import Path

from .config import Settings


def build_index(settings: Settings, paths: list[Path] | None = None) -> int:
    raise NotImplementedError("M3")
