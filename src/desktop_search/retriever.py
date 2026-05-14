"""Retriever — implemented in M4."""

from pathlib import Path
from typing import NamedTuple

from .config import Settings


class Hit(NamedTuple):
    text: str
    source: Path
    section: str | None
    score: float


def search(settings: Settings, query: str, k: int | None = None) -> list[Hit]:
    raise NotImplementedError("M4")
