"""File loaders — implemented in M2."""

from pathlib import Path
from typing import NamedTuple


class Document(NamedTuple):
    text: str
    source: Path
    section: str | None


def load_file(path: Path) -> list[Document]:
    raise NotImplementedError("M2")
