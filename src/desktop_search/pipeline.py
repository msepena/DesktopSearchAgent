"""End-to-end pipeline — implemented in M6."""

from typing import Literal, NamedTuple

from .config import Settings
from .retriever import Hit


class Response(NamedTuple):
    answer: str
    citations: list[str]
    hits: list[Hit]
    source: Literal["local", "web", "none"]


def ask(settings: Settings, question: str) -> Response:
    raise NotImplementedError("M6")
