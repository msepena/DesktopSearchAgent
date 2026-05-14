"""Claude wrapper — implemented in M5."""

from typing import NamedTuple

from .config import Settings
from .retriever import Hit


class Answer(NamedTuple):
    text: str
    citations: list[str]


def answer(settings: Settings, question: str, context: list[Hit]) -> Answer:
    raise NotImplementedError("M5")
