"""End-to-end pipeline: retrieval → confidence gate → answer composition.

The gate compares the top hit's score against ``settings.confidence_threshold``:
- top hit at or above the threshold → answer locally with Claude.
- top hit below the threshold → return the web-fallback stub. M10 will
  replace this with a real web search.
- no hits at all (empty index) → return the ``none`` placeholder.
"""

from typing import Literal, NamedTuple

import anthropic

from . import llm
from .config import Settings
from .indexer import Embedder
from .retriever import Hit, search


_NO_INDEX_MESSAGE = "No indexed content available to answer from."
_WEB_FALLBACK_MESSAGE = (
    "I couldn't find a confident match in your local files. "
    "Web search fallback is not yet implemented (see M10)."
)


class Response(NamedTuple):
    answer: str
    citations: list[str]
    hits: list[Hit]
    source: Literal["local", "web", "none"]


def ask(
    settings: Settings,
    question: str,
    embedder: Embedder | None = None,
    client: anthropic.Anthropic | None = None,
) -> Response:
    hits = search(settings, question, embedder=embedder)

    if not hits:
        return Response(answer=_NO_INDEX_MESSAGE, citations=[], hits=[], source="none")

    if hits[0].score < settings.confidence_threshold:
        return Response(answer=_WEB_FALLBACK_MESSAGE, citations=[], hits=hits, source="web")

    result = llm.answer(settings, question, hits, client=client)
    return Response(
        answer=result.text,
        citations=result.citations,
        hits=hits,
        source="local",
    )
