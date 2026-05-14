"""End-to-end pipeline: retrieval → confidence gate → answer composition.

The gate compares the top hit's score against ``settings.confidence_threshold``:
- top hit at or above the threshold → answer locally with Claude.
- top hit below the threshold → run web search (DuckDuckGo) and answer from
  those results, citing URLs.
- no hits at all (empty index) → return the ``none`` placeholder.
"""

from typing import Literal, NamedTuple

import anthropic

from . import llm
from .config import Settings
from .indexer import Embedder
from .retriever import Hit, search
from .web_search import WebSearchProvider, default_provider


_NO_INDEX_MESSAGE = "No indexed content available to answer from."
_WEB_NO_RESULTS_MESSAGE = (
    "Local match below the confidence threshold and the web search "
    "returned no results."
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
    web_provider: WebSearchProvider | None = None,
) -> Response:
    hits = search(settings, question, embedder=embedder)

    if not hits:
        return Response(answer=_NO_INDEX_MESSAGE, citations=[], hits=[], source="none")

    if hits[0].score < settings.confidence_threshold:
        provider = web_provider if web_provider is not None else default_provider()
        try:
            web_results = provider.search(question, max_results=settings.top_k)
        except Exception as exc:
            return Response(
                answer=f"Local match below threshold; web fallback failed: {exc}",
                citations=[],
                hits=hits,
                source="web",
            )
        if not web_results:
            return Response(
                answer=_WEB_NO_RESULTS_MESSAGE,
                citations=[],
                hits=hits,
                source="web",
            )
        web_answer = llm.answer_from_web(settings, question, web_results, client=client)
        return Response(
            answer=web_answer.text,
            citations=web_answer.citations,
            hits=hits,
            source="web",
        )

    result = llm.answer(settings, question, hits, client=client)
    return Response(
        answer=result.text,
        citations=result.citations,
        hits=hits,
        source="local",
    )
