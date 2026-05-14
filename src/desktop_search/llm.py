"""Claude wrapper.

Builds a single-turn message with the retrieved context, asks Claude to
answer using only that context, and extracts inline ``[source: ...]``
citations from the response. The static system block is sent with
``cache_control: ephemeral`` so repeated questions in a session benefit
from prompt caching.
"""

import re
from typing import NamedTuple

import anthropic

from .config import Settings
from .retriever import Hit
from .web_search import WebResult


_CITATION_RE = re.compile(r"\[source:\s*([^\]]+)\]")

_SYSTEM_PROMPT = (
    "You are DesktopSearchAgent, a careful assistant that answers questions "
    "using ONLY the provided context snippets from the user's local files. "
    "If the answer cannot be derived from the snippets, say so plainly.\n\n"
    "Rules:\n"
    "- Be concise.\n"
    "- After every claim supported by a snippet, add an inline citation in "
    "the exact format: [source: <path> <section>]. Copy the path and section "
    "as given in the context header.\n"
    "- If multiple snippets support a claim, add multiple citations.\n"
    "- Do not fabricate citations or refer to files that are not in the "
    "context.\n"
)

_WEB_SYSTEM_PROMPT = (
    "You are DesktopSearchAgent. The user's local files did not contain a "
    "confident answer, so you have been given web search results instead.\n\n"
    "Rules:\n"
    "- Use ONLY the snippets below; do not invent facts not in them.\n"
    "- Be concise.\n"
    "- After every claim supported by a snippet, add an inline citation in "
    "the exact format: [source: <url>]. Copy the URL exactly as it appears in "
    "the context header.\n"
    "- If multiple snippets support a claim, add multiple citations.\n"
    "- If the snippets don't answer the question, say so plainly.\n"
)


class Answer(NamedTuple):
    text: str
    citations: list[str]


def _format_context(hits: list[Hit]) -> str:
    if not hits:
        return "(no context provided)"
    parts = []
    for i, h in enumerate(hits, start=1):
        header = f"[{i}] source: {h.source}"
        if h.section:
            header += f" {h.section}"
        parts.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(parts)


def _format_web_context(results: list[WebResult]) -> str:
    if not results:
        return "(no web results)"
    parts = []
    for i, r in enumerate(results, start=1):
        header = f"[{i}] source: {r.url}"
        if r.title:
            header += f"\nTitle: {r.title}"
        parts.append(f"{header}\n{r.snippet}")
    return "\n\n---\n\n".join(parts)


def _extract_citations(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _CITATION_RE.finditer(text):
        cite = m.group(0)
        if cite not in seen:
            seen.add(cite)
            out.append(cite)
    return out


def _response_text(response) -> str:
    parts: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def answer(
    settings: Settings,
    question: str,
    context: list[Hit],
    client: anthropic.Anthropic | None = None,
    max_tokens: int = 1024,
) -> Answer:
    if client is None:
        client = anthropic.Anthropic()

    user_content = (
        f"Context snippets:\n\n{_format_context(context)}\n\n"
        f"---\n\nQuestion: {question}"
    )

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    text = _response_text(response)
    return Answer(text=text, citations=_extract_citations(text))


def answer_from_web(
    settings: Settings,
    question: str,
    results: list[WebResult],
    client: anthropic.Anthropic | None = None,
    max_tokens: int = 1024,
) -> Answer:
    if client is None:
        client = anthropic.Anthropic()

    user_content = (
        f"Web search results:\n\n{_format_web_context(results)}\n\n"
        f"---\n\nQuestion: {question}"
    )

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": _WEB_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    text = _response_text(response)
    return Answer(text=text, citations=_extract_citations(text))
