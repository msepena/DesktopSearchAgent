"""Web search fallback. DuckDuckGo via the ``ddgs`` library — no API key required."""

from typing import NamedTuple, Protocol


class WebResult(NamedTuple):
    title: str
    url: str
    snippet: str


class WebSearchProvider(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[WebResult]: ...


class DuckDuckGoSearch:
    """Thin wrapper over the ``ddgs`` text search endpoint."""

    def search(self, query: str, max_results: int = 5) -> list[WebResult]:
        from ddgs import DDGS  # lazy import — keeps test startup cheap

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))

        results: list[WebResult] = []
        for r in raw:
            url = r.get("href") or r.get("url") or ""
            if not url:
                continue
            results.append(
                WebResult(
                    title=r.get("title", ""),
                    url=url,
                    snippet=r.get("body") or r.get("snippet") or "",
                )
            )
        return results


def default_provider() -> WebSearchProvider:
    return DuckDuckGoSearch()
