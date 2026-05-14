import pytest

from desktop_search.web_search import DuckDuckGoSearch, WebResult, default_provider


class _FakeDDGS:
    """Replaces ddgs.DDGS in tests. Returns whatever raw results were configured."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results or []

    def __enter__(self):
        return self

    def __exit__(self, *_) -> None:
        return None

    def text(self, query: str, max_results: int = 5):
        return iter(self._results[:max_results])


def test_default_provider_is_duckduckgo() -> None:
    assert isinstance(default_provider(), DuckDuckGoSearch)


def test_search_normalises_href_to_url(monkeypatch) -> None:
    fake = _FakeDDGS([
        {"title": "Python", "href": "https://python.org", "body": "Welcome to Python."},
        {"title": "PyPI", "href": "https://pypi.org", "body": "Package index."},
    ])
    monkeypatch.setattr("ddgs.DDGS", lambda: fake)
    results = DuckDuckGoSearch().search("python", max_results=5)
    assert len(results) == 2
    assert results[0] == WebResult(
        title="Python", url="https://python.org", snippet="Welcome to Python."
    )


def test_search_accepts_url_key_too(monkeypatch) -> None:
    fake = _FakeDDGS([
        {"title": "Foo", "url": "https://foo.example", "snippet": "snippet"},
    ])
    monkeypatch.setattr("ddgs.DDGS", lambda: fake)
    results = DuckDuckGoSearch().search("foo")
    assert results[0].url == "https://foo.example"
    assert results[0].snippet == "snippet"


def test_search_skips_results_without_url(monkeypatch) -> None:
    fake = _FakeDDGS([
        {"title": "ok", "href": "https://ok.example", "body": "x"},
        {"title": "bad", "body": "no url"},  # filtered out
    ])
    monkeypatch.setattr("ddgs.DDGS", lambda: fake)
    results = DuckDuckGoSearch().search("q")
    assert len(results) == 1
    assert results[0].url == "https://ok.example"


def test_search_respects_max_results(monkeypatch) -> None:
    fake = _FakeDDGS([
        {"title": str(i), "href": f"https://x/{i}", "body": "b"} for i in range(20)
    ])
    monkeypatch.setattr("ddgs.DDGS", lambda: fake)
    results = DuckDuckGoSearch().search("q", max_results=3)
    assert len(results) == 3
