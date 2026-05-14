from pathlib import Path

import pytest
from typer.testing import CliRunner

from desktop_search.cli import app
from desktop_search.indexer import IndexStats
from desktop_search.pipeline import Response
from desktop_search.retriever import Hit


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_config(tmp_path: Path, folders: list[str] | None = None) -> Path:
    cfg = tmp_path / "config.yaml"
    lines = ["folders:"]
    for f in folders or []:
        lines.append(f"  - {f}")
    if not folders:
        lines = ["folders: []"]
    cfg.write_text("\n".join(lines) + "\n")
    return cfg


def test_index_prints_stats(monkeypatch, tmp_path: Path, runner: CliRunner) -> None:
    cfg = _write_config(tmp_path)
    monkeypatch.setattr(
        "desktop_search.cli.build_index",
        lambda settings, paths=None, embedder=None: IndexStats(
            files_scanned=5, files_indexed=3, files_skipped=2, chunks_added=17
        ),
    )
    result = runner.invoke(app, ["index", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "Scanned:      5" in result.stdout
    assert "Indexed:      3" in result.stdout
    assert "Skipped:      2" in result.stdout
    assert "Chunks added: 17" in result.stdout


def test_index_path_flag_overrides_config(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    cfg = _write_config(tmp_path, folders=["/will/be/overridden"])
    captured: dict = {}

    def fake(settings, paths=None, embedder=None):
        captured["paths"] = paths
        return IndexStats()

    monkeypatch.setattr("desktop_search.cli.build_index", fake)
    result = runner.invoke(
        app,
        ["index", "--config", str(cfg), "--path", "/a", "--path", "/b"],
    )
    assert result.exit_code == 0
    assert captured["paths"] == [Path("/a"), Path("/b")]


def test_ask_prints_answer_sources_and_score(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    cfg = _write_config(tmp_path)
    fake_response = Response(
        answer="The answer is X [source: /x/a.md p.1].",
        citations=["[source: /x/a.md p.1]"],
        hits=[Hit(text="text", source=Path("/x/a.md"), section="p.1", score=0.85)],
        source="local",
    )
    monkeypatch.setattr(
        "desktop_search.cli.run_pipeline",
        lambda s, q: fake_response,
    )
    result = runner.invoke(app, ["ask", "what is X?", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "The answer is X" in result.stdout
    assert "/x/a.md" in result.stdout
    assert "p.1" in result.stdout
    assert "0.85" in result.stdout
    # local source: no trailing source label
    assert "(answer source:" not in result.stdout


def test_ask_shows_source_label_for_non_local_branches(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    cfg = _write_config(tmp_path)
    fake_response = Response(
        answer="No confident match.",
        citations=[],
        hits=[Hit(text="t", source=Path("/x/a.md"), section=None, score=0.2)],
        source="web",
    )
    monkeypatch.setattr(
        "desktop_search.cli.run_pipeline",
        lambda s, q: fake_response,
    )
    result = runner.invoke(app, ["ask", "q", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "(answer source: web)" in result.stdout


def test_ui_launches_streamlit(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    cfg = _write_config(tmp_path)
    captured: dict = {}

    def fake_run(cmd, check=False):
        captured["cmd"] = cmd
        return None

    monkeypatch.setattr("desktop_search.cli.subprocess.run", fake_run)
    result = runner.invoke(app, ["ui", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "streamlit" in captured["cmd"]
    assert "run" in captured["cmd"]
    assert any(c.endswith("app.py") for c in captured["cmd"])


def test_end_to_end_index_then_ask(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    """Index a small fixture folder, ask a question, get a cited answer."""

    # Stub the embedder on both modules so retriever and indexer share it.
    class StubEmbedder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr("desktop_search.indexer.FastEmbedEmbedder", StubEmbedder)
    monkeypatch.setattr("desktop_search.retriever.FastEmbedEmbedder", StubEmbedder)

    # Stub the Anthropic client so llm.answer doesn't touch the network.
    class FakeBlock:
        type = "text"

        def __init__(self, t: str) -> None:
            self.text = t

    class FakeResponse:
        def __init__(self, t: str) -> None:
            self.content = [FakeBlock(t)]

    class FakeMessages:
        def create(self, **kwargs):
            return FakeResponse(
                "From the notes [source: " + str(notes_dir / "a.md") + "]."
            )

    class FakeAnthropic:
        def __init__(self, *args, **kwargs) -> None:
            self.messages = FakeMessages()

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "a.md").write_text("hello world from notes " * 30)

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"folders:\n  - {notes_dir}\n"
        f"chroma_path: {tmp_path}/chroma\n"
        f"chunk_size: 80\n"
        f"chunk_overlap: 10\n"
        f"confidence_threshold: 0.0\n"
    )

    result = runner.invoke(app, ["index", "--config", str(cfg)])
    assert result.exit_code == 0, result.stdout
    assert "Indexed:      1" in result.stdout

    result = runner.invoke(app, ["ask", "what's in the notes?", "--config", str(cfg)])
    assert result.exit_code == 0, result.stdout
    assert "From the notes" in result.stdout
    assert "a.md" in result.stdout
