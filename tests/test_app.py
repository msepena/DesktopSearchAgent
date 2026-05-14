from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parent.parent / "src" / "desktop_search" / "app.py"


def test_app_renders_without_exception(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        f"folders: []\nchroma_path: {tmp_path}/chroma\n"
    )
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    # Title is rendered
    assert any(
        "DesktopSearchAgent" in (t.value or "")
        for t in (at.title or [])
    )
    # Sidebar metric shows the chunk count
    assert any(
        getattr(m, "label", "") == "Chunks indexed"
        for m in (at.sidebar.metric or [])
    )


def test_app_sidebar_shows_model_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        f"folders: []\nchroma_path: {tmp_path}/chroma\nllm_model: claude-sonnet-4-6\n"
    )
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    sidebar_text = " ".join(
        str(getattr(el, "value", "") or getattr(el, "body", "") or "")
        for el in at.sidebar.markdown or []
    )
    assert "claude-sonnet-4-6" in sidebar_text
