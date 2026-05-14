import subprocess
import sys
from pathlib import Path

import typer

from . import __version__
from .config import load_settings
from .indexer import build_index
from .pipeline import ask as run_pipeline
from .watcher import watch_folders


app = typer.Typer(
    help="DesktopSearchAgent — local RAG over your laptop files.",
    no_args_is_help=True,
)


_CONFIG_OPTION = typer.Option(
    Path("config.yaml"),
    "--config",
    "-c",
    help="Path to config.yaml.",
)


@app.command()
def index(
    path: list[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Folder to index. Repeatable. Overrides folders in config.yaml.",
    ),
    watch: bool = typer.Option(
        False,
        "--watch",
        "-w",
        help="After the initial index, keep running and re-index on file changes.",
    ),
    config: Path = _CONFIG_OPTION,
) -> None:
    """Build or refresh the local index."""
    settings = load_settings(config)
    override = path if path else None
    stats = build_index(settings, paths=override)
    typer.echo(f"Scanned:      {stats.files_scanned}")
    typer.echo(f"Indexed:      {stats.files_indexed}")
    typer.echo(f"Skipped:      {stats.files_skipped} (unchanged or empty)")
    typer.echo(f"Chunks added: {stats.chunks_added}")

    if not watch:
        return

    typer.echo("\nWatching for changes... (Ctrl+C to stop)")
    observer = watch_folders(settings, paths=override)
    try:
        while observer.is_alive():
            observer.join(1)
    except KeyboardInterrupt:
        typer.echo("\nStopping watcher.")
    finally:
        observer.stop()
        observer.join()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question"),
    config: Path = _CONFIG_OPTION,
) -> None:
    """Ask a question against the index."""
    settings = load_settings(config)
    response = run_pipeline(settings, question)

    typer.echo(response.answer)

    if response.hits:
        typer.echo("")
        typer.echo("Sources:")
        for h in response.hits:
            label = f"  - {h.source}"
            if h.section:
                label += f" ({h.section})"
            label += f"  [score={h.score:.2f}]"
            typer.echo(label)

    if response.source != "local":
        typer.echo("")
        typer.echo(f"(answer source: {response.source})")


@app.command()
def ui(config: Path = _CONFIG_OPTION) -> None:
    """Launch the Streamlit chat UI."""
    app_module = Path(__file__).parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_module)]
    typer.echo(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
