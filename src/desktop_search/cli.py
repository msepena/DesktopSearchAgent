from pathlib import Path

import typer

from . import __version__

app = typer.Typer(help="DesktopSearchAgent — local RAG over your laptop files.", no_args_is_help=True)


@app.command()
def index(
    path: list[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Folder to index. Repeatable. Overrides config.yaml folders.",
    ),
) -> None:
    """Build or refresh the local index (M3)."""
    typer.echo("index: not implemented yet (M3)")


@app.command()
def ask(question: str = typer.Argument(..., help="Natural-language question")) -> None:
    """Ask a question against the index (M6/M7)."""
    typer.echo(f"ask: not implemented yet (M6). question={question!r}")


@app.command()
def ui() -> None:
    """Launch the Streamlit chat UI (M8)."""
    typer.echo("ui: not implemented yet (M8)")


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
