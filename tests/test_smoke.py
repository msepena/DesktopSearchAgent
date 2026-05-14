from typer.testing import CliRunner

from desktop_search import __version__
from desktop_search.cli import app


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"


def test_cli_help_lists_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "ask", "ui", "version"):
        assert cmd in result.stdout


def test_cli_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
