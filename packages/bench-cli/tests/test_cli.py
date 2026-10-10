"""`bench` CLI wiring (the synthesis itself is tested in bench-core)."""

from typer.testing import CliRunner

from bench_cli.main import app

runner = CliRunner()


def test_synth_check_passes_for_committed_github_cassettes() -> None:
    result = runner.invoke(app, ["synth", "github", "--check"])
    assert result.exit_code == 0, result.output
    assert "up to date" in result.output


def test_unknown_integration_is_a_usage_error() -> None:
    result = runner.invoke(app, ["synth", "nope"])
    assert result.exit_code == 2
    assert "no integration 'nope'" in result.output


def test_unknown_scenario_is_a_usage_error() -> None:
    result = runner.invoke(app, ["synth", "github", "--only", "nope"])
    assert result.exit_code == 2
    assert "unknown scenario" in result.output
