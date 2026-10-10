"""`bench new` / `bench update`. Rendering every answer combination runs in CI (generator job)."""

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench_cli import scaffold
from bench_cli.main import app

REPO = scaffold.find_root(Path(__file__).parent)
runner = CliRunner()

PYPROJECT = """\
[project]
name = "x"
dependencies = ["bench-cli", "bench-core", "bench-github"]

[tool.uv.workspace]
members = ["packages/*", "integrations/*"]

[tool.uv.sources]
bench-cli = { workspace = true }
bench-core = { workspace = true }
bench-github = { workspace = true }

[tool.ruff]
line-length = 100
"""


def sources_block(text: str) -> list[str]:
    lines = text.split("\n")
    start = lines.index("[tool.uv.sources]") + 1
    return lines[start : lines.index("", start)]


def test_register_member_keeps_dependencies_and_sources_sorted() -> None:
    text = scaffold.register_member(PYPROJECT, "drive")
    data = tomllib.loads(text)
    assert data["project"]["dependencies"] == [
        "bench-cli",
        "bench-core",
        "bench-drive",
        "bench-github",
    ]
    assert sources_block(text) == [
        "bench-cli = { workspace = true }",
        "bench-core = { workspace = true }",
        "bench-drive = { workspace = true }",
        "bench-github = { workspace = true }",
    ]
    assert text.endswith("[tool.ruff]\nline-length = 100\n")


def test_register_member_appends_last_and_is_idempotent() -> None:
    text = scaffold.register_member(PYPROJECT, "slack")
    assert sources_block(text)[-1] == "bench-slack = { workspace = true }"
    assert scaffold.register_member(text, "slack") == text


def test_register_member_wraps_long_dependency_lists() -> None:
    text = PYPROJECT
    for name in ("jira", "drive", "slack", "confluence", "linear"):
        text = scaffold.register_member(text, name)
    assert 'dependencies = [\n    "bench-cli",\n' in text
    assert all(len(line) <= 100 for line in text.splitlines())
    assert len(tomllib.loads(text)["project"]["dependencies"]) == 8


def test_register_member_needs_a_sources_table() -> None:
    text = PYPROJECT.replace("[tool.uv.sources]", "[tool.other]")
    with pytest.raises(scaffold.ScaffoldError, match=r"no \[tool.uv.sources\]"):
        scaffold.register_member(text, "jira")


@pytest.mark.parametrize("name", ["a", "Jira", "1jira", "my-api", "abcdefghijklmnop"])
def test_invalid_names_are_rejected(name: str) -> None:
    with pytest.raises(scaffold.ScaffoldError, match="invalid name"):
        scaffold.check_name(name)


def test_find_root_walks_up_to_the_workspace(tmp_path: Path) -> None:
    assert scaffold.find_root(REPO / "templates" / "integration") == REPO
    with pytest.raises(scaffold.ScaffoldError, match="not inside"):
        scaffold.find_root(tmp_path)


def test_new_refuses_an_existing_integration() -> None:
    result = runner.invoke(app, ["new", "github", "--defaults", "--no-sync"])
    assert result.exit_code == 1
    assert "integrations/github already exists" in result.output


def test_new_rejects_malformed_data() -> None:
    result = runner.invoke(app, ["new", "jira", "--data", "auth"])
    assert result.exit_code == 2
    assert "expected KEY=VALUE" in result.output


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repo with just the template and a workspace pyproject."""
    for name in ("copier.yml", ".gitattributes"):
        shutil.copy(REPO / name, tmp_path)
    shutil.copytree(REPO / "templates", tmp_path / "templates")
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, "utf-8", newline="\n")
    (tmp_path / "integrations").mkdir()
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "template")
    return tmp_path


def test_generate_then_update_merges_template_changes(repo: Path) -> None:
    data = {"auth": "oauth_refresh", "pagination": "cursor"}
    dst = scaffold.generate(repo, "zed", data, defaults=True)
    answers = (dst / ".copier-answers.yml").read_text("utf-8")
    assert "_src_path: .\n" in answers
    assert "auth: oauth_refresh\n" in answers
    assert (
        "bench-zed"
        in tomllib.loads((repo / "pyproject.toml").read_text("utf-8"))["project"]["dependencies"]
    )
    assert (dst / "src" / "bench_zed" / "auth.py").is_file()
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "generate")

    # A local edit to the integration and a later template change.
    tools = dst / "src" / "bench_zed" / "tools.py"
    tools.write_text(tools.read_text("utf-8") + "# local edit\n", "utf-8", newline="\n")
    scopes = repo / "templates/integration/template/src/bench_{{name}}/scopes.py.jinja"
    scopes.write_text(
        scopes.read_text("utf-8").replace('"""', '"""Template v2. ', 1), "utf-8", newline="\n"
    )
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "template v2")

    scaffold.update(repo, "zed", vcs_ref=None, defaults=True)
    assert (
        (dst / "src" / "bench_zed" / "scopes.py").read_text("utf-8").startswith('"""Template v2. ')
    )
    assert tools.read_text("utf-8").endswith("# local edit\n")
    assert "auth: oauth_refresh\n" in (dst / ".copier-answers.yml").read_text("utf-8")


def test_uncommitted_template_changes_are_refused(repo: Path) -> None:
    (repo / "templates" / "integration" / "questions.yml").write_text("{}\n", "utf-8")
    with pytest.raises(scaffold.ScaffoldError, match="uncommitted changes"):
        scaffold.generate(repo, "zed", {}, defaults=True)
    assert not (repo / "integrations" / "zed").exists()
