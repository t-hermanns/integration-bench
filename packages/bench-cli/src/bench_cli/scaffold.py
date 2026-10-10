"""`bench new` / `bench update`: generate integrations from the copier template.

copier runs from the repo root with the template source ".", so the answers file records
`_src_path: .` plus the template commit, and `copier update` (via `bench update`) can diff
later template changes against it. copier is pinned to the committed HEAD: given the ref
"HEAD" it renders uncommitted changes and records a commit that only existed in its
temporary clone.
"""

import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping
from contextlib import chdir
from pathlib import Path

import copier

NAME_RE = re.compile(r"^[a-z][a-z0-9]{1,14}$")
TEMPLATE_PATHS = ("copier.yml", "templates/integration")
LINE_LENGTH = 100


class ScaffoldError(Exception):
    """A problem the user can fix; the message says how."""


def find_root(start: Path) -> Path:
    """The repo root: the nearest directory with `copier.yml` and a uv workspace."""
    for path in (start, *start.parents):
        pyproject = path / "pyproject.toml"
        if not ((path / "copier.yml").is_file() and pyproject.is_file()):
            continue
        uv = tomllib.loads(pyproject.read_text("utf-8")).get("tool", {}).get("uv", {})
        if "workspace" in uv:
            return path
    raise ScaffoldError(f"{start} is not inside the integration-bench repo")


def check_name(name: str) -> None:
    if not NAME_RE.match(name):
        raise ScaffoldError(
            f"invalid name {name!r}: use 2-15 lowercase letters or digits, starting with a letter"
        )


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def template_changes(root: Path) -> list[str]:
    """Uncommitted changes to the template (`git status --porcelain` lines)."""
    return git(root, "status", "--porcelain", "--", *TEMPLATE_PATHS).splitlines()


def committed_template(root: Path) -> str:
    """The HEAD commit to render from; refuses uncommitted template changes."""
    if changes := template_changes(root):
        raise ScaffoldError(
            "the template has uncommitted changes; commit them first (`bench new` also takes "
            "--allow-dirty while developing the template).\n" + "\n".join(changes)
        )
    return git(root, "rev-parse", "HEAD").strip()


def register_member(pyproject: str, name: str) -> str:
    """Add `bench-<name>` to the root dependencies and workspace sources (kept sorted)."""
    package = f"bench-{name}"
    data = tomllib.loads(pyproject)
    deps: list[str] = data["project"]["dependencies"]
    if package not in deps:
        items = ", ".join(f'"{d}"' for d in sorted([*deps, package]))
        line = f"dependencies = [{items}]"
        if len(line) > LINE_LENGTH:
            line = "\n".join(
                ["dependencies = [", *(f'    "{d}",' for d in sorted([*deps, package])), "]"]
            )
        pyproject, found = re.subn(
            r"^dependencies = \[.*?\]$", line, pyproject, count=1, flags=re.M | re.S
        )
        if not found:
            raise ScaffoldError("root pyproject.toml: no `dependencies = [...]` in [project]")
    if package not in data["tool"]["uv"].get("sources", {}):
        lines = pyproject.split("\n")
        try:
            start = lines.index("[tool.uv.sources]") + 1
        except ValueError:
            raise ScaffoldError("root pyproject.toml: no [tool.uv.sources] table") from None
        end = start
        while end < len(lines) and lines[end].strip() and not lines[end].startswith("["):
            end += 1
        at = next((i for i in range(start, end) if lines[i].split("=")[0].strip() > package), end)
        lines.insert(at, f"{package} = {{ workspace = true }}")
        pyproject = "\n".join(lines)
    check = tomllib.loads(pyproject)
    if package not in check["project"]["dependencies"] or check["tool"]["uv"]["sources"].get(
        package
    ) != {"workspace": True}:
        raise ScaffoldError(f"root pyproject.toml: could not register {package}")
    return pyproject


def generate(
    root: Path,
    name: str,
    data: Mapping[str, str],
    *,
    defaults: bool,
    allow_dirty: bool = False,
) -> Path:
    """Render `integrations/<name>` and register it in the root pyproject.toml."""
    check_name(name)
    dst = root / "integrations" / name
    if dst.exists():
        raise ScaffoldError(f"{dst.relative_to(root).as_posix()} already exists")
    # "HEAD" renders the working tree, but the recorded commit then cannot be updated from.
    ref = "HEAD" if allow_dirty else committed_template(root)
    with chdir(root):
        copier.run_copy(
            ".",
            f"integrations/{name}",
            data={**data, "name": name},
            vcs_ref=ref,
            defaults=defaults,
            unsafe=False,
        )
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text("utf-8")
    pyproject.write_text(register_member(text, name), "utf-8", newline="\n")
    return dst


def update(root: Path, name: str, *, vcs_ref: str | None, defaults: bool) -> None:
    """Pull later template changes into `integrations/<name>` (three-way merge)."""
    check_name(name)
    dst = root / "integrations" / name
    if not (dst / ".copier-answers.yml").is_file():
        raise ScaffoldError(f"integrations/{name} was not generated from the template")
    ref = vcs_ref or committed_template(root)
    with chdir(root):
        copier.run_update(
            f"integrations/{name}",
            vcs_ref=ref,
            defaults=defaults,
            overwrite=True,
            skip_answered=True,
            conflict="inline",
            unsafe=False,
        )


def uv_command() -> list[str]:
    """uv sets `UV` for `uv run`; otherwise look on PATH."""
    uv = os.environ.get("UV") or shutil.which("uv")
    if uv is None:
        raise ScaffoldError("uv not found; run `uv sync` and `uv run bench synth <name>` yourself")
    return [uv]
