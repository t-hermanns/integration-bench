"""The orphan branch `bench-results` (call budget now, nightly results later), as a git worktree.

The worktree lives in `.bench/bench-results` (gitignored). `checkout` creates or updates it,
`publish` commits and pushes; both shell out to git with the remote `origin`.
"""

import subprocess
from pathlib import Path

BRANCH = "bench-results"


class ResultsBranchError(Exception):
    pass


def git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", check=False
    )
    if result.returncode != 0:
        raise ResultsBranchError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def remote_has_branch(root: Path) -> bool:
    return bool(git("ls-remote", "--heads", "origin", BRANCH, cwd=root).strip())


def checkout(root: Path) -> Path:
    """The worktree of `bench-results`, up to date with origin (created as orphan if new)."""
    tree = root / ".bench" / BRANCH
    on_remote = remote_has_branch(root)
    if on_remote:
        git("fetch", "origin", BRANCH, cwd=root)
    if not (tree / ".git").exists():
        if on_remote:
            git("worktree", "add", "-B", BRANCH, str(tree), f"origin/{BRANCH}", cwd=root)
        else:
            git("worktree", "add", "--orphan", "-b", BRANCH, str(tree), cwd=root)
    elif on_remote:
        git("merge", "--ff-only", f"origin/{BRANCH}", cwd=tree)
    return tree


def publish(tree: Path, paths: list[str], message: str) -> bool:
    """Commit `paths` (relative to the worktree) and push; False when nothing changed."""
    git("add", "--", *paths, cwd=tree)
    if not git("status", "--porcelain", "--", *paths, cwd=tree).strip():
        return False
    git("commit", "-m", message, "--", *paths, cwd=tree)
    git("push", "-u", "origin", BRANCH, cwd=tree)
    return True
