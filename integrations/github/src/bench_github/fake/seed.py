"""Default data for the fake: one sandbox repository shaped like the live test repo.

`bench sandbox seed` (later) creates the same issues, pull requests and files in the real
test organization, so scenarios and their assertions hold for synthetic and recorded
cassettes alike. Keep both in sync when changing this data.
"""

from datetime import UTC, datetime, timedelta

from bench_core.clock import Clock
from bench_core.testing import FakeClock

from bench_github.fake.backend import GitHubFake, Repo

OWNER = "bench-sandbox"
REPO = "widgets"
MAINTAINER = "bench-maintainer"
CONTRIBUTOR = "bench-contributor"
BOT = "bench-bot[bot]"

_START = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)


def at(hours: int) -> str:
    return (_START + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


# (title, labels, state, body)
ISSUES: list[tuple[str, list[str], str, str]] = [
    (
        "Crash when config file is empty",
        ["bug"],
        "open",
        "Starting with an empty `widgets.toml` raises `KeyError: 'name'`.",
    ),
    (
        "Add dark mode to the dashboard",
        ["enhancement"],
        "open",
        "Users asked for a dark theme in the web dashboard.",
    ),
    (
        "Timeout when exporting large reports",
        ["bug", "performance"],
        "open",
        "Exports with more than 10k rows time out after 30 s.",
    ),
    (
        "Document the plugin API",
        ["documentation"],
        "open",
        "The plugin hooks are undocumented; add a guide with examples.",
    ),
    (
        "Login button misaligned on mobile",
        ["bug", "ui"],
        "closed",
        "The login button overlaps the footer below 380 px width.",
    ),
    (
        "Support Python 3.14",
        ["enhancement"],
        "closed",
        "Run the test suite on 3.14 and fix deprecation warnings.",
    ),
    (
        "Memory leak in the widget cache",
        ["bug", "performance"],
        "open",
        "The cache never evicts entries; memory grows by ~50 MB per hour.",
    ),
    (
        "Typo in installation guide",
        ["documentation", "good first issue"],
        "open",
        "`pip instal widgets` should be `pip install widgets`.",
    ),
    (
        "Rate limit errors from the sync job",
        ["bug"],
        "open",
        "The nightly sync hits the upstream rate limit and aborts instead of waiting.",
    ),
    (
        "Add CSV import",
        ["enhancement"],
        "open",
        "Allow importing widgets from a CSV file with a header row.",
    ),
    (
        "Flaky test test_scheduler_restart",
        ["bug", "ci"],
        "closed",
        "Fails about once in 20 runs on Windows.",
    ),
    (
        "Crash on startup with Unicode paths",
        ["bug"],
        "open",
        "Paths like `C:/Users/Jörg/widgets` crash the loader on startup.",
    ),
]

COMMENTS: dict[int, list[tuple[str, str]]] = {
    1: [
        (CONTRIBUTOR, "I can reproduce this on 2.3.1."),
        (MAINTAINER, "Thanks! We should fall back to defaults when the file is empty."),
        (CONTRIBUTOR, "I opened a pull request for the fallback."),
    ],
    3: [(MAINTAINER, "Streaming the export instead of building it in memory should fix this.")],
    7: [(CONTRIBUTOR, "Heap profile attached: the cache dict holds every widget ever loaded.")],
}

# (title, author, head, state, draft, merged)
PULLS: list[tuple[str, str, str, str, bool, bool]] = [
    (
        "Fall back to defaults for empty config",
        CONTRIBUTOR,
        "fix/empty-config",
        "open",
        False,
        False,
    ),
    ("Stream report exports", MAINTAINER, "perf/stream-export", "open", True, False),
    ("Fix login button layout", CONTRIBUTOR, "fix/login-mobile", "closed", False, True),
    ("Experiment: new cache backend", CONTRIBUTOR, "exp/cache", "closed", False, False),
]

README = """# widgets

A small library for managing widgets.

## Install

    pip install widgets

## Usage

    import widgets
    widgets.load("widgets.toml")
"""

CORE = '''"""Widget loading."""


def load(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return parse(f.read())
'''

PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000")


def seed(fake: GitHubFake) -> Repo:
    repo = fake.add_repo(OWNER, REPO, collaborators=[MAINTAINER, CONTRIBUTOR])
    fake.known_users.add(BOT)
    hour = 0
    for title, labels, state, body in ISSUES:
        hour += 5
        fake.add_issue(
            repo,
            title,
            author=CONTRIBUTOR if hour % 2 else MAINTAINER,
            body=body,
            labels=labels,
            assignees=[MAINTAINER] if "bug" in labels and state == "open" else [],
            state=state,
            created_at=at(hour),
            updated_at=at(hour + 2),
            milestone="v2.4" if "bug" in labels else None,
        )
    for number, comments in COMMENTS.items():
        for i, (author, text) in enumerate(comments):
            fake.add_comment(
                repo, number, text, author=author, created_at=at(100 + 10 * number + i)
            )
    for title, author, head, state, draft, merged in PULLS:
        hour += 7
        fake.add_pull(
            repo,
            title,
            author=author,
            head=head,
            state=state,
            draft=draft,
            merged=merged,
            created_at=at(hour),
            updated_at=at(hour + 30),
        )
    fake.add_file(repo, "README.md", README)
    fake.add_file(repo, "src/widgets/__init__.py", '"""widgets."""\n\nfrom .core import load\n')
    fake.add_file(repo, "src/widgets/core.py", CORE)
    fake.add_file(repo, "docs/user guide.md", "# User guide\n\nSee the README for basics.\n")
    fake.add_file(repo, "assets/logo.png", PNG)
    fake.add_file(repo, "data/large.csv", b"id,name\n" + b"1,widget\n" * 130_000)
    return repo


def seeded_fake(clock: Clock | None = None, *, require_auth: bool = True) -> GitHubFake:
    fake = GitHubFake(require_auth=require_auth, clock=clock or FakeClock())
    seed(fake)
    return fake
