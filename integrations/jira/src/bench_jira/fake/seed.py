"""Default data for the fake, shaped like the live sandbox.

`bench sandbox seed jira` creates the same issues in project BENCH of the test site,
so the scenarios and their assertions hold for synthetic and recorded cassettes alike.
Keep both in sync when changing this data. Live issue keys depend on creation order:
create them in list order into an empty project.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from bench_core.clock import Clock
from bench_core.testing import FakeClock

from bench_jira.fake.backend import BOT, JiraFake

_START = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)

ALEX = "557058:alex-doe"
SAM = "557058:sam-lee"
USERS = {ALEX: "Alex Doe", SAM: "Sam Lee"}


@dataclass(frozen=True)
class SeedIssue:
    summary: str
    issue_type: str = "Task"
    status: str = "To Do"
    priority: str = "Medium"
    labels: tuple[str, ...] = ()
    assignee: str | None = None
    description: str | None = None
    comments: tuple[tuple[str, str], ...] = ()
    """(author account id, text)"""


ISSUES: list[SeedIssue] = [
    SeedIssue(  # BENCH-1
        "Checkout crashes when the cart is empty",
        issue_type="Bug",
        priority="Highest",
        labels=("checkout", "crash"),
        assignee=ALEX,
        description=(
            "Steps to reproduce:\n"
            "1. Open the shop\n"
            "2. Go to checkout without adding items\n\n"
            "The page shows a 500 error: KeyError: 'total'."
        ),
        comments=(
            (SAM, "Seen in production since the release on Monday."),
            (ALEX, "Found it: the total is only set once an item is added."),
        ),
    ),
    SeedIssue(  # BENCH-2
        "Add dark mode to the dashboard",
        issue_type="Story",
        labels=("ui",),
        description="Users asked for a dark theme. Follow the system setting by default.",
    ),
    SeedIssue(  # BENCH-3
        "Export to CSV times out for large reports",
        issue_type="Bug",
        status="In Progress",
        priority="High",
        labels=("export", "performance"),
        assignee=SAM,
        description="Exports over 10k rows time out after 30 s.",
        comments=((SAM, "Streaming the rows instead of building the file in memory."),),
    ),
    SeedIssue(  # BENCH-4
        "Document the plugin API",
        labels=("docs",),
        priority="Low",
    ),
    SeedIssue(  # BENCH-5
        "Login fails with SSO for new accounts",
        issue_type="Bug",
        status="Done",
        priority="High",
        labels=("auth",),
        assignee=ALEX,
        description="Fixed by renewing the IdP certificate.",
    ),
    SeedIssue(  # BENCH-6
        "Upgrade to the new HTTP client",
        status="Done",
        assignee=BOT["accountId"],
        labels=("tech-debt",),
    ),
    SeedIssue(  # BENCH-7
        "App crashes on startup without network",
        issue_type="Bug",
        priority="High",
        labels=("crash", "offline"),
        description="There is no offline fallback for the update check.",
    ),
    SeedIssue(  # BENCH-8
        "Translate the settings page",
        issue_type="Story",
        status="In Progress",
        priority="Low",
        labels=("i18n", "ui"),
        assignee=BOT["accountId"],
        description="German and French first.",
    ),
    SeedIssue(  # BENCH-9
        "Flaky test in the export suite",
        status="Done",
        priority="Low",
        labels=("tests",),
        assignee=SAM,
        description="Fixed by freezing the clock in the test.",
    ),
    SeedIssue(  # BENCH-10
        "Report totals are off by one",
        issue_type="Bug",
        labels=("reports",),
        assignee=BOT["accountId"],
        description="The last row is not counted in the total.",
    ),
    SeedIssue(  # BENCH-11
        "Add CSV import for contacts",
        issue_type="Story",
        labels=("import",),
    ),
    SeedIssue(  # BENCH-12
        "Editor crashes when saving a report twice",
        issue_type="Bug",
        status="In Progress",
        priority="Highest",
        labels=("crash", "reports"),
        assignee=ALEX,
        description="The second save hits a stale file handle.\n\n- Windows only\n- Since 2.4",
        comments=((BOT["accountId"], "Reproduced on Windows 11."),),
    ),
]


def at(hours: int) -> str:
    return (_START + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.000+0000")


def seed(fake: JiraFake) -> None:
    for account_id, name in USERS.items():
        fake.add_user(account_id, name)
    for n, spec in enumerate(ISSUES):
        issue = fake.add_issue(
            spec.summary,
            issue_type=spec.issue_type,
            status=spec.status,
            priority=spec.priority,
            labels=spec.labels,
            assignee=spec.assignee,
            description=spec.description,
            created=at(n * 5),
        )
        for c, (author, text) in enumerate(spec.comments):
            fake.add_comment(issue["key"], text, author=author, created=at(n * 5 + c + 1))


def seeded_fake(clock: Clock | None = None, *, require_auth: bool = True) -> JiraFake:
    fake = JiraFake(require_auth=require_auth, clock=clock or FakeClock())
    seed(fake)
    return fake
