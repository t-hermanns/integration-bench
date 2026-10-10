"""Default data for the fake, shaped like the live sandbox.

`bench sandbox seed gdrive` creates the same tree in the real account: one app-created
folder `bench-sandbox` in My Drive (with `drive.file` it is all the app can see), two
subfolders, Google Docs and one plain text file. Drive assigns random ids, so scenarios
find files by name; creation order sets `modifiedTime`, which orders search results.
Keep this data and the live sandbox in sync.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from bench_core.clock import Clock
from bench_core.testing import FakeClock

from bench_gdrive.fake.backend import GdriveFake, timestamp
from bench_gdrive.models import DOCUMENT, FOLDER

SANDBOX = "bench-sandbox"
_START = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)


@dataclass(frozen=True)
class SeedFile:
    path: str
    """Folder path below `SANDBOX` plus name, e.g. "Projects/Roadmap 2026"."""
    mime_type: str = DOCUMENT
    content: str = ""


FILES: list[SeedFile] = [
    SeedFile("Projects", FOLDER),
    SeedFile("Meeting notes", FOLDER),
    SeedFile(
        "Projects/Roadmap 2026",
        content=(
            "Roadmap 2026\n"
            "Q1: ship the CSV importer\n"
            "Q2: dark mode for the dashboard\n"
            "Q3: offline support for the mobile app\n"
            "Q4: plugin API, version 1"
        ),
    ),
    SeedFile(
        "Projects/Release checklist",
        content=(
            "Release checklist\n"
            "1. Freeze the main branch\n"
            "2. Run the export tests with 10k rows\n"
            "3. Update the changelog\n"
            "4. Tag the release and announce it"
        ),
    ),
    SeedFile(
        "Meeting notes/Weekly sync 2026-09-07",
        content="Weekly sync 2026-09-07\nDecided: the CSV importer ships in Q1.\nOwner: Dana",
    ),
    SeedFile(
        "Meeting notes/Weekly sync 2026-09-14",
        content="Weekly sync 2026-09-14\nExport timeouts above 10k rows; Lee investigates.",
    ),
    SeedFile(
        "Meeting notes/Weekly sync 2026-09-21",
        content="Weekly sync 2026-09-21\nDark mode moved to Q2. Budget approved.",
    ),
    SeedFile(
        "Onboarding guide",
        content=(
            "Onboarding guide\n"
            "Read the roadmap in Projects first.\n"
            "Weekly syncs happen on Mondays; notes live in Meeting notes."
        ),
    ),
    SeedFile(
        "glossary.txt",
        "text/plain",
        "SSO: single sign-on\nIdP: identity provider\nETL: extract, transform, load\n",
    ),
]


def at(minutes: int) -> str:
    return timestamp((_START + timedelta(minutes=minutes)).timestamp())


def seed(fake: GdriveFake) -> str:
    """The seed tree below a new `SANDBOX` folder; returns that folder's id."""
    folders = {"": fake.add_folder(SANDBOX, modified=at(0))}
    for n, spec in enumerate(FILES, 1):
        parent_path, _, name = spec.path.rpartition("/")
        parent = folders[parent_path]
        if spec.mime_type == FOLDER:
            folders[spec.path] = fake.add_folder(name, parent=parent, modified=at(n))
        else:
            fake.add_file(
                name,
                mime_type=spec.mime_type,
                parent=parent,
                content=spec.content,
                modified=at(n),
            )
    return folders[""]


def seeded_fake(clock: Clock | None = None, *, require_auth: bool = True) -> GdriveFake:
    fake = GdriveFake(require_auth=require_auth, clock=clock or FakeClock())
    seed(fake)
    return fake
