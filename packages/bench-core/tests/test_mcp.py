from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from mcp.client import Client
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from bench_core.mcp import (
    Descriptions,
    ResponseFormat,
    ToolSpec,
    ToolText,
    build_server,
    lifespan_state,
    load_descriptions,
    variant_from_env,
)
from bench_core.permissions import ToolAccess, expand_levels, requires

V1 = """\
instructions: Demo server.
tools:
  demo_get_item:
    title: Get item
    description: Get one item.
    params:
      item_id: Item number.
  demo_create_item:
    description: Create an item.
"""

V2 = """\
tools:
  demo_get_item:
    description: |
      Get one item by its number. Use demo_create_item to add items instead.
    params:
      item_id: Item number, e.g. 42.
      response_format: concise or detailed.
  demo_create_item:
    description: Create an item with a title.
"""


@dataclass
class State:
    items: dict[int, str]
    events: list[str] = field(default_factory=list)


class Item(BaseModel):
    id: int
    title: str


async def get_item(
    item_id: int, ctx: Context[State], response_format: ResponseFormat = "concise"
) -> Item:
    state = lifespan_state(ctx)
    if item_id not in state.items:
        raise ToolError(f"Item {item_id} not found.")
    return Item(id=item_id, title=state.items[item_id])


async def create_item(title: str, ctx: Context[State]) -> Item:
    if title == "broken":
        return Item.model_validate({"id": "not-an-int", "title": title})
    state = lifespan_state(ctx)
    item_id = max(state.items, default=0) + 1
    state.items[item_id] = title
    return Item(id=item_id, title=title)


TOOLS = [
    ToolSpec(get_item, ToolAccess("demo_get_item", requires("items:read"))),
    ToolSpec(create_item, ToolAccess("demo_create_item", requires("items:write"), write=True)),
]


@pytest.fixture
def descriptions_dir(tmp_path: Path) -> Path:
    (tmp_path / "v1.yaml").write_text(V1, encoding="utf-8")
    (tmp_path / "v2.yaml").write_text(V2, encoding="utf-8")
    return tmp_path


def make_server(descriptions: Descriptions, events: list[str] | None = None, **kwargs):
    log = events if events is not None else []

    @asynccontextmanager
    async def lifespan() -> AsyncIterator[State]:
        log.append("open")
        yield State(items={1: "first"}, events=log)
        log.append("close")

    kwargs.setdefault("read_only", False)
    return build_server("demo", TOOLS, descriptions, lifespan, **kwargs)


def text_of(result: CallToolResult) -> str:
    assert isinstance(result.content[0], TextContent)
    return result.content[0].text


# --- Loading ------------------------------------------------------------------------------


def test_load_descriptions_variant_and_contents(descriptions_dir: Path) -> None:
    d = load_descriptions(descriptions_dir, "v1")
    assert d.variant == "v1"
    assert d.instructions == "Demo server."
    assert d.tools["demo_get_item"] == ToolText(
        description="Get one item.", title="Get item", params={"item_id": "Item number."}
    )
    assert d.tools["demo_create_item"].params == {}


def test_variant_from_env(descriptions_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert variant_from_env({}) == "v1"
    assert variant_from_env({"BENCH_DESCRIPTIONS": " v2 "}) == "v2"
    monkeypatch.setenv("BENCH_DESCRIPTIONS", "v2")
    assert load_descriptions(descriptions_dir).variant == "v2"


@pytest.mark.parametrize("variant", ["../v1", "V1", "v1.yaml", "/etc/passwd"])
def test_invalid_variant_names_are_rejected(descriptions_dir: Path, variant: str) -> None:
    with pytest.raises(ValueError, match="not a valid"):
        variant_from_env({"BENCH_DESCRIPTIONS": variant})
    with pytest.raises(ValueError, match="not a valid"):
        load_descriptions(descriptions_dir, variant)


def test_missing_variant_file(descriptions_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="v3"):
        load_descriptions(descriptions_dir, "v3")


@pytest.mark.parametrize(
    "content",
    [
        "tools:\n  demo_get_item:\n    descripton: typo\n",
        "tools:\n  demo_get_item:\n    description: x\n    extra: y\n",
        "tool: {}\n",
        "- a list\n",
        "",
    ],
)
def test_malformed_descriptions_are_rejected(tmp_path: Path, content: str) -> None:
    (tmp_path / "bad.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.yaml"):
        load_descriptions(tmp_path, "bad")


# --- Registration -------------------------------------------------------------------------


@pytest.mark.anyio
async def test_descriptions_come_from_yaml(descriptions_dir: Path) -> None:
    server = make_server(load_descriptions(descriptions_dir, "v1"))
    async with Client(server) as client:
        assert client.instructions == "Demo server."
        tools = {t.name: t for t in (await client.list_tools()).tools}

    get = tools["demo_get_item"]
    assert get.description == "Get one item."
    assert get.title == "Get item"
    assert get.input_schema["properties"]["item_id"]["description"] == "Item number."
    assert "description" not in get.input_schema["properties"]["response_format"]
    assert get.input_schema["properties"]["response_format"]["enum"] == ["concise", "detailed"]
    assert "ctx" not in get.input_schema["properties"]
    assert get.annotations is not None
    assert get.annotations.read_only_hint is True
    assert get.annotations.title == "Get item"
    assert get.output_schema is not None

    create = tools["demo_create_item"]
    assert create.annotations is not None
    assert create.annotations.read_only_hint is False
    assert create.annotations.destructive_hint is False


@pytest.mark.anyio
async def test_switching_variant_changes_only_texts(descriptions_dir: Path) -> None:
    schemas = {}
    for variant in ("v1", "v2"):
        server = make_server(load_descriptions(descriptions_dir, variant))
        async with Client(server) as client:
            schemas[variant] = {t.name: t for t in (await client.list_tools()).tools}

    v1, v2 = schemas["v1"]["demo_get_item"], schemas["v2"]["demo_get_item"]
    assert (
        v2.description == "Get one item by its number. Use demo_create_item to add items instead."
    )
    assert v2.input_schema["properties"]["item_id"]["description"] == "Item number, e.g. 42."
    assert set(v1.input_schema["properties"]) == set(v2.input_schema["properties"])
    assert v1.input_schema["required"] == v2.input_schema["required"] == ["item_id"]


@pytest.mark.parametrize(
    ("tools_yaml", "message"),
    [
        ({"demo_get_item": {"description": "x"}}, "lack tool.*demo_create_item"),
        (
            {
                "demo_get_item": {"description": "x"},
                "demo_create_item": {"description": "y"},
                "demo_delete_item": {"description": "z"},
            },
            "unknown tool.*demo_delete_item",
        ),
        (
            {
                "demo_get_item": {"description": "x", "params": {"itemId": "typo"}},
                "demo_create_item": {"description": "y"},
            },
            "unknown parameter.*itemId",
        ),
    ],
)
def test_descriptions_must_match_code(tools_yaml: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_server(Descriptions.model_validate({"tools": tools_yaml}))


def test_unknown_parameter_check_also_covers_filtered_out_tools(descriptions_dir: Path) -> None:
    d = Descriptions.model_validate(
        {
            "tools": {
                "demo_get_item": {"description": "x"},
                "demo_create_item": {"description": "y", "params": {"titel": "typo"}},
            }
        }
    )
    # Hidden by read_only, but the YAML is still wrong and should not go unnoticed later.
    with pytest.raises(ValueError, match="titel"):
        make_server(d, read_only=True)


def test_tool_names_must_be_namespaced_and_unique(descriptions_dir: Path) -> None:
    d = load_descriptions(descriptions_dir, "v1")

    @asynccontextmanager
    async def lifespan() -> AsyncIterator[None]:
        yield None

    with pytest.raises(ValueError, match="must start with 'other_'"):
        build_server("other", TOOLS, d, lifespan)
    with pytest.raises(ValueError, match="Duplicate"):
        build_server("demo", [*TOOLS, TOOLS[0]], d, lifespan)


# --- Permission filter --------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("kwargs", "env", "enabled", "disabled"),
    [
        ({}, {}, {"demo_get_item", "demo_create_item"}, {}),
        ({"read_only": True}, {}, {"demo_get_item"}, {"demo_create_item": "BENCH_READ_ONLY"}),
        (
            {"read_only": None},
            {"BENCH_READ_ONLY": "1"},
            {"demo_get_item"},
            {"demo_create_item": "BENCH_READ_ONLY"},
        ),
        (
            {"granted": frozenset({"items:read"})},
            {},
            {"demo_get_item"},
            {"demo_create_item": "items:write"},
        ),
        (
            {"granted": frozenset({"items:write"}), "expand": expand_levels},
            {},
            {"demo_get_item", "demo_create_item"},
            {},
        ),
        (
            {"granted": frozenset()},
            {},
            set(),
            {"demo_get_item": "items:read", "demo_create_item": "items:write"},
        ),
    ],
)
async def test_tools_are_filtered_by_scopes_and_read_only(
    descriptions_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict,
    env: dict[str, str],
    enabled: set[str],
    disabled: dict[str, str],
) -> None:
    monkeypatch.delenv("BENCH_READ_ONLY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    server = make_server(load_descriptions(descriptions_dir, "v1"), **kwargs)
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        if "demo_create_item" not in enabled:
            result = await client.call_tool("demo_create_item", {"title": "x"})
            assert result.is_error
            assert "Unknown tool" in text_of(result)

    assert names == enabled
    assert {a.name for a in server.selection.enabled} == enabled
    assert set(server.selection.disabled) == set(disabled)
    for name, reason in disabled.items():
        assert reason in server.selection.disabled[name]


# --- Calls --------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_lifespan_state_is_shared_within_a_session(descriptions_dir: Path) -> None:
    events: list[str] = []
    server = make_server(load_descriptions(descriptions_dir, "v1"), events)
    async with Client(server) as client:
        created = await client.call_tool("demo_create_item", {"title": "second"})
        fetched = await client.call_tool("demo_get_item", {"item_id": 2})
        assert events == ["open"]
    assert events == ["open", "close"]

    assert created.structured_content == {"id": 2, "title": "second"}
    assert fetched.structured_content == {"id": 2, "title": "second"}

    # A new session gets a fresh lifespan.
    async with Client(server) as client:
        again = await client.call_tool("demo_get_item", {"item_id": 2})
    assert again.is_error
    assert events == ["open", "close", "open", "close"]


@pytest.mark.anyio
async def test_tool_error_text_reaches_model(descriptions_dir: Path) -> None:
    server = make_server(load_descriptions(descriptions_dir, "v1"))
    async with Client(server) as client:
        result = await client.call_tool("demo_get_item", {"item_id": 9})
    assert result.is_error
    # The SDK prefixes the tool name; the message itself is kept verbatim.
    assert text_of(result) == "Error executing tool demo_get_item: Item 9 not found."


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            {"item_id": "abc"},
            "Invalid arguments for demo_get_item: item_id: Input should be a valid integer, "
            "unable to parse string as an integer (got 'abc'). Fix them and call again.",
        ),
        (
            {},
            "Invalid arguments for demo_get_item: item_id: Field required. "
            "Fix them and call again.",
        ),
        (
            {"item_id": 1, "response_format": "full"},
            "Invalid arguments for demo_get_item: response_format: Input should be 'concise' "
            "or 'detailed' (got 'full'). Fix them and call again.",
        ),
    ],
)
async def test_argument_errors_are_condensed(
    descriptions_dir: Path, arguments: dict, expected: str
) -> None:
    server = make_server(load_descriptions(descriptions_dir, "v1"))
    async with Client(server) as client:
        result = await client.call_tool("demo_get_item", arguments)
    assert result.is_error
    assert text_of(result) == expected
    assert "pydantic.dev" not in text_of(result)


@pytest.mark.anyio
async def test_long_inputs_are_truncated_in_argument_errors(descriptions_dir: Path) -> None:
    server = make_server(load_descriptions(descriptions_dir, "v1"))
    async with Client(server) as client:
        result = await client.call_tool("demo_get_item", {"item_id": "x" * 500})
    assert result.is_error
    assert "x" * 100 not in text_of(result)
    assert "...)" in text_of(result)


@pytest.mark.anyio
async def test_validation_errors_inside_tool_are_not_blamed_on_arguments(
    descriptions_dir: Path,
) -> None:
    server = make_server(load_descriptions(descriptions_dir, "v1"))
    async with Client(server) as client:
        result = await client.call_tool("demo_create_item", {"title": "broken"})
    assert result.is_error
    assert "Invalid arguments" not in text_of(result)
    assert "not-an-int" not in text_of(result)
