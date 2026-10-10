import importlib
import json
import os
from pathlib import Path

import httpx2
import pytest
from mcp import Client
from mcp.types import Tool as McpTool

from bench_evals.models import PROVIDERS
from bench_evals.schema import UnsupportedSchema, convert, provider_tools

SNAPSHOTS = Path(__file__).parent / "snapshots"
INTEGRATIONS = ("gdrive", "github", "jira", "slack")

SEARCH = {
    "type": "object",
    "title": "search_issuesArguments",
    "properties": {
        "jql": {"title": "Jql", "type": "string", "description": "JQL query."},
        "cursor": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "title": "Cursor",
            "description": "Next page.",
        },
        "page_size": {
            "default": 10,
            "maximum": 50,
            "minimum": 1,
            "title": "Page Size",
            "type": "integer",
            "description": "Results per page.",
        },
        "labels": {
            "anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}],
            "default": None,
            "title": "Labels",
        },
        "response_format": {
            "default": "concise",
            "enum": ["concise", "detailed"],
            "title": "Response Format",
            "type": "string",
        },
    },
    "required": ["jql"],
}


def test_openai_compatible_keeps_constraints_drops_titles_and_null():
    assert convert(SEARCH, "groq") == {
        "type": "object",
        "properties": {
            "jql": {"type": "string", "description": "JQL query."},
            "cursor": {"type": "string", "description": "Next page."},
            "page_size": {
                "default": 10,
                "maximum": 50,
                "minimum": 1,
                "type": "integer",
                "description": "Results per page.",
            },
            "labels": {"items": {"type": "string"}, "type": "array"},
            "response_format": {
                "default": "concise",
                "enum": ["concise", "detailed"],
                "type": "string",
            },
        },
        "required": ["jql"],
    }


def test_cohere_folds_constraints_into_the_description():
    props = convert(SEARCH, "cohere")["properties"]
    assert props["page_size"] == {
        "type": "integer",
        "description": "Results per page. Default: 10. Maximum: 50. Minimum: 1.",
    }
    assert props["response_format"] == {
        "enum": ["concise", "detailed"],
        "type": "string",
        "description": 'Default: "concise".',
    }
    assert props["cursor"] == {"type": "string", "description": "Next page."}


def test_input_is_not_modified():
    before = json.dumps(SEARCH)
    convert(SEARCH, "cohere")
    assert json.dumps(SEARCH) == before


@pytest.mark.parametrize(
    ("prop", "message"),
    [
        ({"$ref": "#/$defs/Filter"}, r"\$\.x: unsupported keyword\(s\) \['\$ref'\]"),
        ({"type": "string", "pattern": "^a"}, r"\['pattern'\]"),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "anyOf other than"),
        ({"anyOf": [{"type": "string"}, {"type": "null"}], "type": "string"}, "anyOf next"),
        ({"type": "array", "items": {"type": "string", "format": "date"}}, r"\$\.x\[\]"),
    ],
)
def test_unsupported_schema_fails_loudly(prop, message):
    with pytest.raises(UnsupportedSchema, match=message):
        convert({"type": "object", "properties": {"x": prop}}, "groq")


def test_provider_tools_names_the_tool():
    bad = McpTool(name="t", input_schema={"type": "object", "properties": {"x": {"oneOf": []}}})
    with pytest.raises(UnsupportedSchema, match=r"^t: "):
        provider_tools([bad], "gemini")


async def all_mcp_tools(variant: str = "v1") -> list[McpTool]:
    tools: list[McpTool] = []
    for name in INTEGRATIONS:
        module = importlib.import_module(f"bench_{name}.server")
        server = module.create_server(
            transport=httpx2.MockTransport(lambda r: httpx2.Response(500)),
            read_only=False,
            variant=variant,
        )
        async with Client(server) as client:
            tools += (await client.list_tools()).tools
    return tools


@pytest.mark.anyio
@pytest.mark.parametrize("provider", sorted(PROVIDERS))
async def test_snapshot(provider):
    """All tools of all servers per provider; `UPDATE_SNAPSHOTS=1` rewrites the files."""
    tools = provider_tools(await all_mcp_tools(), provider)
    actual = json.dumps(
        [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools],
        indent=1,
        ensure_ascii=False,
    )
    path = SNAPSHOTS / f"tools_{provider}.json"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(actual.encode() + b"\n")
    assert path.read_bytes().decode() == actual + "\n"


@pytest.mark.anyio
async def test_v2_rewrites_every_tool_and_converts_for_every_provider():
    """The description experiment needs a v2 text for every tool and every argument."""
    v1 = {t.name: t for t in await all_mcp_tools("v1")}
    v2 = {t.name: t for t in await all_mcp_tools("v2")}
    assert v2.keys() == v1.keys()
    for name, tool in v2.items():
        assert tool.description, name
        assert tool.description != v1[name].description, name
        for arg, prop in tool.input_schema["properties"].items():
            assert prop.get("description"), f"{name}.{arg}"
    for provider in PROVIDERS:
        assert len(provider_tools(list(v2.values()), provider)) == len(v2)
