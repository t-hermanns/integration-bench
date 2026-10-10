"""MCP tool definitions -> the `Tool`s a provider accepts.

Only the JSON Schema our servers emit is handled (Pydantic output of plain, optional,
list and `Literal` parameters). Anything else raises `UnsupportedSchema`, so a new tool
fails the snapshot tests instead of a live run.

All providers get the same normalized schema: no `title`s (pure token cost), optional
parameters (`anyOf: [X, {"type": "null"}]`) collapsed to `X` and `default: null`
dropped. Cohere additionally gets `minimum`, `maximum` and `default` folded into the
description, since its strict mode rejects them; the OpenAI-compatible APIs keep them.
"""

import json
from collections.abc import Iterable
from typing import Any

from mcp.types import Tool as McpTool

from bench_evals.models.base import Tool

SUPPORTED = {
    "type",
    "title",
    "description",
    "properties",
    "required",
    "items",
    "enum",
    "default",
    "minimum",
    "maximum",
    "anyOf",
}
FOLDED = {"cohere": ("minimum", "maximum", "default")}
"""Per provider: keywords moved from the schema into the description."""


class UnsupportedSchema(ValueError):
    pass


def provider_tools(tools: Iterable[McpTool], provider: str) -> list[Tool]:
    """The MCP server's tools (from `list_tools()`) in the form `provider` accepts."""
    out = []
    for tool in tools:
        wire = tool.model_dump(by_alias=True, exclude_none=True)
        try:
            parameters = convert(wire["inputSchema"], provider)
        except UnsupportedSchema as exc:
            raise UnsupportedSchema(f"{tool.name}: {exc}") from None
        out.append(Tool(tool.name, wire.get("description", ""), parameters))
    return out


def convert(schema: dict[str, Any], provider: str) -> dict[str, Any]:
    """The tool's `inputSchema` (wire form, camelCase) for `provider`."""
    if schema.get("type") != "object":
        raise UnsupportedSchema("input schema is not an object")
    return _node(schema, FOLDED.get(provider, ()), "$")


def _node(schema: dict[str, Any], folded: tuple[str, ...], path: str) -> dict[str, Any]:
    schema = _collapse_optional(schema, path)
    unknown = set(schema) - SUPPORTED
    if unknown:
        raise UnsupportedSchema(f"{path}: unsupported keyword(s) {sorted(unknown)}")
    out: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in schema.items():
        if key == "title" or (key == "default" and value is None):
            continue
        if key in folded:
            notes.append(_note(key, value))
        elif key == "properties":
            out[key] = {name: _node(sub, folded, f"{path}.{name}") for name, sub in value.items()}
        elif key == "items":
            out[key] = _node(value, folded, f"{path}[]")
        else:
            out[key] = value
    if notes:
        text = out.get("description", "")
        out["description"] = " ".join([text.rstrip(), *notes]).strip()
    return out


def _collapse_optional(schema: dict[str, Any], path: str) -> dict[str, Any]:
    """`anyOf: [X, null]` -> `X`, keeping the outer description and default."""
    if "anyOf" not in schema:
        return schema
    variants = [v for v in schema["anyOf"] if v != {"type": "null"}]
    if len(variants) != 1 or len(schema["anyOf"]) != 2:
        raise UnsupportedSchema(f"{path}: anyOf other than 'X or null'")
    if "type" in schema or "anyOf" in variants[0]:
        raise UnsupportedSchema(f"{path}: anyOf next to type or nested")
    outer = {k: v for k, v in schema.items() if k != "anyOf"}
    return variants[0] | outer


def _note(key: str, value: Any) -> str:
    label = {"minimum": "Minimum", "maximum": "Maximum", "default": "Default"}[key]
    return f"{label}: {json.dumps(value)}."
