"""Atlassian Document Format (ADF) <-> plain text.

Jira Cloud's v3 API takes and returns rich text (issue descriptions, comments) as ADF
JSON. Models read and write plain text, so the tools convert at the boundary:

- `text_to_adf` turns plain text into a document: blank lines separate paragraphs,
  single newlines become hard breaks, blocks of `- `/`* ` or `1. ` lines become lists and
  ``` fences become code blocks. Everything else stays literal text (no inline markup).
- `adf_to_text` flattens any document back to readable text. Nodes it does not know are
  read through their content, so new node types degrade instead of failing.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

Node = dict[str, Any]

_BULLET = re.compile(r"^\s*[-*] +(.*)$")
_NUMBERED = re.compile(r"^\s*\d+[.)] +(.*)$")
_FENCE = re.compile(r"^\s*```\s*([\w+#.-]*)\s*$")


# --- Text -> ADF --------------------------------------------------------------------------


def _text(value: str) -> list[Node]:
    return [{"type": "text", "text": value}] if value else []


def _inline(lines: Sequence[str]) -> list[Node]:
    content: list[Node] = []
    for i, line in enumerate(lines):
        if i:
            content.append({"type": "hardBreak"})
        content.extend(_text(line))
    return content


def _paragraph(lines: Sequence[str]) -> Node:
    return {"type": "paragraph", "content": _inline(lines)}


def _list(kind: str, items: Sequence[str]) -> Node:
    node: Node = {
        "type": kind,
        "content": [{"type": "listItem", "content": [_paragraph([item])]} for item in items],
    }
    if kind == "orderedList":
        node["attrs"] = {"order": 1}
    return node


def _block(lines: list[str]) -> Node:
    for pattern, kind in ((_BULLET, "bulletList"), (_NUMBERED, "orderedList")):
        matches = [pattern.match(line) for line in lines]
        if all(matches):
            return _list(kind, [m[1] for m in matches if m])
    return _paragraph(lines)


def text_to_adf(text: str) -> Node:
    """A version 1 ADF document for plain `text` (see the module docstring)."""
    content: list[Node] = []
    block: list[str] = []
    code: list[str] | None = None
    language = ""

    def flush() -> None:
        if block:
            content.append(_block(block))
            block.clear()

    for line in text.replace("\r\n", "\n").split("\n"):
        fence = _FENCE.match(line)
        if code is not None:
            if fence and not fence[1]:
                node: Node = {"type": "codeBlock", "content": _text("\n".join(code))}
                if language:
                    node["attrs"] = {"language": language}
                content.append(node)
                code = None
            else:
                code.append(line)
        elif fence:
            flush()
            code, language = [], fence[1]
        elif line.strip():
            block.append(line.rstrip())
        else:
            flush()
    if code is not None:  # unclosed fence: keep the text as it was typed
        block.extend([f"```{language}", *code])
    flush()
    return {"type": "doc", "version": 1, "content": content}


# --- ADF -> text --------------------------------------------------------------------------


def _attrs(node: Mapping[str, Any]) -> Mapping[str, Any]:
    attrs = node.get("attrs")
    return attrs if isinstance(attrs, Mapping) else {}


def _children(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    content = node.get("content")
    if not isinstance(content, list):
        return []
    return [child for child in content if isinstance(child, Mapping)]


def _text_node(node: Mapping[str, Any]) -> str:
    text = str(node.get("text") or "")
    for mark in node.get("marks") or []:
        if isinstance(mark, Mapping) and mark.get("type") == "link":
            href = _attrs(mark).get("href")
            if href and href != text:
                text = f"{text} ({href})"
    return text


def _date(node: Mapping[str, Any]) -> str:
    try:
        millis = int(str(_attrs(node).get("timestamp")))
    except ValueError:
        return ""
    return datetime.fromtimestamp(millis / 1000, UTC).strftime("%Y-%m-%d")


_LEAVES: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "text": _text_node,
    "hardBreak": lambda _: "\n",
    "mention": lambda n: str(_attrs(n).get("text") or "@user"),
    "emoji": lambda n: str(_attrs(n).get("text") or _attrs(n).get("shortName") or ""),
    "inlineCard": lambda n: str(_attrs(n).get("url") or ""),
    "blockCard": lambda n: str(_attrs(n).get("url") or ""),
    "status": lambda n: str(_attrs(n).get("text") or ""),
    "date": _date,
    "rule": lambda _: "---",
    "media": lambda _: "[attachment]",
}


def _inline_text(node: Mapping[str, Any]) -> str:
    return "".join(_render(child) for child in _children(node))


def _indent(text: str, prefix: str, rest: str) -> str:
    lines = text.split("\n")
    return "\n".join([prefix + lines[0], *(rest + line if line else line for line in lines[1:])])


def _items(node: Mapping[str, Any], ordered: bool) -> str:
    start = _attrs(node).get("order", 1) if ordered else 1
    start = start if isinstance(start, int) else 1
    out = []
    for i, item in enumerate(_children(node)):
        marker = f"{start + i}. " if ordered else "- "
        out.append(_indent(_blocks(item, "\n"), marker, " " * len(marker)))
    return "\n".join(out)


def _blocks(node: Mapping[str, Any], separator: str = "\n\n") -> str:
    parts = [_render(child) for child in _children(node)]
    return separator.join(part for part in parts if part)


def _render(node: Mapping[str, Any]) -> str:
    kind = str(node.get("type") or "")
    if kind in _LEAVES:
        return _LEAVES[kind](node)
    match kind:
        case "paragraph":
            return _inline_text(node)
        case "heading":
            level = _attrs(node).get("level", 1)
            return "#" * (level if isinstance(level, int) else 1) + " " + _inline_text(node)
        case "codeBlock":
            return f"```{_attrs(node).get('language') or ''}\n{_inline_text(node)}\n```"
        case "bulletList" | "orderedList":
            return _items(node, ordered=kind == "orderedList")
        case "blockquote":
            return "\n".join(f"> {line}" if line else ">" for line in _blocks(node).split("\n"))
        case "table":
            return "\n".join(_render(row) for row in _children(node))
        case "tableRow":
            return " | ".join(_blocks(cell, " ") for cell in _children(node))
        case _:
            # doc, panel, expand, listItem, mediaSingle, unknown nodes: read the content.
            return _blocks(node)


def adf_to_text(document: Any) -> str | None:
    """Readable text for an ADF document; None for an empty or missing one.

    Jira sends plain strings in a few places (and v2 APIs did everywhere); those pass
    through unchanged.
    """
    if document is None:
        return None
    if isinstance(document, str):
        return document or None
    if not isinstance(document, Mapping):
        return None
    return _render(document).strip() or None
