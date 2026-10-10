"""Plain text <-> Atlassian Document Format."""

from typing import Any

import pytest

from bench_jira.adf import adf_to_text, text_to_adf


def doc(*content: dict[str, Any]) -> dict[str, Any]:
    return {"type": "doc", "version": 1, "content": list(content)}


def para(*content: dict[str, Any]) -> dict[str, Any]:
    return {"type": "paragraph", "content": list(content)}


def text(value: str, **attrs: Any) -> dict[str, Any]:
    return {"type": "text", "text": value, **attrs}


def test_paragraphs_and_hard_breaks() -> None:
    assert text_to_adf("one\ntwo\n\nthree") == doc(
        para(text("one"), {"type": "hardBreak"}, text("two")), para(text("three"))
    )


def test_lists() -> None:
    document = text_to_adf("Steps:\n\n1. open\n2. click\n\n- a\n* b")
    kinds = [node["type"] for node in document["content"]]
    assert kinds == ["paragraph", "orderedList", "bulletList"]
    ordered = document["content"][1]
    assert ordered["attrs"] == {"order": 1}
    assert ordered["content"][1] == {"type": "listItem", "content": [para(text("click"))]}


def test_mixed_block_stays_a_paragraph() -> None:
    document = text_to_adf("Note:\n- not a list on its own")
    assert [node["type"] for node in document["content"]] == ["paragraph"]


def test_code_fences() -> None:
    document = text_to_adf("Run:\n```python\nprint(1)\n\nprint(2)\n```\nDone.")
    assert document["content"][1] == {
        "type": "codeBlock",
        "attrs": {"language": "python"},
        "content": [text("print(1)\n\nprint(2)")],
    }
    assert document["content"][2] == para(text("Done."))


def test_unclosed_fence_keeps_the_text() -> None:
    assert adf_to_text(text_to_adf("```\nx")) == "```\nx"


def test_empty_text_is_an_empty_document() -> None:
    assert text_to_adf("") == doc()
    assert adf_to_text(doc()) is None


@pytest.mark.parametrize(
    "value",
    [
        "Plain line",
        "First\nsecond\n\nThird paragraph",
        "1. one\n2. two\n\n- x\n- y",
        "```sql\nSELECT 1;\n```",
        "Trailing text after list\n\n- a\n- b\n\nEnd",
    ],
)
def test_round_trip(value: str) -> None:
    assert adf_to_text(text_to_adf(value)) == value


def test_reads_rich_nodes() -> None:
    document = doc(
        {"type": "heading", "attrs": {"level": 2}, "content": [text("Summary")]},
        para(
            text("See "),
            text("the docs", marks=[{"type": "link", "attrs": {"href": "https://x.test/d"}}]),
            text(", ping "),
            {"type": "mention", "attrs": {"id": "1", "text": "@Alex Doe"}},
            text(" "),
            {"type": "emoji", "attrs": {"shortName": ":smile:", "text": "😄"}},
            text(" by "),
            {"type": "date", "attrs": {"timestamp": "1790000000000"}},
            text(" "),
            {"type": "status", "attrs": {"text": "BLOCKED", "color": "red"}},
        ),
        {
            "type": "bulletList",
            "content": [
                {
                    "type": "listItem",
                    "content": [
                        para(text("outer")),
                        {
                            "type": "orderedList",
                            "attrs": {"order": 3},
                            "content": [{"type": "listItem", "content": [para(text("inner"))]}],
                        },
                    ],
                }
            ],
        },
        {"type": "blockquote", "content": [para(text("quoted"))]},
        {"type": "rule"},
        {
            "type": "table",
            "content": [
                {
                    "type": "tableRow",
                    "content": [
                        {"type": "tableHeader", "content": [para(text("a"))]},
                        {"type": "tableCell", "content": [para(text("b"))]},
                    ],
                }
            ],
        },
        {"type": "mediaSingle", "content": [{"type": "media", "attrs": {"id": "m"}}]},
        {
            "type": "panel",
            "content": [para({"type": "inlineCard", "attrs": {"url": "https://x.test"}})],
        },
        {"type": "somethingNew", "content": [para(text("still readable"))]},
    )
    assert adf_to_text(document) == (
        "## Summary\n\n"
        "See the docs (https://x.test/d), ping @Alex Doe 😄 by 2026-09-21 BLOCKED\n\n"
        "- outer\n  3. inner\n\n"
        "> quoted\n\n"
        "---\n\n"
        "a | b\n\n"
        "[attachment]\n\n"
        "https://x.test\n\n"
        "still readable"
    )


def test_plain_strings_and_garbage() -> None:
    assert adf_to_text("v2 style") == "v2 style"
    assert adf_to_text("") is None
    assert adf_to_text(None) is None
    assert adf_to_text(42) is None
    assert adf_to_text({"type": "doc", "content": "oops"}) is None
