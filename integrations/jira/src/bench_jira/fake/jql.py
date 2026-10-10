"""A small JQL interpreter for the fake: the subset models actually write.

Supported: `AND`/`OR`/`NOT`, parentheses, `= != ~ !~ > >= < <=`, `in`/`not in`,
`is [not] EMPTY`, quoted and bare values, `currentUser()`, absolute (`2026-09-01`) and
relative (`-7d`, `-2w`, `-3h`, `-30m`) dates and `ORDER BY`. Fields: project, key, status,
statusCategory, issuetype/type, priority, assignee, reporter, labels, summary,
description, comment, text, created, updated.

Errors read like Jira's (unknown field, unknown value, unsupported operator, syntax,
unbounded query), so tool errors in tests match what the model sees live. Like Jira,
`!=` and `not in` never match issues whose field is empty.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

Issue = Mapping[str, Any]
Predicate = Callable[[Issue], bool]

PRIORITY_RANK = {"Lowest": 1, "Low": 2, "Medium": 3, "High": 4, "Highest": 5}
UNBOUNDED = (
    "Unbounded JQL queries are not allowed here. Please add a search restriction to your query."
)


class JqlError(ValueError):
    """The message is what Jira puts into `errorMessages`."""


# --- Tokens -------------------------------------------------------------------------------

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
      | (?P<op>!=|!~|>=|<=|=|~|>|<)
      | (?P<punct>[(),])
      | (?P<word>[^\s"'()=!~<>,]+)
    )""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str  # string | op | punct | word
    text: str

    @property
    def keyword(self) -> str:
        return self.text.upper() if self.kind == "word" else ""


def tokenize(query: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    query = query.rstrip()
    while pos < len(query):
        match = _TOKEN.match(query, pos)
        if match is None or match.end() == pos:
            raise JqlError(f"Error in the JQL Query: The character '{query[pos]}' is reserved.")
        kind = match.lastgroup
        assert kind is not None
        text = match[kind]
        if kind == "string":
            text = re.sub(r"\\(.)", r"\1", text[1:-1])
        tokens.append(Token(kind, text))
        pos = match.end()
    return tokens


# --- Fields -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # value | text | date
    get: Callable[[Issue], list[Any]]
    known: Callable[["Context"], set[str]] | None = None


@dataclass(frozen=True)
class Context:
    """What the fake knows beyond the issues: the caller and the configured values."""

    current_user: str
    now: datetime
    projects: Mapping[str, str] = field(default_factory=dict)  # key -> name
    statuses: Mapping[str, str] = field(default_factory=dict)  # name -> category key
    categories: Mapping[str, str] = field(default_factory=dict)  # key -> name
    issue_types: Sequence[str] = ()
    priorities: Sequence[str] = ()


def _fields(issue: Issue) -> Mapping[str, Any]:
    return issue.get("fields") or {}


def _nested(*path: str) -> Callable[[Issue], list[Any]]:
    def get(issue: Issue) -> list[Any]:
        value: Any = _fields(issue)
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        return [] if value is None else [value]

    return get


def _person(name: str) -> Callable[[Issue], list[Any]]:
    def get(issue: Issue) -> list[Any]:
        person = _fields(issue).get(name)
        if not isinstance(person, Mapping):
            return []
        return [v for v in (person.get("accountId"), person.get("displayName")) if v]

    return get


def _plain(value: Any) -> str:
    """Text of an ADF document (or a plain string), enough for `~` matching."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join([str(value.get("text") or ""), _plain(value.get("content"))])
    if isinstance(value, list):
        return " ".join(_plain(item) for item in value)
    return ""


def _comments(issue: Issue) -> list[Any]:
    comments = (_fields(issue).get("comment") or {}).get("comments") or []
    return [_plain(c.get("body")) for c in comments]


def _text(issue: Issue) -> list[Any]:
    return [
        str(_fields(issue).get("summary") or ""),
        _plain(_fields(issue).get("description")),
        *_comments(issue),
    ]


def _date(name: str) -> Callable[[Issue], list[Any]]:
    def get(issue: Issue) -> list[Any]:
        value = _fields(issue).get(name)
        return [datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")] if value else []

    return get


def _project(issue: Issue) -> list[Any]:
    project = _fields(issue).get("project") or {}
    return [v for v in (project.get("key"), project.get("name")) if v]


def _category(issue: Issue) -> list[Any]:
    category = ((_fields(issue).get("status") or {}).get("statusCategory")) or {}
    return [v for v in (category.get("key"), category.get("name")) if v]


_FIELD_LIST = [
    Field("project", "value", _project, lambda c: {*c.projects, *c.projects.values()}),
    Field("key", "value", lambda i: [i.get("key"), i.get("id")]),
    Field("status", "value", _nested("status", "name"), lambda c: set(c.statuses)),
    Field(
        "statuscategory",
        "value",
        _category,
        lambda c: {*c.categories, *c.categories.values()},
    ),
    Field("issuetype", "value", _nested("issuetype", "name"), lambda c: set(c.issue_types)),
    Field("priority", "value", _nested("priority", "name"), lambda c: set(c.priorities)),
    Field("assignee", "value", _person("assignee")),
    Field("reporter", "value", _person("reporter")),
    Field("labels", "value", lambda i: list(_fields(i).get("labels") or [])),
    Field("summary", "text", lambda i: [_fields(i).get("summary") or ""]),
    Field("description", "text", lambda i: [_plain(_fields(i).get("description"))]),
    Field("comment", "text", _comments),
    Field("text", "text", _text),
    Field("created", "date", _date("created")),
    Field("updated", "date", _date("updated")),
]
FIELDS = {f.name: f for f in _FIELD_LIST}
ALIASES = {
    "issuekey": "key",
    "id": "key",
    "type": "issuetype",
    "createddate": "created",
    "updateddate": "updated",
}


def lookup(name: str) -> Field:
    key = name.casefold()
    found = FIELDS.get(ALIASES.get(key, key))
    if found is None:
        raise JqlError(f"Field '{name}' does not exist or you do not have permission to view it.")
    return found


# --- Values -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Value:
    text: str
    function: bool = False


_RELATIVE = re.compile(r"^([+-]?)(\d+)([mhdw])$")
_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def parse_date(value: str, now: datetime) -> datetime:
    if m := _RELATIVE.match(value.strip()):
        delta = timedelta(**{_UNITS[m[3]]: int(m[2])})
        return now - delta if m[1] == "-" else now + delta
    for fmt in ("%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise JqlError(
        f"Date value '{value}' is invalid. Valid formats include: 'yyyy/MM/dd HH:mm', "
        "'yyyy-MM-dd HH:mm', 'yyyy/MM/dd', 'yyyy-MM-dd', or a period format e.g. '-5d', '4w 2d'."
    )


# --- Parser -------------------------------------------------------------------------------


@dataclass
class Query:
    where: Predicate | None
    order: list[tuple[str, bool]]  # (field, descending)

    def matches(self, issue: Issue) -> bool:
        return self.where is None or self.where(issue)

    def sort(self, issues: list[Issue]) -> list[Issue]:
        result = list(issues)
        for name, descending in reversed(self.order or [("created", True)]):
            result.sort(key=_sort_key(name), reverse=descending)
        return result


def _sort_key(name: str) -> Callable[[Issue], Any]:
    found = lookup(name)

    def key(issue: Issue) -> Any:
        if found.name == "key":
            project, _, number = str(issue.get("key")).partition("-")
            return (project, int(number or 0))
        if found.name == "priority":
            values = found.get(issue)
            return PRIORITY_RANK.get(values[0], 0) if values else 0
        values = found.get(issue)
        return (bool(values), str(values[0]).casefold() if values else "")

    return key


class _Parser:
    def __init__(self, tokens: list[Token], context: Context) -> None:
        self.tokens = tokens
        self.pos = 0
        self.context = context

    # helpers

    def peek(self, offset: int = 0) -> Token | None:
        index = self.pos + offset
        return self.tokens[index] if index < len(self.tokens) else None

    def take(self) -> Token:
        token = self.peek()
        if token is None:
            raise JqlError("Error in the JQL Query: Expecting more of the query, got end of query.")
        self.pos += 1
        return token

    def at_keyword(self, *words: str) -> bool:
        return all((t := self.peek(i)) is not None and t.keyword == w for i, w in enumerate(words))

    def expect_punct(self, char: str) -> None:
        token = self.peek()
        if token is None or token.kind != "punct" or token.text != char:
            got = "end of query" if token is None else f"'{token.text}'"
            raise JqlError(f"Error in the JQL Query: Expecting '{char}' but got {got}.")
        self.pos += 1

    # grammar

    def query(self) -> Query:
        where = None
        if self.peek() is not None and not self.at_keyword("ORDER", "BY"):
            where = self.or_expr()
        order: list[tuple[str, bool]] = []
        if self.at_keyword("ORDER", "BY"):
            self.pos += 2
            order.append(self.order_item())
            while (t := self.peek()) is not None and t.kind == "punct" and t.text == ",":
                self.pos += 1
                order.append(self.order_item())
        if (extra := self.peek()) is not None:
            raise JqlError(
                f"Error in the JQL Query: Expecting either 'OR' or 'AND' but got '{extra.text}'."
            )
        return Query(where, order)

    def order_item(self) -> tuple[str, bool]:
        token = self.take()
        lookup(token.text)
        descending = False
        if self.at_keyword("DESC") or self.at_keyword("ASC"):
            descending = self.take().keyword == "DESC"
        return token.text, descending

    def or_expr(self) -> Predicate:
        parts = [self.and_expr()]
        while self.at_keyword("OR"):
            self.pos += 1
            parts.append(self.and_expr())
        return parts[0] if len(parts) == 1 else lambda i: any(p(i) for p in parts)

    def and_expr(self) -> Predicate:
        parts = [self.unary()]
        while self.at_keyword("AND"):
            self.pos += 1
            parts.append(self.unary())
        return parts[0] if len(parts) == 1 else lambda i: all(p(i) for p in parts)

    def unary(self) -> Predicate:
        if self.at_keyword("NOT"):
            self.pos += 1
            inner = self.unary()
            return lambda i: not inner(i)
        token = self.peek()
        if token is not None and token.kind == "punct" and token.text == "(":
            self.pos += 1
            inner = self.or_expr()
            self.expect_punct(")")
            return inner
        return self.clause()

    def operator(self) -> str:
        if self.at_keyword("NOT", "IN"):
            self.pos += 2
            return "not in"
        if self.at_keyword("IS", "NOT"):
            self.pos += 2
            return "is not"
        if self.at_keyword("IN") or self.at_keyword("IS"):
            return self.take().keyword.lower()
        token = self.take()
        if token.kind != "op":
            raise JqlError(
                f"Error in the JQL Query: Expecting operator but got '{token.text}'. Valid "
                "operators are '=', '!=', '<', '>', '<=', '>=', '~', '!~', 'IN', 'NOT IN', "
                "'IS' and 'IS NOT'."
            )
        return token.text

    def value(self) -> Value:
        token = self.take()
        if token.kind == "punct":
            raise JqlError(f"Error in the JQL Query: Expecting a value but got '{token.text}'.")
        nxt, after = self.peek(), self.peek(1)
        if (
            token.kind == "word"
            and nxt is not None
            and nxt.text == "("
            and after is not None
            and after.text == ")"
        ):
            self.pos += 2
            return Value(token.text, function=True)
        return Value(token.text)

    def values(self) -> list[Value]:
        self.expect_punct("(")
        result = [self.value()]
        while (t := self.peek()) is not None and t.kind == "punct" and t.text == ",":
            self.pos += 1
            result.append(self.value())
        self.expect_punct(")")
        return result

    def clause(self) -> Predicate:
        name = self.take()
        if name.kind not in ("word", "string"):
            raise JqlError(f"Error in the JQL Query: Expecting a field name but got '{name.text}'.")
        found = lookup(name.text)
        op = self.operator()
        if op in ("in", "not in"):
            return self.compare(found, name.text, op, self.values())
        if op in ("is", "is not"):
            token = self.take()
            if token.keyword not in ("EMPTY", "NULL"):
                raise JqlError(
                    f"Error in the JQL Query: The operator '{op.upper()}' only supports "
                    f"EMPTY or NULL, not '{token.text}'."
                )
            empty = op == "is"
            return lambda i: (not found.get(i)) == empty
        return self.compare(found, name.text, op, [self.value()])

    # semantics

    def resolve(self, value: Value) -> str:
        if not value.function:
            return value.text
        if value.text.casefold() == "currentuser":
            return self.context.current_user
        raise JqlError(f"Error in the JQL Query: Unable to find JQL function '{value.text}()'.")

    def unsupported(self, op: str, name: str) -> JqlError:
        return JqlError(f"The operator '{op}' is not supported by the '{name}' field.")

    def compare(self, found: Field, name: str, op: str, raw: list[Value]) -> Predicate:
        if found.kind == "text":
            if op not in ("~", "!~"):
                raise self.unsupported(op, name)
            terms = _terms(raw[0].text)
            hit: Predicate = lambda i: any(_contains(v, terms) for v in found.get(i))  # noqa: E731
            return hit if op == "~" else lambda i: not hit(i)
        if op in ("~", "!~"):
            raise self.unsupported(op, name)
        if found.kind == "date":
            when = [parse_date(self.resolve(v), self.context.now) for v in raw]
            return _date_compare(found, op, when, name, self)
        if op in (">", ">=", "<", "<="):
            raise self.unsupported(op, name)
        wanted = {self.resolve(v).casefold() for v in raw}
        if found.known is not None:
            known = {k.casefold() for k in found.known(self.context)}
            for value in raw:
                if self.resolve(value).casefold() not in known:
                    raise JqlError(
                        f"The value '{value.text}' does not exist for the field '{name}'."
                    )

        def equal(issue: Issue) -> bool:
            return any(str(v).casefold() in wanted for v in found.get(issue))

        if op in ("=", "in"):
            return equal
        # Like Jira: negations never match an empty field.
        return lambda i: bool(found.get(i)) and not equal(i)


def _terms(text: str) -> list[str]:
    return [t.casefold() for t in re.findall(r"[\w*]+", text)]


def _contains(text: Any, terms: list[str]) -> bool:
    words = re.findall(r"\w+", str(text).casefold())
    for term in terms:
        if term.endswith("*"):
            if not any(w.startswith(term[:-1]) for w in words):
                return False
        elif not any(w.startswith(term) for w in words):
            # Jira stems words ("crashes" ~ "crash"); a prefix match is close enough.
            return False
    return True


def _date_compare(
    found: Field, op: str, when: list[datetime], name: str, parser: _Parser
) -> Predicate:
    checks: dict[str, Callable[[datetime, datetime], bool]] = {
        "=": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "in": lambda a, b: a == b,
    }
    if op == "not in":
        return lambda i: bool(found.get(i)) and all(v not in when for v in found.get(i))
    if op not in checks:
        raise parser.unsupported(op, name)
    check = checks[op]
    return lambda i: any(check(v, w) for v in found.get(i) for w in when)


def parse(query: str, context: Context, *, bounded: bool = True) -> Query:
    """Parse `query`; `bounded=True` rejects queries without a restriction (like /search/jql)."""
    parsed = _Parser(tokenize(query), context).query()
    if bounded and parsed.where is None:
        raise JqlError(UNBOUNDED)
    return parsed
