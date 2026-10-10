import json

import httpx2
import pytest

from bench_evals.models import (
    PROVIDERS,
    Assistant,
    CohereModel,
    ModelError,
    OpenAICompatModel,
    Request,
    ScriptedModel,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
    User,
)

pytestmark = pytest.mark.anyio

WEATHER = Tool(
    "get_weather",
    "Current weather for a city.",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


class Api:
    """Answers each request with the next response and keeps the request bodies."""

    def __init__(self, *responses: httpx2.Response):
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    @property
    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self)

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)


# --- ScriptedModel -------------------------------------------------------------------------


async def test_scripted_model_plays_its_steps_and_records_requests() -> None:
    call = ToolCall("c1", "get_weather", {"city": "Berlin"})

    def echo(request: Request) -> Assistant:
        last = request.messages[-1]
        assert isinstance(last, ToolResult)
        return Assistant(f"It is {last.content}.")

    model = ScriptedModel([Assistant(tool_calls=(call,)), echo])
    first = await model.complete("sys", [User("weather?")], [WEATHER], "required")
    assert first.message.tool_calls == (call,)
    assert first.finish_reason == "tool_calls"
    second = await model.complete(
        "sys", [User("weather?"), first.message, ToolResult("c1", "x", "14 C")]
    )
    assert (second.message.text, second.finish_reason) == ("It is 14 C.", "stop")
    assert model.requests[0].tools == (WEATHER,)
    assert model.requests[0].tool_choice == "required"
    with pytest.raises(AssertionError, match="exhausted after 2"):
        await model.complete("sys", [])


# --- Cohere --------------------------------------------------------------------------------


def cohere_reply(message: dict, finish: str = "TOOL_CALL") -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "x",
            "finish_reason": finish,
            "message": {"role": "assistant", **message},
            "usage": {
                "billed_units": {"input_tokens": 10, "output_tokens": 5},
                "tokens": {"input_tokens": 1478, "output_tokens": 50},
            },
        },
        headers={"x-trial-endpoint-call-limit": "20", "x-endpoint-monthly-call-limit": "1000"},
    )


async def test_cohere_round_trip_with_tool_plan() -> None:
    raw_call = {
        "id": "get_weather_q59",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Berlin"}'},
    }
    api = Api(
        cohere_reply({"tool_plan": "I will look it up.", "tool_calls": [raw_call]}),
        cohere_reply({"content": [{"type": "text", "text": "Cloudy, 14 C."}]}, "COMPLETE"),
    )
    model = CohereModel("key", "command-a-03-2025", transport=api.transport)

    reply = await model.complete("Be brief.", [User("Weather in Berlin?")], [WEATHER])
    assert reply.message == Assistant(
        "",
        (ToolCall("get_weather_q59", "get_weather", {"city": "Berlin"}, raw_call),),
        "I will look it up.",
    )
    assert reply.usage == Usage(1478, 50)
    assert reply.finish_reason == "TOOL_CALL"
    assert reply.limits == {
        "x-trial-endpoint-call-limit": "20",
        "x-endpoint-monthly-call-limit": "1000",
    }

    request = api.requests[0]
    assert str(request.url) == "https://api.cohere.com/v2/chat"
    assert request.headers["authorization"] == "Bearer key"
    body = api.body(0)
    assert body["model"] == "command-a-03-2025"
    assert (body["temperature"], body["seed"]) == (0, 0)
    assert "tool_choice" not in body  # Cohere has no "auto": leaving it out means auto
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather in Berlin?"},
    ]
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a city.",
                "parameters": WEATHER.parameters,
            },
        }
    ]

    history = [
        User("Weather in Berlin?"),
        reply.message,
        ToolResult("get_weather_q59", "get_weather", '{"temp_c": 14}'),
    ]
    final = await model.complete("Be brief.", history, [WEATHER], "none")
    assert final.message == Assistant("Cloudy, 14 C.")
    body = api.body(1)
    assert body["tool_choice"] == "NONE"
    assert body["messages"][2:] == [
        {"role": "assistant", "tool_calls": [raw_call], "tool_plan": "I will look it up."},
        {"role": "tool", "tool_call_id": "get_weather_q59", "content": '{"temp_c": 14}'},
    ]


async def test_cohere_required_tool_choice_and_plain_assistant_turns() -> None:
    api = Api(cohere_reply({"content": [{"type": "text", "text": "ok"}]}, "COMPLETE"))
    model = CohereModel("key", "m", transport=api.transport)
    await model.complete("s", [User("a"), Assistant("b"), User("c")], [WEATHER], "required")
    body = api.body()
    assert body["tool_choice"] == "REQUIRED"
    assert body["messages"][2] == {"role": "assistant", "content": "b"}


# --- OpenAI-compatible (Gemini, Groq) ------------------------------------------------------


def openai_reply(message: dict, finish: str = "tool_calls", **headers: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "model": "served-model",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 287, "completion_tokens": 26},
        },
        headers=headers,
    )


async def test_openai_compat_round_trip_echoes_raw_tool_calls() -> None:
    raw_call = {
        "extra_content": {"google": {"thought_signature": "sig=="}},
        "function": {"arguments": '{"city":"Berlin"}', "name": "get_weather"},
        "id": "call_1",
        "type": "function",
    }
    api = Api(
        openai_reply({"role": "assistant", "content": None, "tool_calls": [raw_call]}),
        openai_reply({"role": "assistant", "content": "Cloudy."}, "stop"),
    )
    model = OpenAICompatModel(
        "groq", "https://api.example/v1/", "key", "m", transport=api.transport
    )

    reply = await model.complete("sys", [User("Weather?")], [WEATHER])
    assert reply.message.tool_calls == (
        ToolCall("call_1", "get_weather", {"city": "Berlin"}, raw_call),
    )
    assert reply.message.text == ""
    assert (reply.model, reply.usage, reply.finish_reason) == (
        "served-model",
        Usage(287, 26),
        "tool_calls",
    )
    assert str(api.requests[0].url) == "https://api.example/v1/chat/completions"
    body = api.body(0)
    assert body["tool_choice"] == "auto"
    assert (body["temperature"], body["seed"]) == (0, 0)

    history = [User("Weather?"), reply.message, ToolResult("call_1", "get_weather", "14 C")]
    final = await model.complete("sys", history, [WEATHER])
    assert final.message == Assistant("Cloudy.")
    assert api.body(1)["messages"][2:] == [
        {"role": "assistant", "content": None, "tool_calls": [raw_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "14 C"},
    ]


async def test_openai_compat_fills_missing_ids_and_keeps_bad_arguments() -> None:
    calls = [
        {"type": "function", "function": {"name": "a", "arguments": "{not json"}},
        {"id": "", "type": "function", "function": {"name": "b", "arguments": "[1]"}},
    ]
    api = Api(openai_reply({"role": "assistant", "content": "", "tool_calls": calls}))
    model = OpenAICompatModel("groq", "https://api.example/v1", "key", "m", transport=api.transport)
    reply = await model.complete("sys", [User("x")])
    first, second = reply.message.tool_calls
    assert (first.id, first.arguments) == ("call_0", {"_unparsed": "{not json"})
    assert (second.id, second.arguments) == ("call_1", {"_unparsed": "[1]"})
    assert second.raw is not None
    assert second.raw["id"] == "call_1"
    assert "tools" not in api.body()
    assert "tool_choice" not in api.body()


async def test_tool_calls_without_raw_are_serialised() -> None:
    api = Api(openai_reply({"role": "assistant", "content": "ok"}, "stop"))
    model = OpenAICompatModel("groq", "https://api.example/v1", "key", "m", transport=api.transport)
    history = [Assistant("thinking", (ToolCall("c", "get_weather", {"city": "Rome"}),))]
    await model.complete("sys", history)
    assert api.body()["messages"][1] == {
        "role": "assistant",
        "content": "thinking",
        "tool_calls": [
            {
                "id": "c",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
            }
        ],
    }


# --- Providers -----------------------------------------------------------------------------


def gemini_403() -> httpx2.Response:
    error = {"error": {"code": 403, "message": "Permission denied", "status": "PERMISSION_DENIED"}}
    return httpx2.Response(403, json=[error])


async def test_gemini_retries_one_spurious_403_and_sends_no_seed() -> None:
    api = Api(gemini_403(), openai_reply({"role": "assistant", "content": "hi"}, "stop"))
    model = PROVIDERS["gemini"].build({"GEMINI_API_KEY": "g"}, api.transport)
    reply = await model.complete("sys", [User("x")])
    assert reply.message.text == "hi"
    assert len(api.requests) == 2
    request = api.requests[0]
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert request.headers["authorization"] == "Bearer g"
    assert api.body(0)["model"] == "gemini-3.5-flash-lite"
    assert "seed" not in api.body(0)


async def test_gemini_gives_up_after_the_second_403() -> None:
    api = Api(gemini_403(), gemini_403())
    model = PROVIDERS["gemini"].build({"GEMINI_API_KEY": "g"}, api.transport)
    with pytest.raises(ModelError) as caught:
        await model.complete("sys", [User("x")])
    assert (caught.value.provider, caught.value.status) == ("gemini", 403)
    assert caught.value.detail == "Permission denied"


async def test_groq_does_not_retry_and_reports_retry_after() -> None:
    api = Api(
        httpx2.Response(
            429,
            json={"error": {"message": "Rate limit reached", "type": "tokens"}},
            headers={"retry-after": "7", "x-ratelimit-remaining-tokens": "0"},
        )
    )
    model = PROVIDERS["groq"].build({"GROQ_API_KEY": "q", "GROQ_MODEL": "other"}, api.transport)
    assert model.model == "other"
    with pytest.raises(ModelError) as caught:
        await model.complete("sys", [User("x")])
    assert (caught.value.status, caught.value.retry_after) == (429, 7.0)
    assert str(caught.value) == "groq: HTTP 429: Rate limit reached"
    assert len(api.requests) == 1


async def test_cohere_errors_and_non_json_bodies() -> None:
    api = Api(
        httpx2.Response(401, json={"message": "invalid api token"}),
        httpx2.Response(502, text="bad gateway"),
    )
    model = PROVIDERS["cohere"].build({"COHERE_API_KEY": "c"}, api.transport)
    assert model.model == "command-a-03-2025"
    with pytest.raises(ModelError, match="cohere: HTTP 401: invalid api token"):
        await model.complete("sys", [User("x")])
    with pytest.raises(ModelError, match="HTTP 502: bad gateway"):
        await model.complete("sys", [User("x")])


def test_build_needs_the_api_key() -> None:
    with pytest.raises(KeyError, match="GROQ_API_KEY"):
        PROVIDERS["groq"].build({})


def test_cohere_tool_results_get_string_ids() -> None:
    from bench_evals.models.http import string_ids

    assert json.loads(string_ids('{"id": 42, "body": "hi"}')) == {"id": "42", "body": "hi"}
    assert json.loads(string_ids('[{"id": 1}, {"id": "a"}]')) == [{"id": "1"}, {"id": "a"}]
    for unchanged in ('{"id": "x"}', '{"key": "BENCH-1"}', "Unknown tool: x", "[1, 2]"):
        assert string_ids(unchanged) == unchanged
