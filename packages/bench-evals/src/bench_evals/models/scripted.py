"""`ScriptedModel`: plays back fixed replies, for free tests of the harness."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from bench_evals.models.base import Assistant, Message, Reply, Tool, ToolChoice, Usage


@dataclass(frozen=True)
class Request:
    system: str
    messages: tuple[Message, ...]
    tools: tuple[Tool, ...]
    tool_choice: ToolChoice


Step = Assistant | Callable[[Request], Assistant]
"""A fixed reply, or a function of the request (e.g. to echo a tool result)."""


class ScriptedModel:
    provider = "scripted"

    def __init__(self, steps: Iterable[Step], model: str = "scripted"):
        self.model = model
        self.steps = list(steps)
        self.requests: list[Request] = []

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool] = (),
        tool_choice: ToolChoice = "auto",
    ) -> Reply:
        request = Request(system, tuple(messages), tuple(tools), tool_choice)
        self.requests.append(request)
        if len(self.requests) > len(self.steps):
            raise AssertionError(f"script exhausted after {len(self.steps)} replies")
        step = self.steps[len(self.requests) - 1]
        message = step(request) if callable(step) else step
        finish = "tool_calls" if message.tool_calls else "stop"
        return Reply(message, self.model, Usage(), finish)
