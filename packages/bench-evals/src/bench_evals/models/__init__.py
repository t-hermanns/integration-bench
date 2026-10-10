"""Model adapters behind one provider-neutral interface."""

from bench_evals.models.base import (
    Assistant,
    Message,
    Model,
    ModelError,
    Reply,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    Usage,
    User,
)
from bench_evals.models.http import CohereModel, OpenAICompatModel
from bench_evals.models.providers import PROVIDERS, Provider
from bench_evals.models.scripted import Request, ScriptedModel

__all__ = [
    "PROVIDERS",
    "Assistant",
    "CohereModel",
    "Message",
    "Model",
    "ModelError",
    "OpenAICompatModel",
    "Provider",
    "Reply",
    "Request",
    "ScriptedModel",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolResult",
    "Usage",
    "User",
]
