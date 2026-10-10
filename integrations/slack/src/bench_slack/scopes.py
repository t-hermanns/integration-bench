"""Which Slack scopes each tool needs.

Slack scopes do not imply each other (`chat:write` does not cover reading, and
`channels:history` does not cover `channels:read`), so nothing is expanded. The bot token
(`bench-ci`) and the user token (`bench-dev`) get the same three scopes.
"""

from bench_core.permissions import ToolAccess, no_expansion, requires

EXPAND = no_expansion

CHANNELS_READ = "channels:read"
CHANNELS_HISTORY = "channels:history"
CHAT_WRITE = "chat:write"
ALL = (CHANNELS_READ, CHANNELS_HISTORY, CHAT_WRITE)

LIST_CHANNELS = ToolAccess("slack_list_channels", requires(CHANNELS_READ))
READ_CHANNEL_HISTORY = ToolAccess("slack_read_channel_history", requires(CHANNELS_HISTORY))
READ_THREAD = ToolAccess("slack_read_thread", requires(CHANNELS_HISTORY))
POST_MESSAGE = ToolAccess("slack_post_message", requires(CHAT_WRITE), write=True)
REPLY_IN_THREAD = ToolAccess(
    "slack_reply_in_thread", requires(CHAT_WRITE, CHANNELS_HISTORY), write=True
)
