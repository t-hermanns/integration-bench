"""Which Jira OAuth 2.0 (3LO) scopes each tool needs.

Classic scopes: `read:jira-work` covers issues, comments and transitions for reading,
`write:jira-work` for creating and changing them. Jira does not let the write scope imply
the read scope, so nothing is expanded. API tokens carry no scope list; with them every
tool stays registered and Jira's project permissions decide.
"""

from bench_core.permissions import ToolAccess, no_expansion, requires

EXPAND = no_expansion

READ = "read:jira-work"
WRITE = "write:jira-work"
USER = "read:jira-user"
OFFLINE = "offline_access"
LOGIN_SCOPES = (READ, WRITE, USER, OFFLINE)
"""What `bench auth login jira` asks for (`offline_access` yields a refresh token)."""

SEARCH_ISSUES = ToolAccess("jira_search_issues", requires(READ))
GET_ISSUE = ToolAccess("jira_get_issue", requires(READ))
CREATE_ISSUE = ToolAccess("jira_create_issue", requires(WRITE), write=True)
ADD_COMMENT = ToolAccess("jira_add_comment", requires(WRITE), write=True)
TRANSITION_ISSUE = ToolAccess("jira_transition_issue", requires(READ, WRITE), write=True)
