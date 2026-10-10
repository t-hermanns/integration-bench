"""Which GitHub App permissions each tool needs.

GitHub App permissions are levels (`issues:write` covers `issues:read`), so the server
expands granted permissions with `expand_levels`. Issue search needs no permission of its
own: GitHub filters the hits down to what the credentials can see.
"""

from bench_core.permissions import ToolAccess, expand_levels, requires

EXPAND = expand_levels

SEARCH_ISSUES = ToolAccess("github_search_issues")
GET_ISSUE = ToolAccess("github_get_issue", requires("issues:read"))
CREATE_ISSUE = ToolAccess("github_create_issue", requires("issues:write"), write=True)
ADD_ISSUE_COMMENT = ToolAccess("github_add_issue_comment", requires("issues:write"), write=True)
LIST_PULL_REQUESTS = ToolAccess("github_list_pull_requests", requires("pull_requests:read"))
GET_FILE_CONTENTS = ToolAccess("github_get_file_contents", requires("contents:read"))
