"""Which Todoist scopes each tool needs.

`EXPAND` derives implied scopes from the granted ones before tools are filtered. The
default expands nothing; use `expand_levels` when `x:write` covers `x:read`, or
`expand_implied` with a table when broad scopes cover narrow ones.
"""

from bench_core.permissions import ToolAccess, no_expansion, requires

EXPAND = no_expansion

READ = "items:read"
WRITE = "items:write"

LIST_ITEMS = ToolAccess("todoist_list_items", requires(READ))
GET_ITEM = ToolAccess("todoist_get_item", requires(READ))
CREATE_ITEM = ToolAccess("todoist_create_item", requires(WRITE), write=True)
