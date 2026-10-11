"""Which Todoist scopes each tool needs.

`EXPAND` derives implied scopes from the granted ones before tools are filtered. The
default expands nothing; use `expand_levels` when `x:write` covers `x:read`, or
`expand_implied` with a table when broad scopes cover narrow ones.
"""

from bench_core.permissions import ToolAccess, no_expansion, requires

EXPAND = no_expansion

READ = "data:read"
WRITE = "data:read_write"

LIST_TASKS = ToolAccess("todoist_list_tasks", requires(READ))
GET_TASK = ToolAccess("todoist_get_task", requires(READ))
CREATE_TASK = ToolAccess("todoist_create_task", requires(WRITE), write=True)
