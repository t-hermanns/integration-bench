"""HTTP layer: `ApiClient` with retry policies, a per-tool-call wait budget and pagination."""

from bench_core.http.client import ApiClient, RateLimited, WaitBudget
from bench_core.http.pagination import (
    BodyTokenPaginator,
    LinkHeaderPaginator,
    PageResponse,
    PaginationLoop,
    Paginator,
    cursor,
    decode_cursor,
    encode_cursor,
    fetch_page,
    iter_pages,
    link_header,
    next_cursor,
    next_page_token,
    page_token,
)
from bench_core.http.retry import (
    DriveRetryPolicy,
    GitHubRetryPolicy,
    JiraRetryPolicy,
    RetryPolicy,
    SlackRetryPolicy,
    parse_retry_after,
)

__all__ = [
    "ApiClient",
    "BodyTokenPaginator",
    "DriveRetryPolicy",
    "GitHubRetryPolicy",
    "JiraRetryPolicy",
    "LinkHeaderPaginator",
    "PageResponse",
    "PaginationLoop",
    "Paginator",
    "RateLimited",
    "RetryPolicy",
    "SlackRetryPolicy",
    "WaitBudget",
    "cursor",
    "decode_cursor",
    "encode_cursor",
    "fetch_page",
    "iter_pages",
    "link_header",
    "next_cursor",
    "next_page_token",
    "page_token",
    "parse_retry_after",
]
