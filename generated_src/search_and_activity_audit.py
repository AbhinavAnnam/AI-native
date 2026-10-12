import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------
class SearchServiceUnavailableException(Exception):
    """Raised when the search service cannot be reached or fails."""
    pass


class ActivityRecordPersistenceException(Exception):
    """Raised when persisting an activity record fails."""
    pass


# ----------------------------------------------------------------------
# Data Models
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ActivityRecord:
    """Immutable audit activity record."""
    record_id: str
    organization_id: str
    actor_id: Optional[str]  # None for system actions
    action_type: str
    target_id: str
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PaginatedSearchResults:
    """Container for paginated search results."""
    items: List[Dict[str, Any]]
    total: int
    page: int
    limit: int


@dataclass
class PaginatedActivityFeed:
    """Container for paginated activity feed."""
    items: List[ActivityRecord]
    total: int
    page: int
    limit: int


# ----------------------------------------------------------------------
# In-memory stores (simulating persistence & search index)
# ----------------------------------------------------------------------
_USERS: Dict[str, Dict[str, Any]] = {}
_ORGANIZATIONS: Dict[str, Dict[str, Any]] = {}
_PROJECTS: Dict[str, Dict[str, Any]] = {}
_WORK_ITEMS: Dict[str, Dict[str, Any]] = {}
_COMMENTS: Dict[str, Dict[str, Any]] = {}

# Append-only activity log
_ACTIVITY_LOG: List[ActivityRecord] = []

# Simple search index: document_id -> {"type": str, "organization_id": str, "content": dict}
_SEARCH_INDEX: Dict[str, Dict[str, Any]] = {}

# Lock for thread-safe mutations
_INDEX_LOCK = threading.Lock()
_ACTIVITY_LOCK = threading.Lock()


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def _user_belongs_to_organization(user_id: str, organization_id: str) -> bool:
    user = _USERS.get(user_id)
    return bool(user and user.get("organization_id") == organization_id)


def _document_accessible_to_user(
    user_id: str, organization_id: str, doc: Dict[str, Any]
) -> bool:
    """Basic access check: user must belong to the same organization."""
    return _user_belongs_to_organization(user_id, organization_id)


def _matches_query(content: Dict[str, Any], query: str) -> bool:
    """Case-insensitive partial match across all string values in the content."""
    lowered = query.lower()
    for value in content.values():
        if isinstance(value, str) and lowered in value.lower():
            return True
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and lowered in item.lower():
                    return True
    return False


def _paginate(items: List[Any], page: int, limit: int) -> (List[Any], int):
    """Return a slice of items for the requested page and the total count."""
    if page < 1:
        page = 1
    if limit < 1:
        limit = 10
    total = len(items)
    start = (page - 1) * limit
    end = start + limit
    return items[start:end], total


# ----------------------------------------------------------------------
# Core Functions
# ----------------------------------------------------------------------
def search_content(
    user_id: str,
    organization_id: str,
    query_phrase: str,
    page: int,
    limit: int,
) -> PaginatedSearchResults:
    """
    Executes a paginated full-text and partial-match search across entities
    that the user can access within the specified organization.
    """
    if not _user_belongs_to_organization(user_id, organization_id):
        raise SearchServiceUnavailableException(
            f"User {user_id} does not belong to organization {organization_id}"
        )

    try:
        # Gather candidate documents
        candidates = []
        for doc_id, entry in _SEARCH_INDEX.items():
            if entry["organization_id"] != organization_id:
                continue
            if not _document_accessible_to_user(user_id, organization_id, entry):
                continue
            if _matches_query(entry["content"], query_phrase):
                # Return a shallow copy to avoid external mutation
                candidates.append(
                    {
                        "document_id": doc_id,
                        "document_type": entry["type"],
                        "content": entry["content"],
                    }
                )
        # Sort by relevance placeholder (here by document_id for determinism)
        candidates.sort(key=lambda x: x["document_id"])

        paged_items, total = _paginate(candidates, page, limit)

        return PaginatedSearchResults(
            items=paged_items,
            total=total,
            page=page,
            limit=limit,
        )
    except Exception as exc:
        raise SearchServiceUnavailableException(
            f"Search failed due to unexpected error: {exc}"
        ) from exc


def index_document(
    document_id: str,
    document_type: str,
    content: Dict[str, Any],
) -> None:
    """
    Indexes or updates an entity's content in the search index.
    The operation is treated as asynchronous; for this implementation we
    perform the update immediately but protect it with a lock to simulate
    thread-safe eventual consistency.
    """
    if not document_id or not document_type or not isinstance(content, dict):
        raise ValueError("Invalid arguments supplied to index_document.")

    try:
        with _INDEX_LOCK:
            # Simulate async delay (non-blocking in real world)
            _SEARCH_INDEX[document_id] = {
                "type": document_type,
                "organization_id": content.get("organization_id", ""),
                "content": content,
            }
    except Exception as exc:
        raise SearchServiceUnavailableException(
            f"Failed to index document {document_id}: {exc}"
        ) from exc


def record_activity(
    organization_id: str,
    actor_id: Optional[str],
    action_type: str,
    target_id: str,
    metadata: Dict[str, Any],
) -> ActivityRecord:
    """
    Persists an immutable activity event record.
    The function never mutates existing records; it only appends a new one.
    """
    if not organization_id or not action_type or not target_id:
        raise ValueError("organization_id, action_type, and target_id are required.")

    try:
        record = ActivityRecord(
            record_id=str(uuid.uuid4()),
            organization_id=organization_id,
            actor_id=actor_id,
            action_type=action_type,
            target_id=target_id,
            metadata=metadata or {},
        )
        with _ACTIVITY_LOCK:
            _ACTIVITY_LOG.append(record)
        return record
    except Exception as exc:
        raise ActivityRecordPersistenceException(
            f"Unable to persist activity record: {exc}"
        ) from exc


def get_activity_feed(
    organization_id: str,
    target_id: Optional[str],
    page: int,
    limit: int,
) -> PaginatedActivityFeed:
    """
    Retrieves a paginated audit feed filtered by organization and optionally
    by a specific target resource.
    """
    try:
        # Filter by organization and optional target
        filtered = [
            rec
            for rec in _ACTIVITY_LOG
            if rec.organization_id == organization_id
            and (target_id is None or rec.target_id == target_id)
        ]

        # Most recent first
        filtered.sort(key=lambda r: r.timestamp, reverse=True)

        paged_items, total = _paginate(filtered, page, limit)

        return PaginatedActivityFeed(
            items=paged_items,
            total=total,
            page=page,
            limit=limit,
        )
    except Exception as exc:
        raise ActivityRecordPersistenceException(
            f"Failed to retrieve activity feed: {exc}"
        ) from exc


# ----------------------------------------------------------------------
# Example Stubs for Populating In-Memory Stores (not part of core API)
# ----------------------------------------------------------------------
def _populate_example_data() -> None:
    """Utility to seed the in-memory structures for manual testing."""
    # Users
    _USERS["u1"] = {"user_id": "u1", "organization_id": "org1", "display_name": "Alice"}
    _USERS["u2"] = {"user_id": "u2", "organization_id": "org1", "display_name": "Bob"}

    # Organization
    _ORGANIZATIONS["org1"] = {"organization_id": "org1", "name": "Acme Corp"}

    # Projects
    _PROJECTS["p1"] = {
        "project_id": "p1",
        "organization_id": "org1",
        "title": "Project Alpha",
    }

    # Work items
    _WORK_ITEMS["w1"] = {
        "work_item_id": "w1",
        "project_id": "p1",
        "title": "Implement feature X",
        "description": "Details about feature X implementation.",
    }

    # Comments
    _COMMENTS["c1"] = {
        "comment_id": "c1",
        "work_item_id": "w1",
        "author_id": "u1",
        "text": "Initial design looks good.",
    }

    # Index documents
    index_document(
        document_id="proj:p1",
        document_type="project",
        content={
            "organization_id": "org1",
            "title": "Project Alpha",
            "description": "Top secret project.",
        },
    )
    index_document(
        document_id="wi:w1",
        document_type="work_item",
        content={
            "organization_id": "org1",
            "title": "Implement feature X",
            "description": "Details about feature X implementation.",
        },
    )
    index_document(
        document_id="cmt:c1",
        document_type="comment",
        content={
            "organization_id": "org1",
            "text": "Initial design looks good.",
            "author_id": "u1",
        },
    )

# Populate example data on module import for interactive sessions
_populate_example_data()