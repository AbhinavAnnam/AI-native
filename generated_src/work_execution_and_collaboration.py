import os
import uuid
import datetime
from typing import Optional, Dict, Any, List, BinaryIO
from io import BytesIO
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Exception Definitions
# ----------------------------------------------------------------------
class FileTooLargeException(Exception):
    """Raised when an uploaded file exceeds the allowed size limit."""
    pass

class PathTraversalException(Exception):
    """Raised when a filename attempts directory traversal."""
    pass

class AttachmentNotFoundException(Exception):
    """Raised when an attachment cannot be located."""
    pass

class UnauthorizedAttachmentAccessException(Exception):
    """Raised when a user attempts to access an attachment without permission."""
    pass

class DashboardDataFetchException(Exception):
    """Raised when dashboard aggregation fails."""
    pass

class UnauthorizedAccessException(Exception):
    """Raised when a user lacks permission to access a requested object."""
    pass

class ObjectNotFoundException(Exception):
    """Raised when a referenced object does not exist."""
    pass

# ----------------------------------------------------------------------
# Data Models
# ----------------------------------------------------------------------
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@dataclass
class AttachmentMetadata:
    attachment_id: str
    work_item_id: str
    comment_id: Optional[str]
    filename: str
    size: int
    uploaded_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

@dataclass
class ProjectSummary:
    project_id: str
    status: str

@dataclass
class WorkItemSummary:
    work_item_id: str
    status: str
    due_date: Optional[datetime.datetime]

@dataclass
class DashboardSummary:
    active_projects: List[ProjectSummary]
    assigned_work_items: List[WorkItemSummary]

# ----------------------------------------------------------------------
# In-memory mock storage (for illustration purposes only)
# ----------------------------------------------------------------------
# Users -> organization mapping
USER_ORG: Dict[str, str] = {}

# Projects storage
PROJECTS: Dict[str, Dict[str, Any]] = {}

# Work items storage
WORK_ITEMS: Dict[str, Dict[str, Any]] = {}

# Comments storage
COMMENTS: Dict[str, Dict[str, Any]] = {}

# Attachments storage
ATTACHMENTS: Dict[str, Dict[str, Any]] = {}

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def _sanitize_filename(filename: str) -> str:
    """
    Remove any path components and ensure the filename does not contain
    traversal sequences.
    """
    base = os.path.basename(filename)
    if ".." in base or base.startswith("/") or base.startswith("\\"):
        raise PathTraversalException(f"Unsafe filename detected: {filename}")
    return base

def _check_file_size(file_data: bytes) -> None:
    if len(file_data) > MAX_FILE_SIZE:
        raise FileTooLargeException(f"File size {len(file_data)} exceeds limit of {MAX_FILE_SIZE} bytes.")

def _get_user_organization(user_id: str) -> str:
    org = USER_ORG.get(user_id)
    if not org:
        raise UnauthorizedAccessException(f"User {user_id} has no organization association.")
    return org

def _fetch_object(object_id: str, object_type: str) -> Dict[str, Any]:
    if object_type == "project":
        obj = PROJECTS.get(object_id)
    elif object_type == "work_item":
        obj = WORK_ITEMS.get(object_id)
    elif object_type == "comment":
        obj = COMMENTS.get(object_id)
    elif object_type == "attachment":
        obj = ATTACHMENTS.get(object_id)
    else:
        raise ObjectNotFoundException(f"Unsupported object type: {object_type}")

    if obj is None:
        raise ObjectNotFoundException(f"{object_type.capitalize()} with id {object_id} not found.")
    return obj

# ----------------------------------------------------------------------
# Core Functions
# ----------------------------------------------------------------------
def verify_object_access(user_id: str, object_id: str, object_type: str) -> bool:
    """
    Verifies if a user has permission to view or interact with a target object.
    The check follows the hierarchy: tenant (organization) -> project membership.
    """
    try:
        user_org = _get_user_organization(user_id)
        obj = _fetch_object(object_id, object_type)

        # Direct project check
        if object_type == "project":
            if obj["organization_id"] != user_org:
                raise UnauthorizedAccessException("User belongs to a different organization.")
            return user_id in obj["members"]

        # Work item inherits project permissions
        if object_type == "work_item":
            project = _fetch_object(obj["project_id"], "project")
            if project["organization_id"] != user_org:
                raise UnauthorizedAccessException("User belongs to a different organization.")
            return user_id in project["members"]

        # Comment inherits work item permissions
        if object_type == "comment":
            work_item = _fetch_object(obj["work_item_id"], "work_item")
            project = _fetch_object(work_item["project_id"], "project")
            if project["organization_id"] != user_org:
                raise UnauthorizedAccessException("User belongs to a different organization.")
            return user_id in project["members"]

        # Attachment inherits work item (or comment) permissions
        if object_type == "attachment":
            work_item_id = obj["work_item_id"]
            work_item = _fetch_object(work_item_id, "work_item")
            project = _fetch_object(work_item["project_id"], "project")
            if project["organization_id"] != user_org:
                raise UnauthorizedAccessException("User belongs to a different organization.")
            return user_id in project["members"]

        # Fallback deny
        return False
    except (ObjectNotFoundException, UnauthorizedAccessException):
        return False

def upload_attachment(
    work_item_id: str,
    file_data: bytes,
    filename: str,
    comment_id: Optional[str] = None
) -> AttachmentMetadata:
    """
    Uploads and attaches a file to a work item or comment after performing size
    and safety validations.
    """
    # Validate work item existence
    if work_item_id not in WORK_ITEMS or WORK_ITEMS[work_item_id].get("deleted", False):
        raise ObjectNotFoundException(f"Work item {work_item_id} not found or deleted.")

    # If comment_id is provided, validate comment existence and linkage
    if comment_id is not None:
        comment = COMMENTS.get(comment_id)
        if not comment or comment.get("deleted", False):
            raise ObjectNotFoundException(f"Comment {comment_id} not found or deleted.")
        if comment["work_item_id"] != work_item_id:
            raise ValueError("Comment does not belong to the specified work item.")

    # Size and filename checks
    _check_file_size(file_data)
    safe_name = _sanitize_filename(filename)

    # Create attachment record
    attachment_id = str(uuid.uuid4())
    attachment_record = {
        "attachment_id": attachment_id,
        "work_item_id": work_item_id,
        "comment_id": comment_id,
        "filename": safe_name,
        "size": len(file_data),
        "content": file_data,
        "uploaded_at": datetime.datetime.utcnow(),
        "deleted": False,
        # In a real system, uploader_user_id would be captured from auth context.
        "uploader_user_id": None,
    }
    ATTACHMENTS[attachment_id] = attachment_record

    return AttachmentMetadata(
        attachment_id=attachment_id,
        work_item_id=work_item_id,
        comment_id=comment_id,
        filename=safe_name,
        size=len(file_data),
        uploaded_at=attachment_record["uploaded_at"]
    )

def download_attachment(attachment_id: str, user_id: str) -> BinaryIO:
    """
    Authorizes user access and streams requested attachment binary content.
    """
    attachment = ATTACHMENTS.get(attachment_id)
    if not attachment or attachment.get("deleted", False):
        raise AttachmentNotFoundException(f"Attachment {attachment_id} not found or deleted.")

    # Verify access based on the underlying work item
    if not verify_object_access(user_id, attachment["work_item_id"], "work_item"):
        raise UnauthorizedAttachmentAccessException("User lacks permission to download this attachment.")

    # Return a BytesIO stream for the caller to read
    return BytesIO(attachment["content"])

def delete_attachment(attachment_id: str, user_id: str) -> None:
    """
    Removes attachment file from normal access while maintaining historical record of removal.
    """
    attachment = ATTACHMENTS.get(attachment_id)
    if not attachment:
        raise AttachmentNotFoundException(f"Attachment {attachment_id} not found.")

    # Verify delete permission (same as view permission for this example)
    if not verify_object_access(user_id, attachment["work_item_id"], "work_item"):
        raise UnauthorizedAttachmentAccessException("User lacks permission to delete this attachment.")

    # Soft delete: mark as deleted and clear binary content to free memory
    attachment["deleted"] = True
    attachment["deleted_at"] = datetime.datetime.utcnow()
    attachment["content"] = b""  # Remove actual data but keep metadata

def get_user_dashboard(user_id: str, organization_id: str) -> DashboardSummary:
    """
    Fetches aggregated summary of active projects and assigned work items for a user.
    """
    try:
        # Verify user belongs to the organization
        if _get_user_organization(user_id) != organization_id:
            raise UnauthorizedAccessException("User does not belong to the specified organization.")

        # Active projects where user is a member
        active_projects = [
            ProjectSummary(project_id=pid, status=proj["status"])
            for pid, proj in PROJECTS.items()
            if proj["status"] == "active"
            and organization_id == proj["organization_id"]
            and user_id in proj["members"]
        ]

        # Work items assigned to the user that are not completed and belong to active projects
        assigned_items = []
        for wid, wi in WORK_ITEMS.items():
            if wi.get("deleted", False):
                continue
            if wi.get("assignee_user_id") != user_id:
                continue
            # Ensure the work item belongs to an active project the user can see
            project = PROJECTS.get(wi["project_id"])
            if not project or project["status"] != "active":
                continue
            if user_id not in project["members"]:
                continue
            assigned_items.append(
                WorkItemSummary(
                    work_item_id=wid,
                    status=wi["status"],
                    due_date=wi.get("due_date")
                )
            )

        return DashboardSummary(
            active_projects=active_projects,
            assigned_work_items=assigned_items
        )
    except Exception as exc:
        raise DashboardDataFetchException(f"Failed to build dashboard: {exc}") from exc