import threading
from dataclasses import dataclass
from typing import Dict, Optional, Any


# ==============================
# Exception Definitions
# ==============================
class OrganizationSettingsNotFoundException(Exception):
    """Raised when the requested organization settings cannot be found."""
    pass


class UnauthorizedSettingsModificationException(Exception):
    """Raised when a user attempts to modify organization settings without sufficient privileges."""
    pass


class UnauthorizedAdministrationException(Exception):
    """Raised when a user attempts an administrative action without proper rights."""
    pass


class SessionExpiredException(Exception):
    """Raised when a required session or confirmation token has expired or is missing."""
    pass


class InvalidConfirmationException(Exception):
    """Raised when a provided confirmation token does not match the expected value."""
    pass


class InvalidOwnerRecipientException(Exception):
    """Raised when the intended new owner is not a valid, active member of the organization."""
    pass


class ConcurrentOwnershipTransferException(Exception):
    """Raised when an ownership transfer is already in progress for the organization."""
    pass


# ==============================
# Data Models
# ==============================
@dataclass
class OrganizationSettings:
    organization_id: str
    name: str
    timezone: str


@dataclass
class OwnershipTransferResult:
    success: bool
    message: str


# ==============================
# In-Memory Mock Storage (for illustration)
# ==============================
# organization_id -> OrganizationSettings
_ORG_SETTINGS_STORE: Dict[str, OrganizationSettings] = {}

# organization_id -> {user_id: role}
_ORG_MEMBERS_STORE: Dict[str, Dict[str, str]] = {}

# (organization_id, target_user_id) -> confirmation_token
_CONFIRMATION_TOKENS: Dict[tuple, str] = {}

# organization_id -> threading.Lock (to guard concurrent ownership transfers)
_OWNERSHIP_LOCKS: Dict[str, threading.Lock] = {}


# ==============================
# Helper Functions
# ==============================
def _get_org_settings(org_id: str) -> OrganizationSettings:
    settings = _ORG_SETTINGS_STORE.get(org_id)
    if not settings:
        raise OrganizationSettingsNotFoundException(f"Settings for organization '{org_id}' not found.")
    return settings


def _is_admin(user_id: str, org_id: str) -> bool:
    role = _ORG_MEMBERS_STORE.get(org_id, {}).get(user_id)
    return role in {"Owner", "Admin"}


def _is_owner(user_id: str, org_id: str) -> bool:
    return _ORG_MEMBERS_STORE.get(org_id, {}).get(user_id) == "Owner"


def _ensure_member_exists(org_id: str, user_id: str) -> None:
    if user_id not in _ORG_MEMBERS_STORE.get(org_id, {}):
        raise UnauthorizedAdministrationException(
            f"User '{user_id}' is not a member of organization '{org_id}'."
        )


def _verify_confirmation_token(org_id: str, target_user_id: str, token: str) -> None:
    expected = _CONFIRMATION_TOKENS.get((org_id, target_user_id))
    if expected is None:
        raise SessionExpiredException(
            f"No pending confirmation found for user '{target_user_id}' in organization '{org_id}'."
        )
    if expected != token:
        raise InvalidConfirmationException("Provided confirmation token is invalid.")
    # Token is valid; remove it to prevent reuse
    del _CONFIRMATION_TOKENS[(org_id, target_user_id)]


def _get_ownership_lock(org_id: str) -> threading.Lock:
    if org_id not in _OWNERSHIP_LOCKS:
        _OWNERSHIP_LOCKS[org_id] = threading.Lock()
    return _OWNERSHIP_LOCKS[org_id]


# ==============================
# Core Functions
# ==============================
def update_organization_settings(
    organization_id: str,
    admin_user_id: str,
    settings: Dict[str, Any],
) -> OrganizationSettings:
    """
    Updates organization workspace settings such as name and default timezone.

    Raises:
        OrganizationSettingsNotFoundException: If the organization does not exist.
        UnauthorizedSettingsModificationException: If the user is not an Owner or Admin.
    """
    # Verify organization exists
    current_settings = _get_org_settings(organization_id)

    # Verify admin privileges
    if not _is_admin(admin_user_id, organization_id):
        raise UnauthorizedSettingsModificationException(
            f"User '{admin_user_id}' lacks permission to modify settings for organization '{organization_id}'."
        )

    # Validate incoming settings
    name = settings.get("name")
    timezone = settings.get("timezone")
    if not isinstance(name, str) or not isinstance(timezone, str):
        raise ValueError("Both 'name' and 'timezone' must be provided as strings.")

    # Apply updates
    updated = OrganizationSettings(
        organization_id=organization_id,
        name=name,
        timezone=timezone,
    )
    _ORG_SETTINGS_STORE[organization_id] = updated
    return updated


def remove_organization_member(
    organization_id: str,
    admin_user_id: str,
    target_user_id: str,
    confirmation_token: str,
) -> None:
    """
    Removes a member from an organization after verifying authorization and explicit confirmation.

    Raises:
        UnauthorizedAdministrationException: If the admin_user_id lacks Owner/Admin role.
        InvalidConfirmationException: If the confirmation token does not match.
        SessionExpiredException: If there is no pending confirmation for the target user.
    """
    # Verify organization exists
    if organization_id not in _ORG_SETTINGS_STORE:
        raise OrganizationSettingsNotFoundException(
            f"Organization '{organization_id}' does not exist."
        )

    # Verify admin privileges
    if not _is_admin(admin_user_id, organization_id):
        raise UnauthorizedAdministrationException(
            f"User '{admin_user_id}' is not authorized to remove members from organization '{organization_id}'."
        )

    # Ensure target user is a member
    _ensure_member_exists(organization_id, target_user_id)

    # Prevent self-removal via admin endpoint (owner can use dedicated flow)
    if admin_user_id == target_user_id:
        raise UnauthorizedAdministrationException("Administrators cannot remove themselves via this endpoint.")

    # Verify confirmation token
    _verify_confirmation_token(organization_id, target_user_id, confirmation_token)

    # Perform removal
    del _ORG_MEMBERS_STORE[organization_id][target_user_id]


def transfer_ownership(
    organization_id: str,
    current_owner_id: str,
    new_owner_id: str,
    previous_owner_role: str,
) -> OwnershipTransferResult:
    """
    Transfers primary organization ownership to another active member.

    Raises:
        InvalidOwnerRecipientException: If the new owner is not a valid active member.
        ConcurrentOwnershipTransferException: If another transfer is in progress.
        UnauthorizedAdministrationException: If the current_owner_id is not the actual owner.
    """
    # Verify organization exists
    if organization_id not in _ORG_SETTINGS_STORE:
        raise OrganizationSettingsNotFoundException(
            f"Organization '{organization_id}' does not exist."
        )

    # Verify current owner
    if not _is_owner(current_owner_id, organization_id):
        raise UnauthorizedAdministrationException(
            f"User '{current_owner_id}' is not the current owner of organization '{organization_id}'."
        )

    # Verify new owner is an active member and not the same as current owner
    members = _ORG_MEMBERS_STORE.get(organization_id, {})
    if new_owner_id == current_owner_id:
        raise InvalidOwnerRecipientException("New owner must be different from the current owner.")
    if new_owner_id not in members:
        raise InvalidOwnerRecipientException(
            f"User '{new_owner_id}' is not a member of organization '{organization_id}'."
        )

    # Validate role to assign to previous owner
    if previous_owner_role not in {"Admin", "Member"}:
        raise ValueError("previous_owner_role must be either 'Admin' or 'Member'.")

    lock = _get_ownership_lock(organization_id)
    acquired = lock.acquire(blocking=False)
    if not acquired:
        raise ConcurrentOwnershipTransferException(
            f"An ownership transfer is already in progress for organization '{organization_id}'."
        )

    try:
        # Perform the role swaps atomically
        members[current_owner_id] = previous_owner_role
        members[new_owner_id] = "Owner"
        # Update the store
        _ORG_MEMBERS_STORE[organization_id] = members
        return OwnershipTransferResult(
            success=True,
            message=f"Ownership transferred from '{current_owner_id}' to '{new_owner_id}'.",
        )
    finally:
        lock.release()


# ==============================
# Mock Initialization (for testing purposes)
# ==============================
def _initialize_mock_data() -> None:
    """Populate the in-memory stores with sample data to illustrate usage."""
    org_id = "org-123"
    _ORG_SETTINGS_STORE[org_id] = OrganizationSettings(
        organization_id=org_id,
        name="Acme Corp",
        timezone="UTC",
    )
    _ORG_MEMBERS_STORE[org_id] = {
        "user-1": "Owner",
        "user-2": "Admin",
        "user-3": "Member",
    }
    # Example confirmation token for removing user-3
    _CONFIRMATION_TOKENS[(org_id, "user-3")] = "confirm-abc123"


# Initialize mock data when the module is imported
_initialize_mock_data()