"""
notifications_and_messaging module

Handles in-app notifications and transactional email delivery.
Enforces business rules:
- Notifies users on relevant events based on explicit notification preferences.
- Distinguishes unread from read items; marking as read does not delete.
- Delivers transactional emails asynchronously.
- Email delivery failures do not roll back domain mutations.
- Idempotency prevents duplicate notifications during retries.
"""

import uuid
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NotificationDeliveryException(Exception):
    """Raised when a notification cannot be delivered to the user."""
    pass

class NotificationNotFoundException(Exception):
    """Raised when a notification with the given ID does not exist."""
    pass

class EmailProviderUnavailableException(Exception):
    """Raised when the email provider is unavailable or fails to send."""
    pass

class EmailTemplateNotFoundException(Exception):
    """Raised when the requested email template is not found."""
    pass

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

class NotificationStatus(Enum):
    UNREAD = "unread"
    READ = "read"

@dataclass
class Notification:
    id: str
    user_id: str
    event_type: str
    payload: Dict[str, Any]
    status: NotificationStatus
    created_at: datetime
    read_at: Optional[datetime] = None
    idempotency_key: Optional[str] = None

@dataclass
class EmailDeliveryResult:
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    delivered_at: Optional[datetime] = None

# ---------------------------------------------------------------------------
# Internal Stores (in-memory for this module)
# ---------------------------------------------------------------------------

# Notification store: notification_id -> Notification
_notification_store: Dict[str, Notification] = {}

# Idempotency key -> notification_id mapping to prevent duplicates
_idempotency_index: Dict[str, str] = {}

# User notification preferences: user_id -> set of event_types they want notified about
# If a user has no entry, they receive all notifications (default).
_user_preferences: Dict[str, Set[str]] = {}

# Email templates: template_id -> dict with 'subject' and 'body' (body may contain {token} placeholders)
_email_templates: Dict[str, Dict[str, str]] = {
    "account_verification": {
        "subject": "Verify Your Account",
        "body": "Hello {name}, please verify your account using this code: {verification_code}"
    },
    "password_recovery": {
        "subject": "Password Recovery",
        "body": "Hello {name}, use this link to reset your password: {reset_link}"
    },
    "invitation": {
        "subject": "You've Been Invited",
        "body": "Hello {name}, you have been invited to join {workspace_name}. Accept here: {invite_link}"
    },
    "comment_update": {
        "subject": "New Comment on {item_title}",
        "body": "Hello {name}, {commenter_name} commented on {item_title}: {comment_text}"
    }
}

# Simulated email provider state
_email_provider_available: bool = True

# Logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _generate_idempotency_key(user_id: str, event_type: str, payload: Dict[str, Any]) -> str:
    """
    Generate a deterministic idempotency key from user_id, event_type, and payload.
    This ensures that retrying the same notification does not create duplicates.
    """
    canonical_payload = json.dumps(payload, sort_keys=True, default=str)
    raw = f"{user_id}:{event_type}:{canonical_payload}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _is_user_subscribed(user_id: str, event_type: str) -> bool:
    """
    Check if the user has explicitly opted in to receive notifications for this event type.
    If the user has no preference entry, they are considered subscribed to all events.
    """
    if user_id not in _user_preferences:
        return True
    return event_type in _user_preferences[user_id]

def _render_email_template(template_id: str, tokens: Dict[str, Any]) -> tuple:
    """
    Render an email template with the provided tokens.
    Returns (subject, body).
    Raises EmailTemplateNotFoundException if template_id is not found.
    """
    if template_id not in _email_templates:
        raise EmailTemplateNotFoundException(f"Email template '{template_id}' not found.")
    
    template = _email_templates[template_id]
    subject = template["subject"]
    body = template["body"]
    
    # Render tokens into subject and body
    try:
        rendered_subject = subject.format(**tokens)
        rendered_body = body.format(**tokens)
    except KeyError as e:
        raise EmailTemplateNotFoundException(
            f"Missing token '{e}' required by template '{template_id}'."
        )
    
    return rendered_subject, rendered_body

def _dispatch_email_async(recipient_email: str, subject: str, body: str) -> EmailDeliveryResult:
    """
    Simulate asynchronous email dispatch via the configured background provider.
    In a real system, this would enqueue a job to a message broker.
    Returns an EmailDeliveryResult.
    """
    global _email_provider_available
    
    if not _email_provider_available:
        raise EmailProviderUnavailableException("Email provider is currently unavailable.")
    
    # Simulate successful delivery
    message_id = str(uuid.uuid4())
    delivered_at = datetime.now(timezone.utc)
    
    logger.info(
        "Email dispatched asynchronously: to=%s, subject=%s, message_id=%s",
        recipient_email, subject, message_id
    )
    
    return EmailDeliveryResult(
        success=True,
        message_id=message_id,
        delivered_at=delivered_at
    )

# ---------------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------------

def create_notification(user_id: str, event_type: str, payload: Dict[str, Any]) -> Notification:
    """
    Generates an in-app notification record for a target user.
    
    Business Rules:
    - Only creates notification if user is subscribed to the event type.
    - Uses idempotency key to prevent duplicate notifications during retries.
    - If a duplicate is detected, returns the existing notification.
    
    Args:
        user_id: The ID of the user to notify.
        event_type: The type of event (e.g., 'work_item_assigned', 'mention', 'comment_update').
        payload: A dictionary containing event-specific data.
        
    Returns:
        Notification: The created or existing notification record.
        
    Raises:
        NotificationDeliveryException: If the notification cannot be created.
    """
    try:
        # Check if user is subscribed to this event type
        if not _is_user_subscribed(user_id, event_type):
            logger.debug(
                "User %s is not subscribed to event type %s; skipping notification.",
                user_id, event_type
            )
            # Return a notification with a special status indicating it was suppressed
            # But per business rules, we should still return a Notification object.
            # We'll create it but mark it as read immediately to indicate suppression.
            # Actually, the rule says "based on explicit notification preferences" - 
            # if not subscribed, we should not create the notification.
            # However, the function signature requires returning a Notification.
            # We'll raise an exception to indicate the notification was not delivered.
            raise NotificationDeliveryException(
                f"User {user_id} is not subscribed to event type '{event_type}'."
            )
        
        # Generate idempotency key
        idempotency_key = _generate_idempotency_key(user_id, event_type, payload)
        
        # Check if this notification already exists (idempotency)
        if idempotency_key in _idempotency_index:
            existing_notification_id = _idempotency_index[idempotency_key]
            if existing_notification_id in _notification_store:
                logger.debug(
                    "Duplicate notification detected for idempotency key %s; returning existing notification %s.",
                    idempotency_key, existing_notification_id
                )
                return _notification_store[existing_notification_id]
        
        # Create new notification
        notification_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        notification = Notification(
            id=notification_id,
            user_id=user_id,
            event_type=event_type,
            payload=payload,
            status=NotificationStatus.UNREAD,
            created_at=now,
            read_at=None,
            idempotency_key=idempotency_key
        )
        
        # Store the notification
        _notification_store[notification_id] = notification
        _idempotency_index[idempotency_key] = notification_id
        
        logger.info(
            "Notification created: id=%s, user_id=%s, event_type=%s",
            notification_id, user_id, event_type
        )
        
        return notification
        
    except NotificationDeliveryException:
        raise
    except Exception as e:
        logger.error("Failed to create notification for user %s: %s", user_id, str(e))
        raise NotificationDeliveryException(f"Failed to create notification: {str(e)}")

def mark_notification_as_read(notification_id: str, user_id: str) -> None:
    """
    Updates notification status to read for a specified user.
    
    Business Rules:
    - Marking as read does not delete the notification.
    - Only the owner of the notification can mark it as read.
    
    Args:
        notification_id: The ID of the notification to mark as read.
        user_id: The ID of the user who owns the notification.
        
    Raises:
        NotificationNotFoundException: If the notification does not exist.
        NotificationDeliveryException: If the user does not own the notification.
    """
    if notification_id not in _notification_store:
        raise NotificationNotFoundException(
            f"Notification with ID '{notification_id}' not found."
        )
    
    notification = _notification_store[notification_id]
    
    # Verify ownership
    if notification.user_id != user_id:
        raise NotificationDeliveryException(
            f"User '{user_id}' does not own notification '{notification_id}'."
        )
    
    # Update status to read
    notification.status = NotificationStatus.READ
    notification.read_at = datetime.now(timezone.utc)
    
    logger.info(
        "Notification marked as read: id=%s, user_id=%s",
        notification_id, user_id
    )

def send_transactional_email(recipient_email: str, template_id: str, tokens: Dict[str, Any]) -> EmailDeliveryResult:
    """
    Dispatches transactional email via configured background provider.
    
    Business Rules:
    - Email delivery is asynchronous.
    - Email delivery failures or delays must not cause underlying domain mutations 
      or invitations to roll back. This is achieved by catching exceptions and 
      returning a failure result rather than propagating the exception for 
      non-critical failures. However, for template not found or provider 
    unavailable, we raise specific exceptions as they indicate configuration 
    issues that should be surfaced.
    
    Args:
        recipient_email: The email address of the recipient.
        template_id: The ID of the email template to use.
        tokens: A dictionary of tokens to substitute in the template.
        
    Returns:
        EmailDeliveryResult: The result of the email delivery attempt.
        
    Raises:
        EmailTemplateNotFoundException: If the template is not found or tokens are missing.
        EmailProviderUnavailableException: If the email provider is unavailable.
    """
    try:
        # Render the template
        subject, body = _render_email_template(template_id, tokens)
        
        # Dispatch email asynchronously
        result = _dispatch_email_async(recipient_email, subject, body)
        
        logger.info(
            "Transactional email sent: to=%s, template=%s, success=%s",
            recipient_email, template_id, result.success
        )
        
        return result
        
    except EmailTemplateNotFoundException:
        raise
    except EmailProviderUnavailableException:
        raise
    except Exception as e:
        logger.error(
            "Unexpected error sending transactional email to %s with template %s: %s",
            recipient_email, template_id, str(e)
        )
        # Per business rule, email delivery failures should not cause rollbacks.
        # We return a failure result instead of raising, unless it's a known 
        # configuration error (template not found, provider unavailable).
        return EmailDeliveryResult(
            success=False,
            error=str(e)
        )