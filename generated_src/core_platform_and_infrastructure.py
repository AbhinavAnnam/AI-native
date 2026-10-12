import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, Optional, Union

# ==============================
# Exception Definitions
# ==============================
class ValidationError(Exception):
    """Raised when input validation fails."""
    pass

class AuthorizationError(Exception):
    """Raised when authentication or permission checks fail."""
    pass

class InternalServerException(Exception):
    """Raised for unexpected internal errors."""
    pass

class IdempotencyConflictException(Exception):
    """Raised when an idempotent operation conflicts with existing state."""
    pass

class InvalidTimezoneException(Exception):
    """Raised when a provided timezone string is invalid."""
    pass

class ConcurrencyConflictException(Exception):
    """Raised when optimistic locking detects a conflict."""
    pass

class DependencyUnavailableException(Exception):
    """Raised when an external dependency cannot be reached."""
    pass

class JobEnqueueException(Exception):
    """Raised when a background job cannot be enqueued."""
    pass

class MaxRetriesExceededException(Exception):
    """Raised when a job exceeds its maximum retry attempts."""
    pass

# ==============================
# Data Structures
# ==============================
@dataclass(frozen=True)
class RequestContext:
    user_id: str
    organization_id: str
    auth_token: Optional[str] = None
    permissions: Optional[Dict[str, bool]] = field(default_factory=dict)

@dataclass
class ApiResponse:
    status_code: int
    body: Union[Dict[str, Any], str]

@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_seconds: int = 5

@dataclass
class JobReference:
    job_id: str
    status: str  # e.g., "queued", "failed", "completed"

# ==============================
# Internal In-Memory Stores
# ==============================
_idempotency_store: Dict[str, Any] = {}
_idempotency_lock = threading.Lock()

_job_queue: Dict[str, Dict[str, Any]] = {}
_job_queue_lock = threading.Lock()

# ==============================
# Helper Functions
# ==============================
def _validate_identifier(identifier: str) -> None:
    """Ensures the identifier contains only safe characters."""
    if not identifier or not identifier.replace("-", "").isalnum():
        raise ValidationError(f"Invalid identifier: {identifier}")

def _ensure_authenticated(context: RequestContext) -> None:
    """Ensures the request context carries a valid auth token."""
    if not context.auth_token:
        raise AuthorizationError("Missing authentication token.")

def _mask_error_message(exc: Exception) -> str:
    """Masks internal error details for external responses."""
    if isinstance(exc, (ValidationError, AuthorizationError)):
        return str(exc)
    return "An unexpected error occurred."

def _relative_delta(delta: timedelta) -> str:
    """Converts a timedelta into a human-readable relative string."""
    seconds = int(delta.total_seconds())
    if seconds < 0:
        seconds = -seconds
        suffix = "from now"
    else:
        suffix = "ago"

    if seconds < 60:
        return f"{seconds} seconds {suffix}"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minutes {suffix}"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hours {suffix}"
    days = hours // 24
    if days < 30:
        return f"{days} days {suffix}"
    months = days // 30
    if months < 12:
        return f"{months} months {suffix}"
    years = months // 12
    return f"{years} years {suffix}"

# ==============================
# Core Functions
# ==============================
def handle_api_request(request_context: RequestContext, payload: Any) -> ApiResponse:
    """
    Routes and processes API requests with unified error handling and authentication context.
    The payload is expected to be a dict containing at least an 'action' key.
    """
    try:
        _ensure_authenticated(request_context)

        if not isinstance(payload, dict) or "action" not in payload:
            raise ValidationError("Payload must be a dict with an 'action' field.")

        action = payload["action"]
        data = payload.get("data", {})

        # Example routing logic (expandable as needed)
        if action == "get_resource":
            resource_id = data.get("resource_id")
            _validate_identifier(resource_id)
            # Simulated fetch (replace with real data access)
            resource = {"id": resource_id, "owner_org": request_context.organization_id}
            return ApiResponse(status_code=200, body=resource)

        elif action == "list_resources":
            # Pagination parameters
            page = int(data.get("page", 1))
            page_size = int(data.get("page_size", 20))
            if page < 1 or page_size < 1:
                raise ValidationError("Page and page_size must be positive integers.")
            # Simulated collection
            total = 125  # pretend total count
            start = (page - 1) * page_size
            end = start + page_size
            items = [{"id": f"res-{i}", "org": request_context.organization_id}
                     for i in range(start, min(end, total))]
            pagination = {
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": (total + page_size - 1) // page_size,
            }
            return ApiResponse(status_code=200, body={"items": items, "pagination": pagination})

        elif action == "create_resource":
            # Idempotent creation example
            idempotency_key = payload.get("idempotency_key")
            if not idempotency_key:
                raise ValidationError("Missing idempotency_key for create operation.")
            def creation_logic():
                new_id = str(uuid.uuid4())
                # Simulated persistence
                return {"id": new_id, "created_by": request_context.user_id}
            result = execute_idempotent_operation(idempotency_key, creation_logic)
            return ApiResponse(status_code=201, body=result)

        else:
            raise ValidationError(f"Unsupported action: {action}")

    except (ValidationError, AuthorizationError) as known_exc:
        return ApiResponse(status_code=400, body={"error": _mask_error_message(known_exc)})
    except IdempotencyConflictException as ic_exc:
        return ApiResponse(status_code=409, body={"error": _mask_error_message(ic_exc)})
    except ConcurrencyConflictException as cc_exc:
        return ApiResponse(status_code=409, body={"error": _mask_error_message(cc_exc)})
    except DependencyUnavailableException as du_exc:
        return ApiResponse(status_code=503, body={"error": _mask_error_message(du_exc)})
    except Exception as unexpected:
        # Log the unexpected exception in real implementation
        return ApiResponse(status_code=500, body={"error": _mask_error_message(unexpected)})

def format_relative_date(utc_timestamp: datetime, user_timezone: str) -> str:
    """
    Formats a UTC timestamp into a localized relative date string based on the user's timezone.
    """
    if utc_timestamp.tzinfo is None:
        utc_timestamp = utc_timestamp.replace(tzinfo=timezone.utc)
    else:
        utc_timestamp = utc_timestamp.astimezone(timezone.utc)

    try:
        tz = timezone.utc if user_timezone.lower() == "utc" else datetime.now().astimezone().tzinfo
        # Use zoneinfo if available for proper tz handling
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(user_timezone)
        except Exception:
            # Fallback to UTC if zone cannot be resolved
            raise InvalidTimezoneException(f"Invalid timezone: {user_timezone}")
    except InvalidTimezoneException:
        raise
    except Exception:
        raise InvalidTimezoneException(f"Invalid timezone: {user_timezone}")

    localized_dt = utc_timestamp.astimezone(tz)
    now_local = datetime.now(tz)

    delta = now_local - localized_dt
    return _relative_delta(delta)

def execute_idempotent_operation(idempotency_key: str, operation_closure: Callable[[], Any]) -> Any:
    """
    Executes a closure idempotently using a unique key to prevent duplicate processing.
    """
    if not idempotency_key:
        raise ValidationError("Idempotency key must be a non-empty string.")

    with _idempotency_lock:
        if idempotency_key in _idempotency_store:
            stored = _idempotency_store[idempotency_key]
            if isinstance(stored, Exception):
                raise stored
            return stored

        try:
            result = operation_closure()
            _idempotency_store[idempotency_key] = result
            return result
        except Exception as exc:
            # Store the exception to ensure repeat calls raise the same error
            _idempotency_store[idempotency_key] = exc
            raise IdempotencyConflictException(str(exc)) from exc

def enqueue_background_job(
    job_type: str,
    organization_id: str,
    payload: Any,
    retry_policy: RetryPolicy
) -> JobReference:
    """
    Enqueues an asynchronous task for background worker processing.
    Implements dead-letter queue policies for retries.
    """
    if not job_type:
        raise ValidationError("Job type must be provided.")
    _validate_identifier(organization_id)

    job_id = str(uuid.uuid4())
    job_record = {
        "job_id": job_id,
        "type": job_type,
        "organization_id": organization_id,
        "payload": payload,
        "retry_policy": retry_policy,
        "attempts": 0,
        "status": "queued",
    }

    try:
        with _job_queue_lock:
            _job_queue[job_id] = job_record
    except Exception as exc:
        raise JobEnqueueException(f"Failed to enqueue job: {exc}") from exc

    # In a real system, a worker would pick up the job. Here we simulate immediate processing.
    try:
        _process_job(job_record)
    except MaxRetriesExceededException as mre:
        job_record["status"] = "dead_letter"
        raise
    except Exception as exc:
        job_record["status"] = "failed"
        raise JobEnqueueException(f"Job processing failed: {exc}") from exc

    return JobReference(job_id=job_id, status=job_record["status"])

# ==============================
# Simulated Worker Logic
# ==============================
def _process_job(job_record: Dict[str, Any]) -> None:
    """
    Simulated job processing that respects retry policy.
    Raises MaxRetriesExceededException when retries are exhausted.
    """
    while job_record["attempts"] < job_record["retry_policy"].max_retries:
        try:
            job_record["attempts"] += 1
            # Placeholder for actual job execution logic.
            # For demonstration, we assume success.
            job_record["status"] = "completed"
            return
        except DependencyUnavailableException:
            # Simulate retryable failure
            if job_record["attempts"] >= job_record["retry_policy"].max_retries:
                raise MaxRetriesExceededException(
                    f"Job {job_record['job_id']} exceeded max retries."
                )
            # Backoff simulation (omitted actual sleep for brevity)
            continue
    # If loop exits without return, max retries were hit
    raise MaxRetriesExceededException(
        f"Job {job_record['job_id']} exceeded max retries."
    )