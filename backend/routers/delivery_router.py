"""Thin HTTP adapter; reusable documentation and publishing live in stages/."""
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from backend.models.delivery import DeliveryRequest, DeliveryView, PublishRequest
from backend.routers import phase2_router as workflow
from backend.routers.critic_router import _persist, _state
from backend.stages.critic import CriticError
from backend.stages.documentation import DeliveryError, delivery_view, generate_documentation
from backend.stages.pull_request import publish_pull_request
from backend.tools.git_provider import RepositoryError

def local_origin(request: Request):
    # Streamlit calls from Python (no Origin); browser clients must be local.
    # This protects token-backed publishing even with the legacy app's broad CORS.
    origin = request.headers.get("origin")
    if origin:
        try:
            valid = urlsplit(origin).hostname in {"127.0.0.1", "localhost", "::1"}
        except ValueError:
            valid = False
        if not valid:
            raise HTTPException(403, "Delivery actions must come from the local application.")


router = APIRouter(prefix="/api/delivery", tags=["Documentation & Pull Request"], dependencies=[Depends(local_origin)])


def _generate(request):
    config, state = _state(request.thread_id)
    document = generate_documentation(state.values, request.thread_id, request.input_fingerprint)
    _persist(config, state, {"documentation": document.model_dump(), "delivery_thread_id": request.thread_id})
    return delivery_view(workflow.agent_app.get_state(config).values)


def _publish(request):
    config, state = _state(request.thread_id)
    def save_progress(value):
        _persist(config, state, {"publication_progress": value})
    result = publish_pull_request(state.values, request.thread_id, request.document_id,
                                  request.input_fingerprint, request.title, on_progress=save_progress)
    _persist(config, state, {"pull_request": result.model_dump()})
    return delivery_view(workflow.agent_app.get_state(config).values)


async def _action(request, operation):
    async with workflow.workflow_locks[request.thread_id]:
        try:
            return await run_in_threadpool(operation, request)
        except HTTPException:
            raise
        except (DeliveryError, CriticError, RepositoryError) as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(409, "Delivery inputs could not be validated or saved. Refresh the candidate and check workspace permissions.") from exc


@router.post("/documentation", response_model=DeliveryView)
async def documentation(request: DeliveryRequest):
    return await _action(request, _generate)


@router.post("/pull-request", response_model=DeliveryView)
async def pull_request(request: PublishRequest):
    return await _action(request, _publish)
