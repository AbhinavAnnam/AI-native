"""Local HTTP adapter. It never loads API keys or starts another workflow engine."""
import math
import re
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit

import httpx


class BackendError(RuntimeError):
    def __init__(self, message, *, status_code=None, retry_after=None, execution_state_unknown=False):
        super().__init__(message)
        self.details = {"message": message, "status_code": status_code, "retry_after": retry_after,
                        "execution_state_unknown": execution_state_unknown}


def validate_thread_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", value):
        raise BackendError("Use a run ID containing only letters, numbers, underscores, dots, or hyphens.")
    return value


class BackendClient:
    def __init__(self, base_url="http://127.0.0.1:8000", documents_root=None, *, timeout=900, client=None):
        try:
            url = urlsplit(base_url)
            valid = (url.scheme == "http" and url.hostname in {"127.0.0.1", "localhost", "::1"}
                     and not url.username and not url.password and not url.query and not url.fragment
                     and url.path in {"", "/"})
            _ = url.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("MCP must connect to a local HTTP backend URL without credentials or a path.")
        if not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("MCP timeout must be between 0 and 3600 seconds.")
        self.base_url = base_url.rstrip("/")
        self.documents_root = Path(documents_root or Path(__file__).resolve().parents[1] / "docs").resolve()
        self.timeout, self.client = timeout, client

    async def request(self, method, path, *, payload=None, files=None):
        if payload and "thread_id" in payload:
            validate_thread_id(payload["thread_id"])
        kwargs = {"json": payload} if payload is not None else {}
        if files is not None:
            kwargs["files"] = files
        try:
            if self.client is not None:
                response = await self.client.request(method, self.base_url + path, **kwargs)
            else:
                # No proxy/netrc inheritance, redirects, or automatic mutation retries.
                async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5),
                                             trust_env=False, follow_redirects=False) as client:
                    response = await client.request(method, self.base_url + path, **kwargs)
        except httpx.ConnectError as exc:
            raise BackendError("Start the existing FastAPI backend and check its local port.", status_code=503) from exc
        except httpx.RequestError as exc:
            raise BackendError("Backend connection was interrupted. Read get_run_state before retrying; the operation may still be running.",
                               status_code=504, execution_state_unknown=method != "GET") from exc
        if not 200 <= response.status_code < 300:
            messages = {404: "Run or API endpoint not found. Check the run ID and backend version.",
                        409: "Backend rejected this action. Read get_run_state and check review freshness, pending repairs, approval, or publishing prerequisites.",
                        422: "Backend rejected the input or model response. Inspect the saved run and backend UI for details.",
                        429: "Provider quota is temporarily exhausted. Keep the run and respect retry_after.",
                        402: "Backend reported a provider billing restriction. Check provider configuration.",
                        503: "A backend/provider is temporarily unavailable. Completed work remains saved."}
            retry_after = None
            try:
                delay = float(response.headers.get("retry-after", ""))
                if math.isfinite(delay):
                    retry_after = max(0, min(86400, math.ceil(delay)))
            except ValueError:
                pass
            # Never echo arbitrary provider bodies: they can contain credentials/source.
            raise BackendError(messages.get(response.status_code, "Backend request failed. Inspect the saved run and backend UI before retrying."),
                               status_code=response.status_code, retry_after=retry_after,
                               execution_state_unknown=method != "GET" and response.status_code >= 500)
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")
            return data
        except ValueError as exc:
            raise BackendError("Backend returned an invalid JSON object. Inspect saved state before retrying.",
                               status_code=502, execution_state_unknown=method != "GET") from exc

    async def extract(self, pdf_paths):
        if not 1 <= len(pdf_paths) <= 10:
            raise BackendError("Provide between one and ten PDF paths inside the configured documents root.")
        try:
            with ExitStack() as stack:
                files, total = [], 0
                for name in pdf_paths:
                    path = (self.documents_root / name).resolve(strict=True)
                    if not path.is_relative_to(self.documents_root) or path.suffix.lower() != ".pdf" or not path.is_file():
                        raise BackendError("PDF inputs must be regular .pdf files inside the configured documents root.")
                    handle = stack.enter_context(path.open("rb"))
                    size = path.stat().st_size
                    total += size
                    if size > 20 * 1024 * 1024 or total > 50 * 1024 * 1024:
                        raise BackendError("MCP uploads allow 20 MB per PDF and 50 MB total.")
                    if not handle.read(5).startswith(b"%PDF-"):
                        raise BackendError("An input does not have a PDF header.")
                    handle.seek(0)
                    files.append(("files", (path.name, handle, "application/pdf")))
                return await self.request("POST", "/api/phase1/extract", files=files)
        except (OSError, ValueError) as exc:
            raise BackendError("A PDF could not be read. Check its path and the configured documents root.") from exc
