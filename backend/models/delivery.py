"""Validated contracts shared by the delivery services and HTTP adapter."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DeliveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thread_id: str = Field(min_length=1, max_length=255)
    input_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class PublishRequest(DeliveryRequest):
    document_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    title: str = Field(default="Implement reviewed requirements", min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def valid_title(cls, value):
        if not value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("Provide a nonempty, single-line PR title.")
        return value.strip()


class FileChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    status: Literal["added", "modified", "unchanged"]
    added_lines: int = Field(ge=0)
    removed_lines: int = Field(ge=0)
    symbols: list[str]


class DocumentationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    candidate_version: str
    input_fingerprint: str
    filename: str
    markdown: str
    files: list[FileChange]
    requirements_assessed: int
    findings: int
    testing: Literal["not_run"] = "not_run"


class PullRequestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    number: int = Field(gt=0)
    url: str
    head: str
    base: str
    commit_sha: str
    draft: Literal[True] = True


class DeliveryView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documentation: DocumentationResult | None = None
    documentation_stale: bool = False
    pull_request: PullRequestResult | None = None
