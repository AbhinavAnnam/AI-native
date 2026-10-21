"""Transport-independent contracts for reviewing and repairing a candidate."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ReviewRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=255)
    max_batches: int | None = Field(default=None, ge=1, le=32)


class RepairRequest(ReviewRequest):
    input_fingerprint: str = Field(min_length=64, max_length=64)


class ApprovalRequest(RepairRequest):
    target_module: str


class Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str
    status: Literal["met", "partial", "missing", "unclear"]
    evidence: str = Field(min_length=1, max_length=700)
    file_path: str | None = None
    line: int | None = Field(default=None, ge=1)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_ids: list[str] = Field(min_length=1, max_length=40)
    severity: Literal["high", "medium", "low"]
    file_path: str
    line: int | None = Field(default=None, ge=1)
    explanation: str = Field(min_length=1, max_length=1000)
    suggested_fix: str = Field(min_length=1, max_length=1000)
    category: Literal["defect", "uncertain", "suggestion"] = "defect"
    requirement_quote: str = Field(default="", max_length=400)
    code_quote: str = Field(default="", max_length=500)
    counterexample: str = Field(default="", max_length=600)


class BatchReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=600)
    coverage: list[Assessment] = Field(max_length=64)
    findings: list[Finding] = Field(max_length=25)
    limitations: list[str] = Field(max_length=15)


class Coverage(Assessment):
    description: str
    target_module: str | None


class ReportFinding(Finding):
    id: str


class CriticReport(BaseModel):
    candidate_version: str
    input_fingerprint: str
    verdict: Literal["pass", "changes_requested", "needs_review"]
    summary: str
    coverage: list[Coverage]
    findings: list[ReportFinding]
    limitations: list[str]
    batches: int
