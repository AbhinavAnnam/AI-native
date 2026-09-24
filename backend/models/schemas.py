from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ==========================================
# Phase 1: Specification Extraction Schemas
# ==========================================

class FunctionSpec(BaseModel):
    signature: str = Field(
        description="Full function signature including parameters, type annotations, and return type."
    )
    purpose: str = Field(description="Detailed explanation of what the function performs.")


class FeatureSpec(BaseModel):
    feature_name: str = Field(description="Human-readable feature module name.")
    target_module: str = Field(
        description="Dynamically generated snake_case module identifier (e.g., iam_identity, organization_management, work_management)."
    )
    business_rules: List[str] = Field(
        description="List of domain policies, access controls, edge cases, and business logic."
    )
    functions: List[FunctionSpec] = Field(
        description="List of executable function signatures with parameter specifications."
    )
    exceptions: List[str] = Field(
        description="System, validation, and domain exception class names."
    )


class ConstraintCategory(BaseModel):
    category_name: str = Field(
        description="Snake_case constraint category identifier."
    )
    rules: List[str] = Field(
        description="Explicit list of technical rules, guardrails, and constraints."
    )


class ProjectSpecs(BaseModel):
    project_title: str = Field(
        description="Title or system name extracted dynamically from the specification document."
    )
    features: List[FeatureSpec] = Field(
        description="List of functional architectural modules."
    )
    cross_cutting_constraints: List[ConstraintCategory] = Field(
        description="Categorized mapping of cross-cutting rules and guardrails."
    )


# ==========================================
# Phase 2: HITL & Code Commit Schemas
# ==========================================

class HITLApprovalPayload(BaseModel):
    action: Optional[Literal["approve", "reject"]] = Field(
        default="approve",
        description="Approval action choice ('approve' or 'reject')."
    )
    draft_id: Optional[str] = Field(
        default=None,
        description="Unique identifier of the code draft under review."
    )
    target_module: Optional[str] = Field(
        default=None,
        description="Target module identifier (e.g., iam_identity)."
    )
    approved_code: Optional[str] = Field(
        default=None,
        description="Final approved code to commit to disk."
    )
    approved: Optional[bool] = Field(
        default=True,
        description="Boolean approval status."
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Optional feedback or revision notes."
    )
    modified_spec: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional modified spec object passed back from the UI."
    )


# ==========================================
# Phase 2: LangGraph Agent Engine Schemas
# ==========================================

class TaskClassification(BaseModel):
    task_type: Literal["simple_query", "architecture_planning", "code_generation", "code_refinement"] = Field(
        ..., description="Classified type of incoming task."
    )
    recommended_model: Literal["fast", "heavy"] = Field(
        ..., description="'fast' for simple queries; 'heavy' for complex architectural/code tasks."
    )
    reasoning: str = Field(..., description="Brief explanation for routing choice.")


class StartExecutionRequest(BaseModel):
    thread_id: str = Field(..., description="Unique thread identifier for state persistence.")
    task_spec: str = Field(..., description="Task or module specification prompt to execute.")


class HumanFeedbackRequest(BaseModel):
    thread_id: str = Field(..., description="Unique thread identifier for state persistence.")
    user_feedback: str = Field(..., description="Human review input or revision feedback.")
    modified_spec: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional modified spec object passed back from the UI.",
    )