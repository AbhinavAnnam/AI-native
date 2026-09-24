from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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


class HITLApprovalPayload(BaseModel):
    approved: bool = Field(
        description="Whether the generated specification or execution plan is approved by the human."
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Optional feedback or revision notes if rejected or modified.",
    )
    modified_spec: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional modified spec object passed back from the user interface.",
    )