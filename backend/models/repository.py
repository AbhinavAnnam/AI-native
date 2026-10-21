"""Transport-independent contracts for repository-based coding runs."""
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

MAX_REPOSITORY_MODULES = 50


def validate_repository_url(value: str) -> str:
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Use a GitHub HTTPS repository URL without embedded credentials, query, or fragment.")
    return value


def validate_python_path(value: str) -> str:
    parts = value.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if (
        not value.endswith(".py")
        or any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", p) for p in parts)
        or any(p.endswith(".") or p.split(".")[0].upper() in reserved for p in parts)
    ):
        raise ValueError("Use a repository-relative Python path without traversal, hidden directories, or reserved names.")
    return value


class RepositoryRequest(BaseModel):
    url: str
    base_branch: Optional[str] = Field(default=None, description="Omit to use the repository's default branch.")
    file_mappings: dict[str, str] = Field(
        default_factory=dict, max_length=MAX_REPOSITORY_MODULES,
        description="Optional file overrides. Empty means select files from the repository automatically.",
    )

    _validate_url = field_validator("url")(validate_repository_url)

    @field_validator("base_branch")
    @classmethod
    def validate_branch(cls, value):
        if value is not None and (not value or value.startswith("-") or any(c.isspace() for c in value)):
            raise ValueError("Provide a branch name or omit it to use the default branch.")
        return value

    @field_validator("file_mappings")
    @classmethod
    def validate_mappings(cls, value):
        for module, path in value.items():
            if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
                raise ValueError("Mapping keys must match specification target_module identifiers.")
            validate_python_path(path)
        if len({path.casefold() for path in value.values()}) != len(value):
            raise ValueError("Each module must map to a different file.")
        return value


class RepositoryWorkspace(BaseModel):
    run_id: str
    url: str
    repo_dir: str
    base_branch: str
    baseline_commit: str
    work_branch: str
    file_mappings: dict[str, str]
    original_codes: dict[str, str]
    original_hashes: dict[str, Optional[str]]
    context: str
