"""Lifecycle of the generated output shared by both coder entry points."""
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def clear_generated_workspace() -> str:
    """Start a new run with an empty generated_src, or fail before generation."""
    workspace = Path.cwd().resolve()
    target = workspace / "generated_src"
    resolved = target.resolve()
    # The writer also uses the current working directory. Do not follow a
    # symlink/junction to another directory when deleting previous output.
    if resolved != target or resolved.parent != workspace:
        raise ValueError("generated_src must be a regular directory inside the workspace.")
    if target.exists():
        shutil.rmtree(resolved)
    target.mkdir()
    logger.info("Cleared generated workspace for a new run: %s", target)
    return str(target)
