"""Bounded Git operations; no shell interpolation or interactive credential prompts."""
import base64
import os
import subprocess
from pathlib import Path

from backend.models.repository import validate_repository_url


class RepositoryError(RuntimeError):
    pass


def run_git(arguments: list[str], cwd: Path | None = None, *, github_token: str | None = None) -> str:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1", GCM_INTERACTIVE="Never")
    if github_token:
        # Process-local authentication: never put credentials in argv, URLs, or .git/config.
        auth = base64.b64encode(f"x-access-token:{github_token}".encode()).decode()
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                   GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {auth}")
    hooks = Path(__file__).resolve().parents[2] / "workspace" / ".disabled-hooks"
    try:
        result = subprocess.run(
            ["git", "-c", f"core.hooksPath={hooks}", "-c", "protocol.file.allow=never",
             "-c", "protocol.ext.allow=never", *arguments],
            cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RepositoryError("Git operation timed out after 120 seconds.") from exc
    except OSError as exc:
        raise RepositoryError("Could not start Git. Check that Git is installed.") from exc
    if result.returncode:
        # Do not return credential-helper output or arbitrary remote text to clients/LLMs.
        stderr = result.stderr.lower()
        reason = "Check the repository URL, branch, and access permissions."
        for markers, explanation in (
            (("dubious ownership",), "Git rejected the workspace ownership. Start the backend under the Windows account that owns this project."),
            (("could not resolve host", "failed to connect", "connection timed out", "ssl certificate problem"),
             "Git could not connect securely to GitHub. Check the network, proxy, and certificate settings."),
            (("authentication failed", "could not read username", "repository not found", "terminal prompts disabled"),
             "Git could not access the repository. Confirm that it is public or that Git credentials are configured for the backend account."),
            (("remote branch", "not a valid branch name"), "Git could not use the base branch. Check its name or leave it blank for the default branch."),
            (("cannot lock ref", "index.lock", "permission denied", "access is denied"),
             "Git could not lock or write workspace files. Check folder permissions and whether another Git operation is using this clone."),
        ):
            if any(marker in stderr for marker in markers):
                reason = explanation
                break
        operation = arguments[0] if arguments and arguments[0] in {
            "clone", "rev-parse", "branch", "checkout", "ls-files", "check-ref-format",
        } else "operation"
        raise RepositoryError(f"Git {operation} failed (exit code {result.returncode}). {reason}")
    return result.stdout.strip()


def clone_repository(url: str, destination: Path, base_branch: str | None = None) -> None:
    validate_repository_url(url)
    arguments = ["clone", "--single-branch", "--no-recurse-submodules"]
    if base_branch is not None:
        run_git(["check-ref-format", "--branch", base_branch])
        arguments.extend(["--branch", base_branch])
    run_git([*arguments, "--", url, str(destination)])
