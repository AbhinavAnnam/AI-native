"""Prepare a repository snapshot and persist only individually approved Python files."""
import ast
import hashlib
import re
from pathlib import Path
from uuid import uuid4

from backend.models.repository import MAX_REPOSITORY_MODULES, RepositoryRequest, RepositoryWorkspace, validate_python_path
from backend.tools import git_provider


def repository_file(repo_dir: str, relative_path: str) -> Path:
    root = Path(repo_dir).resolve(strict=True)
    path = root.joinpath(*relative_path.split("/"))
    if not path.resolve().is_relative_to(root):
        raise ValueError("File must stay inside the repository workspace.")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError("Repository file paths must not contain symbolic links or junctions.")
    return path


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def suggest_file_mappings(specification: dict, tracked: list[str]) -> dict[str, str]:
    """Match existing source names; place unmatched modules beside the source package."""
    features = specification.get("features", [])
    if not isinstance(features, list) or not 1 <= len(features) <= MAX_REPOSITORY_MODULES:
        raise ValueError(f"Repository coding supports between 1 and {MAX_REPOSITORY_MODULES} planned code modules. Group related features without dropping requirements.")
    modules = [feature.get("target_module") for feature in features if isinstance(feature, dict)]
    if len(modules) != len(features) or any(not isinstance(module, str) for module in modules) or len(set(modules)) != len(modules):
        raise ValueError("Each specification feature needs a unique target_module.")

    candidates = []
    for relative in tracked:
        try:
            validate_python_path(relative)
        except ValueError:
            continue
        path = Path(relative)
        if path.name in {"__init__.py", "conftest.py", "setup.py"} or path.name.startswith("test_"):
            continue
        if any(part.lower() in {"test", "tests", "testing", "venv", "node_modules"} for part in path.parts[:-1]):
            continue
        candidates.append(relative)

    def tokens(name):
        ignored = {"and", "management", "manager", "service", "module", "py"}
        return {
            word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word
            for word in re.findall(r"[a-z0-9]+", name.lower()) if word not in ignored
        }

    roots = set()
    for relative in candidates:
        parts = Path(relative).parts
        roots.add("/".join(parts[:2]) if parts[0] == "src" and len(parts) > 2 else (parts[0] if len(parts) > 1 else ""))
    mappings = {}
    for module in modules:
        name = module.removesuffix(".py")
        scored = []
        for relative in candidates:
            dotted = relative.removesuffix(".py").replace("/", ".")
            exact = dotted == name or dotted.endswith("." + name) or Path(relative).stem == name.rsplit(".", 1)[-1]
            score = 100 if exact else len(tokens(name) & tokens(Path(relative).stem))
            if score:
                scored.append((score, relative))
        if scored:
            best_score = max(score for score, _ in scored)
            matches = [relative for score, relative in scored if score == best_score]
            if len(matches) != 1:
                raise ValueError(f"File selection is ambiguous for {module}: {', '.join(matches)}. Choose target files in Advanced settings.")
            destination = matches[0]
        else:
            if len(roots) > 1:
                raise ValueError(f"Multiple source packages found for new module {module}. Choose target files in Advanced settings.")
            source_root = next(iter(roots), "")
            package = source_root.rsplit("/", 1)[-1]
            module_path = name.removeprefix(package + ".") if package else name
            destination = "/".join(part for part in (source_root, module_path.replace(".", "/") + ".py") if part)
        if destination.casefold() in {path.casefold() for path in mappings.values()}:
            raise ValueError(f"Multiple features select the same file ({destination}). Combine those features or choose different target files in Advanced settings.")
        mappings[module] = validate_python_path(destination)
    return mappings


def prepare_repository(
    request: RepositoryRequest, workspace_root: Path | None = None, specification: dict | None = None,
) -> RepositoryWorkspace:
    """Clone once per run; inspect text without executing repository code or setup scripts."""
    root = (workspace_root or Path.cwd() / "workspace").resolve()
    run_id = uuid4().hex
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    repo = run_dir / "repo"
    git_provider.clone_repository(request.url, repo, request.base_branch)
    baseline = git_provider.run_git(["rev-parse", "HEAD"], repo)
    base_branch = git_provider.run_git(["branch", "--show-current"], repo)
    work_branch = f"ai-native/{run_id}"
    git_provider.run_git(["checkout", "-b", work_branch], repo)

    tracked = git_provider.run_git(["ls-files"], repo).splitlines()
    if len(tracked) > 2_000:
        raise ValueError("This first repository workflow supports at most 2,000 tracked files.")
    if not request.file_mappings:
        mappings = suggest_file_mappings(specification or {}, tracked)
        request = RepositoryRequest.model_validate({**request.model_dump(), "file_mappings": mappings})

    originals, hashes = {}, {}
    for module, relative in request.file_mappings.items():
        path = repository_file(str(repo), relative)
        if path.exists() and path.stat().st_size > 128_000:
            raise ValueError(f"Selected file is too large for this workflow: {relative}")
        originals[module] = path.read_text(encoding="utf-8") if path.exists() else ""
        hashes[module] = file_hash(path)

    blocks = [
        "Target module to file mappings:\n" + "\n".join(f"{module}: {path}" for module, path in request.file_mappings.items()),
        "Repository files:\n" + "\n".join(tracked),
    ]
    budget = 60_000
    for relative in tracked:
        if any(part.startswith(".") for part in relative.split("/")):
            continue
        if not (relative.endswith(".py") or relative in {"README.md", "pyproject.toml", "AGENTS.md"}):
            continue
        try:
            path = repository_file(str(repo), relative)
            if not path.is_file() or path.stat().st_size > min(budget, 16_000):
                continue
            contents = path.read_text(encoding="utf-8")
        except (ValueError, UnicodeError):
            continue
        budget -= len(contents.encode("utf-8"))
        blocks.append(f"--- {relative} ---\n{contents}")

    workspace = RepositoryWorkspace(
        run_id=run_id, url=request.url, repo_dir=str(repo), base_branch=base_branch,
        baseline_commit=baseline, work_branch=work_branch, file_mappings=request.file_mappings,
        original_codes=originals, original_hashes=hashes,
        context="\n\n".join(blocks),
    )
    (run_dir / "repository.json").write_text(workspace.model_dump_json(indent=2), encoding="utf-8")
    return workspace


def save_approved_file(workspace: RepositoryWorkspace, module: str, code: str) -> str:
    relative = validate_python_path(workspace.file_mappings[module])
    path = repository_file(workspace.repo_dir, relative)
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc.msg}") from exc
    if file_hash(path) != workspace.original_hashes[module]:
        raise ValueError(f"{relative} changed since it was loaded; refusing to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8", newline="")
    return file_hash(path)
