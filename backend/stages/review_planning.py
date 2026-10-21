"""Plan complete-source review scopes without repeating global rules per module."""
import ast
import json
import re


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def dependencies(files):
    """Conservative transitive import graph among the supplied Python modules."""
    graph = {path: set() for path in files}
    symbols = {}
    for path, code in files.items():
        try:
            symbols[path] = {n.name for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        except SyntaxError:
            symbols[path] = set()
    for path, code in files.items():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
                local_call = (isinstance(node.func, ast.Name) or (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and node.func.value.id in {"self", "cls"}))
                if called and not (local_call and called in symbols[path]):
                    graph[path].update(p for p in files if p != path and called in symbols[p])
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            else:
                continue
            for other in files:
                if other != path and any(name and (other[:-3].replace("/", ".").endswith(name) or
                        other.rsplit("/", 1)[-1][:-3] == name.rsplit(".", 1)[-1]) for name in names):
                    graph[path].add(other)
    for _ in files:
        for path in graph:
            graph[path].update({p for d in list(graph[path]) for p in graph[d]} - {path})
    return graph


def plan_reviews(snapshot, *, limit, per_batch, max_batches):
    from backend.stages.critic import CriticError, _outline
    if limit < 1500:
        raise CriticError("CRITIC_MAX_INPUT_BYTES must leave at least 1,500 bytes for review data.")
    files, requirements = snapshot["files"], snapshot["requirements"]
    graph = dependencies(files)
    context, remaining = {}, 2000
    for path, code in sorted(snapshot["context"].items(), key=lambda p: len(p[1].encode())):
        if len(code.encode()) <= remaining:
            context[path] = code
            remaining -= len(code.encode())
    def payload(reqs, paths, project=False):
        omitted = sorted(set(files) - set(paths))
        return {"requirements": reqs, "sources": {p: files[p] for p in paths},
                "changes": {p: {"original": snapshot["originals"][p]} for p in paths},
                "repository_context": context, "interfaces": {}, "fragmented": False,
                "scope": "project" if project else "module", "omitted_sources": omitted,
                "context_omitted": sorted(set(snapshot["context"]) - set(context))}
    whole = payload(requirements, list(files), True)
    if len(requirements) <= per_batch and _size(whole) <= min(limit, 20000):
        return [whole]
    groups = []
    for module, path in snapshot["mapping"].items():
        reqs = [r for r in requirements if r["target_module"] == module]
        if reqs:
            groups.append((reqs, [path, *sorted(graph[path])], False))
    # Lexical routing selects full candidate files, not snippets presented as complete code.
    # Any omitted files are disclosed; absence in this scope cannot establish a global omission.
    stop = {"the", "and", "must", "should", "with", "from", "that", "when", "not", "for", "are", "all", "user", "users", "system"}
    def words(text):
        return {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", text.lower().replace("_", " "))} - stop
    descriptors = {p: words(p + " " + _outline(code) + " " + " ".join(r["description"] for r in requirements
        if r["target_module"] is not None and snapshot["mapping"].get(r["target_module"]) == p)) for p, code in files.items()}
    global_groups = {}
    global_requirements = [r for r in requirements if r["target_module"] is None]
    if global_requirements and len(global_requirements) <= per_batch and _size(payload(global_requirements, list(files), True)) <= limit:
        global_groups[tuple(files)] = global_requirements
    for requirement in ([] if global_groups else global_requirements):
        terms = words(requirement["description"])
        scores = {p: len(terms & terms_in_file) for p, terms_in_file in descriptors.items()}
        best = max(scores.values(), default=0)
        paths = [p for p in files if not best or scores[p] >= max(1, best * .5)]
        paths = sorted(set(paths) | {d for p in paths for d in graph[p]})
        global_groups.setdefault(tuple(paths), []).append(requirement)
    # Combine global groups when their full implementing sources fit together.
    for paths, reqs in global_groups.items():
        if groups and groups[-1][2] and len(groups[-1][0]) + len(reqs) <= per_batch:
            previous, previous_paths, _ = groups[-1]
            union = sorted(set(previous_paths) | set(paths))
            if _size(payload(previous + reqs, union, True)) <= limit:
                groups[-1] = (previous + reqs, union, True)
                continue
        groups.append((reqs, list(paths), True))
    batches = []
    for reqs, paths, project in groups:
        for offset in range(0, len(reqs), per_batch):
            subset = reqs[offset:offset + per_batch]
            selected = list(paths)
            data = payload(subset, selected, project)
            while _size(data) > limit and len(selected) > 1:
                selected.pop()
                data = payload(subset, selected, project)
            if _size(data) <= limit:
                batches.append(data)
                continue
            # Oversized source is explicitly fragmented and cannot receive a passing verdict.
            path = selected[0]
            data["fragmented"] = True
            data["sources"] = {path: ""}
            data["changes"] = {}
            available = limit - _size(data) - 150
            chunk, start = [], 1
            for line_number, line in enumerate(files[path].splitlines(True), 1):
                if _size("".join(chunk) + line) > available:
                    if not chunk:
                        raise CriticError("A source line or requirement group exceeds the review budget.")
                    batches.append({**data, "sources": {path: "".join(chunk)}, "start_line": start})
                    chunk, start = [], line_number
                chunk.append(line)
            if chunk:
                batches.append({**data, "sources": {path: "".join(chunk)}, "start_line": start})
    if len(batches) > max_batches:
        raise CriticError(f"Review needs {len(batches)} groups; the configured limit is {max_batches}. No model calls were made.")
    return batches
