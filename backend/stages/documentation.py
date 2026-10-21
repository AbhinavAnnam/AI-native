"""Deterministic documentation: no inference, project execution, or Git mutation."""
import ast
import difflib
import hashlib
import html
from collections import Counter

from backend.models.critic import CriticReport
from backend.models.delivery import DeliveryView, DocumentationResult, FileChange
from backend.stages.critic import ensure_current_review, pending_repair_matches


class DeliveryError(RuntimeError):
    """An actionable delivery prerequisite failed."""


def accepted_snapshot(state, expected_fingerprint=None):
    snapshot = ensure_current_review(state, expected_fingerprint)
    if pending_repair_matches(snapshot, state.get("critic_pending_repair") or {}):
        raise DeliveryError("Complete the pending repair and critic review before continuing.")
    drafts = state.get("drafts") or []
    if state.get("status") != "approved" or any(d.get("status") != "COMMITTED" for d in drafts):
        raise DeliveryError("Approve and save all reviewed files in Phase 3 first.")
    for path, source in snapshot["files"].items():
        if snapshot["disk"][path] != source:
            raise DeliveryError("Saved files differ from the approved candidate. Refresh and review again.")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise DeliveryError("An approved file has invalid Python syntax. Revise it before continuing.") from exc
    report = CriticReport.model_validate(state["critic_report"])
    expected_ids = {r["id"] for r in snapshot["requirements"]}
    actual_ids = [c.requirement_id for c in report.coverage]
    if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
        raise DeliveryError("The critic report does not cover the complete requirement inventory. Run Critic again.")
    return snapshot, report


def _cell(value):
    return html.escape(str(value), quote=False).replace("|", "&#124;").replace("\n", "<br>")


def _symbols(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def generate_documentation(state, thread_id, expected_fingerprint=None):
    """Return a reproducible, structured report for the accepted candidate."""
    snapshot, report = accepted_snapshot(state, expected_fingerprint)
    repo = state.get("repository") or {}
    # A fixed run-specific name avoids overwriting a repository's existing README/report.
    suffix = hashlib.sha256(thread_id.encode()).hexdigest()[:12]
    filename = f"AI_NATIVE_REVIEW_{suffix}.md"
    changes = []
    for path, source in sorted(snapshot["files"].items()):
        original = snapshot["originals"].get(path, "")
        delta = list(difflib.ndiff(original.splitlines(), source.splitlines()))
        before, after = _symbols(original), _symbols(source)
        symbols = [f"{name}: {'added' if name not in before else 'removed' if name not in after else 'changed'}"
                   for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]
        changes.append(FileChange(path=path, status="added" if not original else
                                  "unchanged" if original.strip() == source.strip() else "modified",
                                  added_lines=sum(s.startswith('+ ') for s in delta),
                                  removed_lines=sum(s.startswith('- ') for s in delta), symbols=symbols))
    counts = Counter(c.status for c in report.coverage)
    categories = Counter(f.category for f in report.findings)
    title = (state.get("base_specs") or {}).get("project_title") or "Requirements implementation"
    lines = [f"# {_cell(title)} — change report", "", f"Run: `{thread_id}`",
             f"Candidate: `{snapshot['candidate_version']}`", "",
             "## Change summary", "", _cell(report.summary), "",
             f"Critic verdict: **{report.verdict}**. {len(report.coverage)} requirements assessed.",
             f"Coverage: {', '.join(f'{k}: {counts[k]}' for k in ('met', 'partial', 'missing', 'unclear'))}.",
             f"Findings: {categories['defect']} confirmed defects, {categories['uncertain']} uncertain findings, "
             f"{categories['suggestion']} suggestions.", ""]
    if repo:
        lines += [f"Repository: {_cell(repo['url'])}", f"Base branch: `{repo['base_branch']}`",
                  f"Baseline commit: `{repo['baseline_commit']}`", f"Working branch: `{repo['work_branch']}`", ""]
    else:
        lines += ["Destination: `generated_src/`. No repository was supplied; PR publishing is unavailable.", ""]
    lines += ["## Files and modifications", "", "| File | Change | Lines added | Lines removed |",
              "| --- | --- | ---: | ---: |"]
    for change in changes:
        lines.append(f"| {_cell(change.path)} | {change.status} | {change.added_lines} | {change.removed_lines} |")
    lines.append("")
    for change in changes:
        if change.symbols:
            lines += [f"### {_cell(change.path)}", "", *[f"- {_cell(s)}" for s in change.symbols], ""]
    lines += ["## Validation and testing", "",
              "- Approved Python files were checked for valid syntax and matched to the reviewed candidate.",
              "- **Automated tests were not run in this delivery workflow. The testing phase was skipped.**",
              "- The critic is a static, model-assisted assessment; its verdict does not establish runtime correctness.",
              "- Remaining findings are disclosed below for the pull request reviewer.", "",
              "## Requirement coverage", "", "| Requirement | Module | Description | Status | Evidence |",
              "| --- | --- | --- | --- | --- |"]
    for c in report.coverage:
        lines.append("| " + " | ".join(_cell(v) for v in (
            c.requirement_id, c.target_module or "Project-wide", c.description, c.status, c.evidence)) + " |")
    lines += ["", "## Remaining findings", ""]
    if not report.findings:
        lines += ["The critic reported no remaining findings. This is not a test result.", ""]
    for f in report.findings:
        lines += [f"### {f.id}: {f.category} ({f.severity})", "",
                  f"File: {_cell(f.file_path)}, line {f.line or 'unspecified'}.",
                  f"Requirements: {', '.join(f.requirement_ids)}", "", _cell(f.explanation), "",
                  "Suggested correction: " + _cell(f.suggested_fix), ""]
        for label, value in (("Requirement evidence", f.requirement_quote), ("Code evidence", f.code_quote),
                             ("Counterexample", f.counterexample)):
            if value:
                lines += [f"{label}: {_cell(value)}", ""]
    lines += ["## Review limitations", "", *[f"- {_cell(v)}" for v in report.limitations], "",
              "This report was assembled from saved requirements, approved code, the repository baseline, "
              "and the current critic report. No additional model call was made.", ""]
    markdown = "\n".join(lines)
    document_id = hashlib.sha256((snapshot["input_fingerprint"] + markdown).encode()).hexdigest()
    return DocumentationResult(document_id=document_id, candidate_version=snapshot["candidate_version"],
        input_fingerprint=snapshot["input_fingerprint"], filename=filename, markdown=markdown, files=changes,
        requirements_assessed=len(report.coverage), findings=len(report.findings))


def current_documentation(state, thread_id, document_id, expected_fingerprint):
    fresh = generate_documentation(state, thread_id, expected_fingerprint)
    saved = state.get("documentation") or {}
    if fresh.document_id != document_id or saved != fresh.model_dump():
        raise DeliveryError("Documentation is outdated. Generate and preview the report again before publishing.")
    return fresh


def delivery_view(values):
    documentation = values.get("documentation")
    stale = False
    if documentation:
        try:
            current = generate_documentation(values, values.get("delivery_thread_id", ""))
            stale = current.model_dump() != documentation
        except (ValueError, OSError, RuntimeError):
            stale = True
    return DeliveryView(documentation=documentation, documentation_stale=stale,
                        pull_request=values.get("pull_request"))
