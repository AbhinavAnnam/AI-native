"""Publish only an explicitly approved, documented candidate. Never merge or force push."""
import re
from pathlib import Path

from backend.models.delivery import PullRequestResult
from backend.stages.documentation import DeliveryError, current_documentation
from backend.stages.repository import repository_file
from backend.tools.git_provider import RepositoryError, run_git
from backend.tools.github_provider import GitHubProvider, github_token, repository_slug


def _names(output):
    return {name for name in output.split("\0") if name}


def publish_pull_request(state, thread_id, document_id, expected_fingerprint, title, *,
                         provider=None, git=run_git, on_progress=None):
    """Reusable service; callbacks persist completed local steps for safe retries."""
    document = current_documentation(state, thread_id, document_id, expected_fingerprint)
    repo = state.get("repository")
    if not repo:
        raise DeliveryError("PR creation requires a repository run. You can download the documentation instead.")
    published = state.get("pull_request")
    if published:
        result = PullRequestResult.model_validate(published)
        if result.document_id != document_id:
            raise DeliveryError("This run already published a different candidate. Start a new run for another PR.")
        return result
    slug = repository_slug(repo["url"])
    root = Path(repo["repo_dir"])
    branch, base = repo["work_branch"], repo["base_branch"]
    if not re.fullmatch(r"[a-f0-9]{32}", repo["run_id"]) or branch != f"ai-native/{repo['run_id']}" or branch == base:
        raise DeliveryError("Publishing requires this run's separate ai-native working branch.")
    if git(["branch", "--show-current"], root) != branch:
        raise DeliveryError("The clone is on a different branch. Restore the run's working branch before publishing.")
    if repository_slug(git(["remote", "get-url", "origin"], root)) != slug:
        raise DeliveryError("The repository origin changed. Publishing was stopped.")
    baseline = repo["baseline_commit"]
    if not re.fullmatch(r"[a-f0-9]{40,64}", baseline):
        raise DeliveryError("Invalid repository baseline.")
    git(["merge-base", "--is-ancestor", baseline, "HEAD"], root)
    allowed = set(repo["file_mappings"].values()) | {document.filename}
    changed = set()
    for args in (["diff", "--name-only", "-z", baseline, "HEAD"],
                 ["diff", "--name-only", "-z"], ["diff", "--cached", "--name-only", "-z"],
                 ["ls-files", "--others", "--exclude-standard", "-z"]):
        changed |= _names(git(args, root))
    if changed - allowed:
        raise DeliveryError("The clone contains unrelated changes. Move or commit them separately before publishing this run.")
    progress = state.get("publication_progress") or {}
    report_path = repository_file(str(root), document.filename)
    baseline_paths = _names(git(["ls-tree", "-r", "--name-only", "-z", baseline], root))
    if document.filename in baseline_paths:
        raise DeliveryError("The report filename already exists in the baseline. Publishing will not overwrite it.")
    if report_path.exists() and report_path.read_bytes() != document.markdown.encode():
        raise DeliveryError("The report file contains other content. Move it aside before publishing.")
    # Check the Git index flags/modes as well as the source bytes. An ignored/skip-worktree
    # candidate must never silently disappear from the commit.
    for module, relative in repo["file_mappings"].items():
        path = repository_file(str(root), relative)
        source = next(d["code"] for d in state["drafts"] if d["target_module"] == module)
        if not path.is_file() or path.read_bytes() != source.encode():
            raise DeliveryError("An approved source file changed on disk. Review the candidate again.")
    provider = provider or GitHubProvider(github_token())
    provider.preflight(slug)
    try:
        git(["var", "GIT_AUTHOR_IDENT"], root)
        git(["var", "GIT_COMMITTER_IDENT"], root)
    except RepositoryError as exc:
        raise DeliveryError("Configure Git user.name and user.email for the backend account before publishing.") from exc
    head = git(["rev-parse", "HEAD"], root)
    if not progress and head != baseline:
        raise DeliveryError("This branch contains commits outside the saved publishing workflow. Inspect the clone before publishing.")
    if progress and (progress.get("document_id") != document_id or progress.get("commit_sha") != head):
        raise DeliveryError("The branch changed after a publishing attempt. Inspect the clone before retrying.")
    report_path.write_bytes(document.markdown.encode())
    git(["add", "--", *sorted(allowed)], root)
    staged = _names(git(["diff", "--cached", "--name-only", "-z"], root))
    if staged - allowed:
        raise DeliveryError("Unexpected staged files were detected. Publishing was stopped.")
    # Verify the actual index blob, including assume-unchanged/skip-worktree cases.
    for relative in sorted(allowed):
        expected_blob = git(["hash-object", "--", relative], root)
        indexed = git(["rev-parse", f":{relative}"], root)
        if indexed != expected_blob:
            raise DeliveryError("Git did not stage the complete approved candidate. Check index flags and attributes.")
    if staged:
        git(["-c", "commit.gpgSign=false", "commit", "--no-gpg-sign", "-m", "Implement reviewed requirements and add change report"], root)
        head = git(["rev-parse", "HEAD"], root)
    progress = {"document_id": document_id, "commit_sha": head}
    if on_progress:
        on_progress(progress)
    # A direct, validated URL ignores altered pushurl/refspec configuration. No force.
    git(["push", "--", f"https://github.com/{slug}.git", f"HEAD:refs/heads/{branch}"], root,
        github_token=provider.token)
    existing = provider.find_pull_request(slug, branch, base)
    body = (f"Implements the approved requirements candidate.\n\n"
            f"Change analysis and complete requirement coverage: [{document.filename}](https://github.com/{slug}/blob/{head}/{document.filename}).\n\n"
            f"Requirements assessed: {document.requirements_assessed}. Remaining critic findings: {document.findings}.\n\n"
            "Automated tests were not run; the testing phase was skipped. Syntax validation and static critic review were performed. "
            "See the report for known defects, uncertain items, and suggestions.\n\n"
            f"Candidate: `{document.candidate_version}`\nDocument: `{document.document_id}`")
    value = existing or provider.create_pull_request(slug, branch, base, title.strip(), body)
    value = provider.validate_pull_request(value, slug, branch, base, head)
    return PullRequestResult(document_id=document_id, number=value["number"], url=value["html_url"],
                             head=branch, base=base, commit_sha=head)
