# AI-Native MCP input reference

Read the connected tool schemas before constructing calls. Names below are server-side names; the client may prefix them. Successful calls return existing backend JSON in `structuredContent.data` (some clients display this simply as `data`). MCP errors have `isError=true`; safe error text can contain `status_code`, `retry_after`, and `execution_state_unknown`.

The examples use fictional IDs, specification content, and all-zero fingerprints. Replace them with user-authorized inputs and actual returned state. Preserve complete extraction JSON in real calls. Approval/publication examples deliberately have `confirm=false`; change it only after the relevant explicit user approval described in `SKILL.md`.

## Extract requirements

`pdf_paths` are local to the MCP server, inside its configured documents root. The adapter accepts 1–10 PDFs, at most 20 MB each and 50 MB combined. A path on another laptop is not automatically accessible to this server.

Tool: `extract_requirements`

```json
{"pdf_paths": ["requirements.pdf"]}
```

## Generate drafts for a new run

`task_spec` is a string containing the complete serialized specification, not a nested JSON object. Use a new unique run ID. Omit `repository` for non-repository generation; its existing output cleanup still applies. Optional `base_branch` and `file_mappings` belong inside `repository`; retain supplied mappings rather than inventing them.

Tool: `generate_code`

```json
{
  "request": {
    "thread_id": "example-run-001",
    "task_spec": "{\"project_title\":\"Example only\",\"features\":[{\"feature_name\":\"Accounts\",\"target_module\":\"accounts\",\"business_rules\":[\"Reject duplicate email addresses\"],\"functions\":[],\"exceptions\":[]}],\"cross_cutting_constraints\":[]}",
    "repository": {"url": "https://github.com/example/example.git"}
  }
}
```

## Read saved state

Tool: `get_run_state`

```json
{"thread_id": "example-run-001"}
```

Use returned `modules`, `missing_modules`, `status`, `error`, `critic_report`, `critic_stale`, `critic_progress`, `critic_pending_repair`, `critic_repair_rounds`, `documentation`, `documentation_stale`, and `pull_request` to select the next authorized action. Keep the actual `thread_id`; a repository workspace `run_id` is a different identifier.

## Request a code revision

Tool: `revise_code`

```json
{"request": {"thread_id": "example-run-001", "target_module": "accounts", "user_feedback": "Reject email duplicates case-insensitively while preserving existing public interfaces."}}
```

Omit `target_module` for an authorized project-wide revision. An optional `modified_spec` changes requirements and should represent a user-requested specification change. Approval-like feedback is rejected here; use `approve_code` instead.

## Retry missing drafts

Tool: `retry_missing_modules`

```json
{"thread_id": "example-run-001"}
```

## Run or resume critic

Tool: `run_critic`

```json
{"request": {"thread_id": "example-run-001"}}
```

An optional `request.max_batches` is the existing cost ceiling (1–32), not a desired batch count. Usually omit it to retain backend configuration. Do not change it solely to overcome a failed review.

## Repair and automatically re-review

Tool: `repair_findings`

```json
{"request": {"thread_id": "example-run-001", "input_fingerprint": "0000000000000000000000000000000000000000000000000000000000000000"}}
```

Take the actual hash from `critic_report.input_fingerprint`. Automatic re-review belongs to this action; an extra `run_critic` call is unnecessary when it already returned a complete current report. A pending interrupted repair resumes its existing round.

## Accept/save the reviewed candidate

Tool: `approve_code`

```json
{"request": {"thread_id": "example-run-001", "input_fingerprint": "0000000000000000000000000000000000000000000000000000000000000000"}, "confirm": false}
```

Add top-level `target_module` only for explicit single-file acceptance; otherwise all reviewed files are accepted. Set `confirm=true` only after user acceptance of that candidate. Read the returned fingerprint for documentation, since approval writes can refresh it. `COMMITTED` means saved locally, not pushed to GitHub.

## Generate documentation

Tool: `generate_documentation`

```json
{"request": {"thread_id": "example-run-001", "input_fingerprint": "0000000000000000000000000000000000000000000000000000000000000000"}}
```

This requires accepted files and a current complete review. It returns Markdown under `documentation.markdown` plus `documentation.document_id` and performs no additional model inference. It does not publish.

## Publish a draft PR

Tool: `create_draft_pr`

```json
{"request": {"thread_id": "example-run-001", "input_fingerprint": "0000000000000000000000000000000000000000000000000000000000000000", "document_id": "0000000000000000000000000000000000000000000000000000000000000000", "title": "Implement reviewed requirements"}, "confirm": false}
```

Use the actual previewed `documentation.document_id` and latest review fingerprint, with `confirm=true` only after publication authorization. The backend uses its configured GitHub token; the agent does not need to request, read, or include secret values. Publishing targets the supplied repository (including a supplied fork); automatic cross-fork upstream PR targeting is not implemented. Report returned `pull_request.url`; never infer publication from local approval alone.
