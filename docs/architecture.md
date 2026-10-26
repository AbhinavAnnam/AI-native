# Workflow boundaries

LangGraph in `backend/agent/graph.py` owns execution order, revisions, and per-file human approval. FastAPI routes and the Streamlit UI provide access to that workflow. The FastAPI lifespan opens a SQLite checkpointer at `workspace/checkpoints.sqlite` and closes it on shutdown. The synchronous graph runs in a worker thread so its SQLite checkpointer and synchronous model/file operations do not require an asynchronous saver.

## Implemented stages

| Capability | Implementation | Contract |
| --- | --- | --- |
| Requirements extraction | `phase1_rag/` | Uploaded documents to structured project specifications |
| Code module planning | `backend/stages/planning.py` | Extracted features to grouped specifications and original-feature membership |
| Repository preparation | `backend/stages/repository.py` | `RepositoryRequest` to `RepositoryWorkspace` |
| Code generation/revision | `phase2_coder/coder.py` | Specification, current source, optional repository context to drafts |
| Draft provider routing | `phase2_coder/providers.py` | Complete module prompt to AST-valid text through bounded, recovering fallbacks |
| Critic review and repair | `backend/stages/critic.py` | Candidate snapshot to validated report; current findings plus existing coder callable to repaired drafts |
| Approved repository writes | `backend/stages/repository.py` | Workspace, mapped module, approved Python source to updated content hash |
| Documentation | `backend/stages/documentation.py` | Accepted candidate and complete critic report to deterministic `DocumentationResult` |
| Draft PR publishing | `backend/stages/pull_request.py` | Previewed document and approved repository state to `PullRequestResult` |

Repository contracts live in `backend/models/repository.py`. Concrete Git operations live in `backend/tools/git_provider.py`. Reusable stages do not import FastAPI, Streamlit, or LangGraph.

## Documentation and PR boundary

Phase 4 combines the user interface while keeping two service operations: `generate_documentation` and `publish_pull_request`. `backend/models/delivery.py` defines strict Pydantic request/result contracts; `backend/routers/delivery_router.py` supplies locking, SQLite persistence and HTTP errors. The optional MCP adapter calls these existing HTTP routes; no tester phase is added.

## MCP adapter

`mcp_tools/server.py` exposes typed tools through the official SDK's stdio transport. `mcp_tools/api_client.py` forwards to the running local backend, preserving its graph instance, SQLite connection, workflow locks, model budgets, review/approval guards, and publishing behavior. The adapter never imports `backend.main` or starts another graph. Input models reuse the backend's request contracts. Structured successes wrap the existing JSON response under `data`; safe tool errors preserve HTTP status, retry hints, and uncertainty after connection failures. Actions are never automatically replayed.

PDF access is restricted to an operator-selected local directory and bounded uploads. Backend URLs are local loopback only; credentials remain in the backend. Approval and publication require explicit caller confirmation, while backend fingerprints and prerequisites remain authoritative. Clients must retain human approval controls; confirmation flags are not authentication. Optional SDK dependencies are installed separately so the working API/UI environment stays unchanged. See [MCP setup](mcp.md).

Project approval reuses the existing per-file save guards under one workflow lock, refreshing the report fingerprint after each expected write. Completed saves remain checkpointed if a subsequent write fails. This is resumable approval, not an atomic multi-file filesystem transaction. Missing/invalid drafts, stale review and pending repair block approval/delivery.

Documentation is rendered locally from accepted source, baseline diffs, AST symbol changes, complete coverage and all remaining findings. The Markdown preview and structured result live in the existing SQLite checkpoint. Their deterministic content hash binds publishing to the preview. The generated report clearly says that automated tests were not run; syntax validation and model review do not establish runtime correctness. No additional LLM inference is required.

Explicit publishing writes the report into the clone, stages only mapped source paths and that report, verifies index contents, and commits/pushes the run's separate branch. Unrelated changes and unrecognized commits stop publishing. Credentials are read from `GITHUB_TOKEN`/`GH_TOKEN`; Git authentication uses process-local headers rather than credentials in command arguments or stored remote URLs. Hooks and interactive Git prompts remain disabled. `backend/tools/github_provider.py` handles bounded GitHub REST calls with sanitized errors and draft-only PR validation. Browser-origin checks protect these token-backed routes from remote websites despite the legacy app's broad CORS configuration.

The commit SHA is checkpointed before pushing. Retrying after a push/API failure reuses that commit and looks up the matching open PR before creating another. PR URLs, branches, repository identity, draft state and head commit are checked against the expected result. A process interruption in the narrow interval between local commit and checkpointing may require manual inspection; unknown commits are never automatically pushed. PR creation requires write access to the supplied repository; fork workflows are outside this phase.

## Extraction integrity

`phase1_rag/extractor.py` requests schema-constrained `ProjectSpecs` for each document chunk. Native and fallback responses are locally validated; invalid schema, unexpected fields, empty facts and truncated/refused responses cannot count as successful chunks. Every uploaded document must provide readable text and every chunk must succeed before finalization.

`phase1_rag/consolidation.py` replaces the model-based reduction with a deterministic merge by exact module/category identifiers. It preserves every rule, function signature/purpose pair, exception and constraint, removing only exact duplicates. An input/output fact inventory guards the merge. Conflicting requirements remain visible. No model rewrites the full extracted specification and there is no lossy key allowlist.

`phase1_rag/grouping.py` enforces five to seven code modules for plans above seven. A compact ID-only request uses existing Phase 1 providers; its schema limits output to five through seven groups, and every feature ID must occur exactly once. `merge_module_groups` in `backend/stages/planning.py` copies all rules, function/purpose pairs, exceptions and feature labels locally and checks the remapped fact inventory. Cross-cutting constraints stay unchanged. Grouping retains a 2,048-token output cap, at most two Gemini candidates (20-second request timeout, no SDK retries) and two fallback attempts without backoff retries. Oversized descriptors (18 KB), invalid plans and unavailable providers use deterministic local grouping without additional inference: seed package namespaces and large features, then assign remaining features to compatible groups favoring domain-name overlap and balanced byte sizes. At most seven groups are considered per feature. New groups cannot cross namespaces or exceed the 18 KB feature-data budget. If no safe grouping is found, extraction raises an explicit error; it never returns fourteen modules or truncates requirements to satisfy the cap. Plans already at seven or fewer remain intact, including small documents below five. The resulting `ProjectSpecs` drives both repository and non-repository coding; Phase 2 routing is unchanged.

Logs expose per-chunk/final counts, and the frontend shows a specification summary and clears stale specs when a new extraction fails. This protects extracted facts during consolidation, not semantic completeness of a PDF-to-model extraction. Chunk intermediates are held in memory for the merge; the existing API does not save private intermediate response artifacts. Existing coder/critic and repository storage behavior is unchanged.

## Repository run

Automatic repository runs plan code modules before file selection. `plan_code_modules` merges equal or unambiguously nested domain names within a package namespace, ignoring generic labels such as management/settings. Ambiguous parent domains stay separate. It preserves exact rules, function/purpose pairs, exceptions and global constraints, and checks the remapped fact inventory. The graph retains the original `task_spec`, the planned `base_specs`, and `module_groups` in checkpoints; the API and Phase 2 UI expose group membership. Explicit file overrides bypass grouping to preserve requested module/path identities.

`MAX_REPOSITORY_MODULES` sets a shared ceiling of 50 for automatic planning and explicit mapping contracts. The graph checks the planned count before cloning. Exceeding a limit reports an error without dropping features or changing existing source.

The request optionally supplies a GitHub HTTPS URL, base branch, and module-to-file overrides. When provided, the graph validates that overrides match all specification modules before cloning. Otherwise, repository preparation matches module names against source file names, ignoring tests and package initializers; unmatched modules are placed alongside the source package. Ambiguous matches or collisions stop generation with an actionable error. Each run gets an independent clone and work branch under `workspace/<run_id>/repo/`; `repository.json` beside the clone records the initial snapshot. Graph state retains repository metadata, current drafts, and the expected file hashes after approved writes. A `candidate_version` content hash identifies the specification, baseline, mappings, and draft code; it changes when that content changes and remains stable during approval.

Only mapped Python files can be saved through repository approval. Repository-relative paths are checked for traversal, hidden directories, Windows reserved names, and escaping links. Approval validates Python syntax and checks the current file hash against the last known version before writing. Repository preparation reads bounded text context and does not execute project code. The coder does not run repository tests in this slice.

Without a repository request, the existing generation mode continues to write approved modules into `generated_src/`.

## Draft provider recovery

`phase2_coder/providers.py` owns normal draft generation and ordinary revisions; critic repair still delegates to `backend/stages/repair.py`. The default order is Groq 20B, Groq 120B, Gemini Flash Lite, Nemotron 3 Ultra `:free`, then other free OpenRouter routes (optional override, DeepSeek V4 Flash, North Mini Code, general free router) if budget remains. Up to four inference calls share a 120-second module budget. Groq requests default to 15 seconds, Nemotron to 60 seconds, and ordinary requests to 45 seconds, all cancellable and bounded by remaining time. Earlier routes reserve capacity for a healthy Gemini request and one module-level syntax correction. Multiple configured provider families skip quota waits; single-provider setups retain at most ten seconds of short waits. Nemotron Ultra uses native low reasoning effort and 1,024 completion tokens of headroom above the existing 16,384-token code allowance; internal reasoning is not assigned an unsupported numeric token limit. Full prompts are never truncated to fit Groq: input estimates plus the actual completion cap must fit 7,500 tokens and leave at least 2,048 completion tokens, otherwise a fallback receives the complete prompt. Finish reasons and AST validation reject incomplete/empty results.

Process-local model cooldowns use hashed API-key identity and monotonic deadlines. Retry hints, including daily-token hints, determine quota expiry; timeouts cool down for three minutes, other transient failures for 30 seconds and invalid/unavailable-model responses for five minutes. Cooling routes skip endpoint discovery. Shared OpenRouter account limits suppress redundant attempts against its other free routes; explicitly identified upstream limits can still use another provider. Expired models are eligible on subsequent generation calls without restart. Public free-endpoint metadata is cached for 60 seconds and only zero prompt/completion prices qualify. Paid OpenRouter overrides are rejected. No new dependencies or secrets are required.

The coder reports bounded module failures separately from successful drafts. The graph saves successful results and keeps previous candidates for failed revisions, without starting another unbounded fallback chain. SQLite persistence still occurs at graph node completion; interrupted generation does not gain per-file checkpoints in this change.

Phase 2 feedback recognizes the explicit `Retry missing modules` action. `generate_code_node` selects only specification modules without a valid nonempty saved draft, retains the full project requirements/context, and merges new successes with untouched earlier drafts. Start/state/feedback responses compute `missing_modules` from persisted specifications and drafts, so older incomplete runs can resume without a new clone. The frontend exposes that single retry action when needed. A complete run receives a no-op rather than fresh inference. Ordinary project revisions still regenerate all selected modules.

Draft providers allow one syntax-correction request per file, within the same four-call and 120-second defaults. It carries full original requirements, the rejected code and only the exception type/line/column diagnostic; raw SyntaxError messages and source text are not logged. Corrected Groq requests are re-estimated against the complete input/output budget. Syntax, empty and truncated-response errors are distinct from outage cooldowns and do not disable a provider for later files. Actual network, server, quota and key failures retain cooldowns. The fallback queue checks previously skipped providers for recovery before another route, revisiting each at most once without extending the budget. No additional endpoint or dependency is introduced.

## Next stages

Add testing, documentation, and pull-request functions under `backend/stages/`, with structured result models under `backend/models/`. Keep branching, retry limits, and approval decisions in the graph. Critic and tester results must identify the candidate version they evaluated; code revisions invalidate those results.

The tester will evaluate candidate files in an isolated test workspace before approval and verify the final approved files. Documentation will use the recorded baseline commit, actual diff, review findings, and test evidence. PR creation will commit selected outputs and push the run's working branch through explicit repository tools.

## Future MCP and skills

An MCP server can wrap these same stage functions with typed tool registration. The graph can continue calling Python functions directly. Skills can describe how compatible clients should invoke and interpret the tools. Neither adapter should duplicate stage logic or workflow state.

## Current limits

Repository mode supports GitHub HTTPS, Python files, file-name matching with optional explicit overrides, and bounded context. It does not perform semantic repository-wide planning, run repository code, create commits, push, or create PRs. Agent Studio resumes pending reviews from SQLite after backend restarts when the database and repository workspace remain available. The initial repository manifest alone is insufficient to recover reviews created before persistence was enabled. The existing global draft queue remains a development convenience; persisted graph drafts are authoritative for Agent Studio approvals. This SQLite setup targets the local application; production concurrency and deployment require a separate persistence/deployment design.

## Critic integration

`backend/models/critic.py` defines the transport contracts. `backend/stages/critic.py` builds stable requirement inventories and bounded source snapshots, validates review results, and coordinates targeted draft repairs. Its operations do not import FastAPI, Streamlit, or LangGraph. The existing coder is injected as a bounded callable for repairs. No new provider, framework, database, MCP server, or skill is introduced.

`backend/routers/critic_router.py` adapts thread IDs to persisted graph state and calls the stage in a worker thread. Optional graph fields store the report, repair-attempt count, and previous candidate in the existing SQLite checkpoints. Updating those fields preserves the generation interrupt; successful repairs reopen completed runs for approval. Failed repairs keep prior drafts and reports while counting the attempted round.

Phase 2 exposes draft generation and revision. Phase 3 exposes review, targeted repair with automatic re-review, and per-file approval. The approval boundary checks a current input fingerprint even when called through the older feedback API; the legacy direct-save endpoint is disabled. A non-passing static report remains visible and may be explicitly accepted by the human for later testing. Valid syntax and existing repository overwrite checks remain mandatory.

Reports bind to specifications, mappings, baseline, exact candidate code, relevant current disk inputs, and budget settings. Authored file changes invalidate the report. Expected writes of reviewed source refresh its fingerprint without another model call. Reports can be downloaded as JSON; all authoritative state stays in SQLite. The current Phase 2 thread is selected automatically. Optional recovery controls can load a previous browser session.

Budgeting favors cached reports, complete current source with compact baseline representations, shared interface outlines, and bounded requirement batches. Incomplete source fragments never produce `pass`. Automatic repair does not run tests or write target source files. Two repair attempts are permitted per thread, and failed attempts count. A later tester must supply execution evidence before PR readiness can be established.

### Scoped reviews and grounded findings

`backend/stages/review_planning.py` owns reusable planning without HTTP or checkpoint dependencies. It groups module-specific requirements with full implementing source and detected transitive imports/calls. Global constraints are assessed separately, once; full project source is used when it fits. Default bounds are 64 requirements, 128 KB input (including a reserve), and 8,192 nominal output tokens. Larger or omitted scopes are disclosed. Fragments cannot pass. Only Groq-only deployments subdivide otherwise complete scopes for the smaller Groq envelope.

`critic-v4` requires native structured output and local validation. Findings distinguish defects, uncertainty, and suggestions, with requirement/source quotes and a concrete counterexample. `ground_review` downgrades unsupported defects and unsupported missing/partial assessments. Missing context never establishes missing implementation. The prompt prohibits strengthening permissive requirements into invented obligations. These are evidence and consistency checks, not a proof of model correctness.

Only confirmed defects and supported module-specific unmet requirements target repairs; global uncertainty no longer targets every module. Existing coder injection remains in `bounded_repairer`. Original candidates stay authoritative until the staged repaired candidate has a complete review. `reuse_review_progress` checks exact batch keys and source/dependency changes before carrying unaffected results into that review; project-wide checks are invalidated by any changed candidate file. Runtime dependency discovery and project test execution remain outside this stage.

### One action, bounded execution

The API wraps each user action in a shared 600-second deadline (configurable up to 660), covering repair, re-review and at most one short quota recovery. Stage limits default to 600 seconds and cannot extend the operation deadline. Ordinary sockets have 45-second cancellable deadlines; Nemotron JSON reviews have 90 seconds, configurable up to 120. A short quota pause up to 20 seconds resumes saved work once inside the operation. Long waits and permanent failures return a clear error with progress preserved. No unbounded retry loop exists.

The Streamlit frontend sends one POST per action and polls saved status with read-only GETs while showing progress. It never resubmits a POST automatically. The same Repair findings button resumes either staged generation or its review after an external interruption. Two user-triggered repair rounds remain the ceiling; resuming a pending round does not consume another round. Approval stays disabled while a pending candidate lacks a complete review.

`critic_progress` checkpoints validated results and retry times. `critic_pending_repair` stores staged code, completed modules, original drafts and the initial review seed used for selective re-review. The assessment version and budget settings bind report fingerprints. Old v3 judgments are intentionally stale under v4. Existing staged repaired code survives migration, but is reviewed before promotion; unfinished old-policy repair instructions are not executed automatically.

### Providers and failures

Critic OpenRouter routes are explicitly free; extraction overrides cannot leak into critic or repair. Groq uses its complete-envelope guard. Large reviews with Gemini configured try one preferred OpenRouter route before Gemini rather than chaining another slow generic-router call. Without Gemini, the bounded general free-router fallback remains available. Phase 1 and ordinary Phase 2 defaults are unchanged.

`phase1_rag/providers.py` holds process-local key/provider/model cooldowns shared by review and repair. Timeouts cool a route for ten minutes; truncation/invalid/unavailable responses for five. A new button click honors that cooldown, expired routes recover automatically, and backend restart clears this process memory. Separate quota state honors Retry-After/Google RetryInfo. No secrets or provider response text are retained. Native Gemini requests disable tool calling and sampling overrides, adapt only the wire schema, and retain strict local schema/citation checks.

Invalid model responses return sanitized HTTP 422 diagnostics; billing rejection returns 402; quota returns 429 with Retry-After; transport failures remain 502. These errors never promote an unreviewed repair or discard completed module work. Private code is not exported for diagnostics.

Offline regression tests cover cross-file requirements, quote grounding, dependency-aware cache invalidation, partial repairs, quota recovery, migration, approval and UI behavior. Read-only planning on the saved 187-requirement candidate yields seven groups without repeated requirements or fragments. A public synthetic Gemini check completed in 5.4 seconds, detected the planted email bug, and avoided ranking/dashboard/eventual-consistency false positives. No private full-project live review was performed by the agent for this verification.

### Final-group consistency recovery

Model-facing source_line_limits identify each supplied file's actual citation range. prepare_review removes unverifiable optional line numbers and changes met-without-source to unclear before semantic validation. It does not manufacture an assessment or infer a missing source path. Unknown, duplicate, or missing requirement IDs remain invalid. CriticError now carries fixed, sanitized field/category codes for these failures; a native Gemini consistency failure can make one corrective request for that group within the shared deadline. The retry never bypasses exact coverage checks. Existing saved group keys/fingerprints remain unchanged.

Groq GPT-OSS-120B receives a same-model JSON-object fallback after native json_validate_failed, within the existing total attempt allowance. Local schema and semantic validation remain mandatory, and oversized envelopes remain excluded. Public synthetic JSON-object verification completed in 2.0 seconds and detected the planted signed-integer bug; native schema generation was rejected. OpenRouter's GPT-OSS-120B free endpoint list was empty when checked on 2026-10-18; no unavailable or paid route was introduced.

Nemotron Ultra replaces Super for default draft and critic routes. Its active free endpoint rejects strict response-format requests, so critic calls omit that API parameter and include the complete strict schema in the system prompt. Returned JSON still passes the original local Pydantic schema and requirement/source validation before any progress is saved. Prompt-only reviews retain the NVIDIA 90-second timeout through an internal flag that is not sent to OpenRouter. Native low reasoning effort replaces numeric thinking limits. Ultra critic completions use the report estimate plus max(6,144, report estimate) tokens of reasoning headroom, bounded to 8,192-16,384 total tokens; other providers retain their existing limits. A real truncated batch with a 4,864-token report estimate and 3,613 thinking tokens now receives 11,008 total tokens rather than 5,888. Quota estimates include the additional reservation. This is output headroom, not a hard internal-thinking limit; the existing request/operation deadlines and truncated-response rejection still apply. On 2026-10-20, a public synthetic two-requirement review completed through this adapter in 13.3 seconds, passed validation, identified a planted arithmetic defect and accepted the correct function. Production project completion and speed are not established by that probe.

### Repair provider outages and Google usage

Repair counts actual inference attempts rather than truncating the candidate list before cooldown checks. Candidate order retains fitting Groq, then any explicit free override, verified North Mini Code free, the general free router, and the existing Gemini allowlist. Public free availability reuses the cached helper from phase2_coder/providers.py; ordinary Phase 2 routing is unchanged. At most three inference calls per file include all fallbacks and a possible one-second server-error retry after alternate candidates. Missing/cooled/free-unavailable routes consume no inference slots, and a final independent Google slot is reserved when configured.

Temporary HTTP 5xx responses are logged by class and numeric status, cool the route for 30 seconds, and can use an alternate configured model. HTTP 429 honors retry hints and stops further models on the same Google/OpenRouter key for that call. HTTP 401/403 also prevents redundant same-key attempts. A successful retry clears the model's failure cooldown. Native review can select the healthy Google model after repair cools its primary; one extra transient fallback shares the same allowance as a semantic correction.

ProviderUnavailableError crosses the coder boundary without being converted into empty code. HTTP 503 reports temporary provider unavailability with retryable=true; original authoritative drafts, completed staged repairs and the repair round remain intact. A later click resumes only unfinished work. The UI distinguishes this from invalid code or report failures. A public synthetic North Mini Code repair completed in 4.6 seconds, producing valid Python with the requested correction without consuming Gemini inference quota. Private project repair inference was not run by the agent for verification.
