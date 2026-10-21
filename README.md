# AI-Native Requirements Extraction and Code Generation

AI-Native is a Python prototype for moving from software requirements documents to structured specifications and reviewed Python module drafts. A FastAPI service exposes the workflows, and a Streamlit application provides a control center for document upload, specification review, code generation, human approval, and execution logs.

The project currently has four connected phases:

- **Phase 1 — Requirements extraction:** upload one or more text-based PDF documents. The backend extracts their text, divides it into manageable chunks, asks configured language-model providers to extract requirements, consolidates the chunk results, validates the final object, and returns structured JSON.
- **Phase 2 ? Code generation:** generate persisted drafts and request per-file or project revisions through LangGraph.
- **Phase 3 ? Critic and repair:** compare drafts with requirements, inspect findings, request targeted repairs, and approve accepted files into the local repository clone or `generated_src/`.
- **Phase 4 — Documentation and pull request:** preview a Markdown report of accepted changes, coverage and remaining findings, then explicitly create a GitHub draft PR. The tester phase is skipped.

This repository is a development prototype. Model-generated requirements and source code should be reviewed before being used in a production system.

## Contents

- [Plain-language workflow and file guide](docs/project-guide.md)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [How the workflows work](#how-the-workflows-work)
- [Requirements and setup](#requirements-and-setup)
- [Running the application](#running-the-application)
- [HTTP API](#http-api)
- [Configuration reference](#configuration-reference)
- [Generated files and data](#generated-files-and-data)
- [Limitations and troubleshooting](#limitations-and-troubleshooting)

## Architecture

```text
                 +---------------------------+
                 |  Streamlit control center |
                 |  frontend/app.py          |
                 +-------------+-------------+
                               | HTTP
                               v
                 +---------------------------+
                 | FastAPI application       |
                 | backend/main.py           |
                 +------+--------------------+
                        |
            +-----------+------------+
            |                        |
            v                        v
  +---------------------+  +-----------------------+
  | Phase 1 extraction  |  | Phase 2 code drafting|
  | PDF text, chunking, |  | CoderAgent, review,  |
  | provider fallback,  |  | approval, file tools |
  | schema validation   |  +-----------------------+
  +---------------------+             ^
                                       |
                          +------------+-----------+
                          | LangGraph Agent Studio |
                          | classify, draft, revise|
                          +------------------------+
```

The FastAPI routes are divided by workflow. Phase 1 is implemented in `backend/routers/phase1_router.py`, Phase 2 routes are in `backend/routers/phase2_router.py`, and log retrieval is in `backend/routers/logs_router.py`. Request and response models for the main payloads are defined in `backend/models/schemas.py`.

## Project structure

```text
AI-Native/
├── backend/
│   ├── main.py                    # FastAPI app, logging, CORS, and router setup
│   ├── deps.py                    # Shared Sentence Transformers and ChromaDB clients
│   ├── models/
│   │   └── schemas.py             # Phase 1 and Phase 2 Pydantic models
│   ├── routers/
│   │   ├── phase1_router.py       # POST /api/phase1/extract
│   │   ├── phase2_router.py       # Code draft and agent routes
│   │   └── logs_router.py         # GET /api/logs
│   └── agent/
│       ├── graph.py               # LangGraph generation and review loop
│       └── router.py              # Gemini model selection and retries
├── frontend/
│   └── app.py                     # Streamlit control center
├── phase1_rag/
│   ├── extractor.py               # PDF text extraction, typed chunk inference
│   ├── consolidation.py           # Lossless local merge and requirement checks
│   ├── grouping.py                # Compact module planning; requirements copied unchanged
│   ├── providers.py               # Gemini, Groq, and OpenRouter extraction calls
│   ├── parser.py                  # Directory parser for PDF/DOCX/Markdown/text
│   └── embedder.py                # Persistent ChromaDB document embedder utility
├── phase2_coder/
│   ├── coder.py                   # Module drafting and provider fallbacks
│   └── hitl_manager.py            # In-memory human review queue
├── mcp_tools/
│   └── file_tools.py              # Python syntax validation and module file tools
├── docs/                          # Example/source documents
├── artifacts/                     # Saved review artifacts, if present
├── requirements.txt
├── run_phase1.py                  # Earlier standalone workflow, not the API entry point
└── README.md
```

## How the workflows work

### Phase 1: PDF to structured specification

The main Phase 1 entry point is `POST /api/phase1/extract`.

1. The Streamlit user uploads one or more PDF files.
2. FastAPI receives the files as multipart form data and checks that at least one supported model-provider key is configured.
3. `phase1_rag/extractor.py` reads each PDF with `pypdf`, extracts text page by page, and combines the non-empty pages.
4. The document text is split into chunks (25,000 characters by default, with 1,500 characters of overlap). For each chunk, the extraction prompt asks for modules or services, business rules, function signatures, exceptions, and architectural constraints.
5. Every chunk requests the `ProjectSpecs` schema and is locally validated. Unknown fields, empty requirements, invalid JSON and truncated/refused model responses are rejected; Gemini can try the next configured model. If any chunk still fails, the pipeline reports its number instead of returning a partial document.
6. `phase1_rag/consolidation.py` merges validated chunks locally by identical module/category identifiers and removes exact duplicates. Different rule wording and function purposes remain visible. No requirement-bearing JSON key is deleted to fit a provider budget.
7. `phase1_rag/grouping.py` enforces five to seven code modules for larger plans. After local overlap merging, specifications with more than seven modules make one compact grouping request through the existing Phase 1 providers. The schema requires five to seven groups containing every feature ID exactly once. The model sees IDs, names and a few function signatures; it never rewrites the full requirements. Local code copies every rule, function/purpose pair, exception and cross-cutting constraint unchanged, with a fact-inventory check. Invalid/unavailable grouping or oversized descriptors use a deterministic local fallback that favors related names and balanced sizes, without extra model calls. Plans already containing seven or fewer modules stay intact; small documents are not padded to five. Package namespaces stay separate and each new group must fit the 18 KB feature-data budget. If those safeguards cannot be met within seven modules, extraction reports an explicit error instead of returning extra modules or dropping details.
8. The final object is validated and returned to the UI with visible module/rule/function counts. Unreadable uploaded documents fail explicitly rather than being skipped. Phase 2 uses the grouped specification for both repository and new-code runs. This reduces generation calls but does not change Phase 2 provider quotas or fallback routing.

The response object contains:

- `project_title`: the project or system name inferred from the source.
- `features`: domain feature/module objects. Each feature has a human-readable name, a snake_case `target_module`, `business_rules`, `functions`, and `exceptions`.
- `cross_cutting_constraints`: categories of system-wide rules, each with a `category_name` and a list of actionable `rules`.

The schema is defined in `backend/models/schemas.py` and rejects unexpected fields. The preservation check proves that consolidation kept the chunk results; it cannot prove the model extracted every requirement from the original PDF. Review the displayed specification against the source before generating code. A failed new extraction clears the previous UI specification so stale requirements cannot silently seed a new run.

#### Extraction provider behavior

When `GEMINI_API_KEY` is set, Phase 1 tries Gemini models in configured priority order. If Gemini fails, the extraction layer can fall back to Groq and then OpenRouter when those keys are configured. Provider model defaults and optional single-model overrides are implemented in `phase1_rag/providers.py` and `phase1_rag/extractor.py`.

The default Groq extraction candidates are `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. The default OpenRouter candidates are `qwen/qwen3.5-35b-a3b` and `openai/gpt-oss-20b`. Setting `GROQ_EXTRACTION_MODEL` or `OPENROUTER_EXTRACTION_MODEL` replaces that provider's default candidate list with the specified model. The configured provider must support the JSON response format used by extraction.

### Phase 2: specification to draft code

Use Agent Studio to generate and revise persisted drafts. Phase 2 displays separate files and accepts per-file or project feedback. Approval now lives in **Phase 3: Critic & Repair**. Draft and repair generation do not write unapproved code into the repository clone or `generated_src/`.

Draft routing lives in `phase2_coder/providers.py`: Groq `openai/gpt-oss-20b`, Groq `openai/gpt-oss-120b`, Gemini `gemini-3.5-flash-lite`, then OpenRouter `nvidia/nemotron-3-ultra-550b-a55b:free`. Additional free OpenRouter routes (`deepseek/deepseek-v4-flash:free`, `cohere/north-mini-code:free`, `openrouter/free`, or a free override) follow if budget remains. Skipped candidates do not spend inference attempts. The coder no longer enumerates every active Groq model or walks a long Gemini model list. Critic repair retains its separate existing routing.

Quota cooldowns are process-local and scoped to the provider, model and API key. They expire automatically: models can be used again without a backend restart. When multiple provider families are configured, quota waits yield immediately to the next fallback. Single-provider setups retain up to ten seconds of short quota waits per module. Daily quota retry hints are honored without waiting minutes inside the request. Timed-out models cool down for three minutes to avoid repeating a slow request on every following file; other temporary network/server failures cool down for 30 seconds and invalid/unavailable endpoints for five minutes. Free OpenRouter endpoint availability and zero token pricing are rechecked after 60 seconds. Only `:free` IDs or `openrouter/free` are accepted for OpenRouter draft overrides; a paid extraction-model setting cannot change this policy. Free-provider availability and account quotas still apply.

Each module has one shared attempt/time budget, SDK retries are disabled, and network requests have cancellable wall-clock timeouts. Groq input plus the actual output cap must fit the request budget; oversized full specifications use a fallback without dropping requirements. Truncated responses, invalid Python and empty implementations are rejected. Successful modules are returned even when another module fails; bounded failures do not restart the legacy router fallback, and a failed revision retains its previous candidate. Drafts are persisted when the graph generation node completes; this does not add per-file checkpoints during an interrupted node.

Groq requests default to a 15-second deadline and Nemotron to 60 seconds. The shared module budget always applies; earlier providers reserve capacity for a healthy Gemini request and one module-level syntax correction. Nemotron Ultra uses native low reasoning effort and adds 1,024 completion tokens of headroom to the existing 16,384-token code allowance; this is not a hard bound on internal reasoning. These limits bound waiting without shortening requirements or accepting truncated responses; successful live completion time and semantic correctness remain provider-dependent.

Optional Phase 2 settings (existing `.env` keys continue to work):

Phase 2 lists `missing_modules` in start, state and feedback responses. Use **Retry missing modules** on the current run to generate only missing/invalid drafts while retaining successful code, file paths and approval state. The button submits `Retry missing modules` through the existing feedback endpoint; it does not create another thread or clone. Repeating it when every draft is valid performs no inference.

A Python syntax failure receives at most one targeted correction per module, including the original requirements, the rejected candidate and a safe line/column diagnostic. The correction counts toward the same inference and time budget. Syntax, empty and incomplete response failures do not create model-wide outage cooldowns. Logs record fixed response categories or syntax exception types and positions, without source text or provider error messages. Quotas, authentication failures and real network/server failures retain their existing cooldowns. A provider skipped during cooldown is revisited once if it recovers during another fallback, without extending the deadline or waiting indefinitely.

| Setting | Default | Purpose |
| --- | --- | --- |
| `CODER_GROQ_MODELS` | `openai/gpt-oss-20b,openai/gpt-oss-120b` | Ordered Groq candidates before Nemotron and Gemini |
| `CODER_OPENROUTER_MODEL` | unset | Additional explicit `:free` route after Gemini, subject to active free endpoints |
| `CODER_MODEL_TIMEOUT_SECONDS` | `45` | Ordinary inference deadline; Groq also uses its smaller cap |
| `CODER_GROQ_TIMEOUT_SECONDS` | `15` | Groq request deadline, capped by the ordinary deadline |
| `CODER_NVIDIA_TIMEOUT_SECONDS` | `60` | Nemotron request deadline, capped at 120 seconds and remaining module budget |
| `CODER_MODULE_TIMEOUT_SECONDS` | `120` | Shared inference/time budget per module |
| `CODER_MAX_PROVIDER_ATTEMPTS` | `4` | Maximum inference calls per module, including short quota retries |
| `CODER_MAX_WAIT_SECONDS` | `10` | Total short quota wait allowance in single-provider setups |
| `CODER_GROQ_REQUEST_TOKEN_LIMIT` | `7500` | Conservative input plus output allowance; output capped at 4096 tokens |

The older `POST /api/phase2/generate-drafts` endpoint remains available for draft generation, but its in-memory queue is not the critic workflow. The old direct-save endpoint returns HTTP 409 and directs clients to Agent Studio; it cannot bypass critic review.

### Agent Studio: iterative LangGraph workflow

The Agent Studio provides an alternate interface to generation and feedback. `POST /api/phase2/start` begins a thread using a caller-provided `thread_id` and a text `task_spec`. The graph chooses a model tier from the number of features, creates drafts through `CoderAgent`, and pauses for human feedback. `POST /api/phase2/feedback` resumes the thread.

The task specification can be JSON with a `features` array (for example, the Phase 1 result) or free-form text. Free-form text is wrapped into a single `generated_module` feature. Approval is per file and requires `target_module`; other feedback is treated as revision guidance. Without a repository, approved files are written under `generated_src/`.

### Modify an existing GitHub repository

1. Upload the requirements PDF in Phase 1 and review its extracted specification.
2. In Agent Studio, enable **Modify an existing repository**.
3. Enter the GitHub HTTPS URL and optionally a base branch. A blank branch uses the remote default.
4. Leave **Advanced: choose target files** at its default `{}` to select files automatically. The workflow matches module names to existing Python source files and places unmatched modules alongside the source package. Review the resulting file paths before approval. If file selection is ambiguous, use the optional JSON override. For example, if the specification's module is `identity_and_organization`:

   ```json
   {"identity_and_organization": "src/organization_demo/organizations.py"}
   ```

5. Start the agent. Review each draft and request revisions. In **Phase 3: Critic & Repair**, run the critic, inspect findings, optionally repair, and approve files individually there.

Each run clones into `workspace/<run_id>/repo/`, records the baseline commit, and creates an `ai-native/<run_id>` branch. The coder receives existing code, the tracked file listing, and bounded context from Python files, README, project configuration, and root project instructions. Approval writes only the mapped file in the clone; files changed externally since loading are not overwritten. Existing `generated_src/` output is preserved in repository mode.

Automatic repository runs first use `backend/stages/planning.py` to group clearly overlapping domain names, such as organization settings with organization management or attachments with a comments-and-attachments domain. Generic words alone cannot trigger a merge; ambiguous parents and different package namespaces remain separate. Rules, function signatures/purposes and exceptions are preserved verbatim, removing only exact duplicates, and a fact inventory checks preservation. Cross-cutting requirements remain unchanged. Phase 2 shows the merged groups, and SQLite retains them alongside the original `task_spec` JSON and grouped `base_specs`. Explicit file overrides preserve the caller's module identities and bypass automatic grouping.

This slice supports GitHub HTTPS URLs, up to 50 planned Python modules, and repositories with at most 2,000 tracked files. The same module limit applies to automatic and explicit file mappings. Over-limit plans stop before cloning instead of truncating the specification. Automatic file selection uses names and package layout; it does not perform semantic repository-wide planning. Equal file matches, multiple source roots for a new module, or unresolved file collisions require an override or specification adjustment. Selected files are limited to 128 KB each; supplementary context has a 60 KB budget and a 16 KB per-file limit. Context outside those limits is omitted. Private repositories use the host's existing Git credential configuration; do not embed tokens in URLs. Git must be installed. Repository preparation does not initialize submodules, install dependencies, or execute project scripts.

Repository drafts currently receive syntax validation only; test execution is deferred to the upcoming tester stage and is reported as not run. The legacy generation path also reports zero collected tests as unverified. Saving approved files does not create a Git commit, push, or PR yet. Generated tests, documentation, and PR stages are the next increments. The critic performs static review and draft-only repairs; it does not execute repository code.

Stage responsibilities and extension points are described in [docs/architecture.md](docs/architecture.md).

The FastAPI application stores LangGraph checkpoints in `workspace/checkpoints.sqlite`. Pending Agent Studio reviews, candidate code, repository metadata, and per-file approval state survive backend restarts. Keep this database and repository workspaces to resume reviews. The separate standalone draft queue remains in memory. Existing reviews lost before checkpoint persistence was added cannot be recovered automatically; download any drafts still visible in the browser before starting a new thread.

## Critic review and repair

1. Generate drafts in Phase 2. No per-file approval button is shown there.
2. Open **Phase 3: Critic & Repair** and click **Run Critic**.
3. Inspect requirement coverage, file findings, severity, evidence, and limitations.
4. Click **Repair findings** for confirmed defects. Uncertain findings and optional improvements are not sent to the coder. The existing coder updates affected modules, then the critic reviews the new candidate automatically. Unclear items alone do not trigger speculative repairs; they may need clarification or later testing.
5. Inspect repaired code and its difference from the preceding candidate. **Approve all & continue** saves the current reviewed project and opens Phase 4. Optional per-file approval remains available. You may accept unresolved findings; the documentation discloses them. Valid Python syntax and a complete current review remain mandatory. Approval writes local files; publishing is a separate Phase 4 action.

The critic automatically uses the current Phase 2 drafts. No run ID or batch settings are needed. Optional **Recovery options** at the bottom can reopen a previous browser session using its run ID and existing SQLite checkpoints. Older saved runs can be reviewed and repaired without automatically overwriting their files. A successful repair reopens a completed run for approval.

Reports contain stable requirement IDs for business rules, functions, exceptions, and cross-cutting rules. Verdicts are `pass`, `changes_requested`, or `needs_review`. Python AST checks are deterministic; requirement judgments are LLM assessments, not execution evidence. Invalid/incomplete coverage and fabricated file or out-of-range line references are rejected. Source fragments cannot receive a passing verdict.

A full input fingerprint covers candidate source, specification, mappings, baseline source, relevant supplied disk context, and review-budget settings. Identical inputs reuse the checkpointed report without inference. Revisions or disk/context changes make it outdated; repair and approval recheck the fingerprint. Expected writes during per-file approval refresh the fingerprint without spending tokens on another review. Repository overwrite protection still applies after a new review.

The core operations are `backend.stages.critic.review_candidate(snapshot, cached, reviewer)` and `repair_candidate(state, snapshot, report, repairer)`. Contracts are in `backend/models/critic.py`; HTTP orchestration is in `backend/routers/critic_router.py`. There is no MCP implementation in this change; a future adapter can call these transport-independent operations.

### Critic API

| Endpoint | Request |
| --- | --- |
| `POST /api/critic/review` | `{"thread_id": "..."}` |
| `POST /api/critic/repair` | `{"thread_id": "...", "input_fingerprint": "<report fingerprint>"}` |
| `POST /api/critic/approve` | `{"thread_id": "...", "input_fingerprint": "<report fingerprint>", "target_module": "accounts"}` |
| `POST /api/critic/approve-all` | `{"thread_id": "...", "input_fingerprint": "<report fingerprint>"}` |

Successful responses include the existing workflow state plus `critic_report`, `critic_stale`, `critic_repair_rounds`, `critic_previous_drafts`, `critic_progress`, a `critic_pending_repair` flag, and `critic_repair_in_progress` for resumable generation. Read them without inference through `GET /api/phase2/state/{thread_id}`. Missing threads return 404; outdated reports, budget limits, and invalid reports return 409; provider transport failures return 502; HTTP 402 identifies provider credit or spending-budget rejections. Pure quota pauses return 429 with `Retry-After`, a safe explanation, and completed/total batch counts. Invalid model reports return 422 with sanitized field/category diagnostics and `retryable=false`; a later fallback quota cannot hide an earlier invalid report or billing failure. HTTP 402 reports credit/spending-budget limits without triggering automatic retries. OpenRouter may distinguish account credits, API-key limits and in-flight budgets; the application does not assume that every 402 requires a purchase. A failed repair preserves the previous candidate and consumes an attempt. Refresh state after a failed or timed-out repair before retrying. The optional `max_batches` field on review/repair requests (1?32) is an explicit cost budget; leaving it out preserves the default.

### Review quality, speed, and provider limits

`backend/stages/review_planning.py` builds bounded scopes using full module source and import/call dependencies. Module requirements are checked with their implementation; project-wide requirements are assessed separately, once rather than once per module. A whole-project source group is preferred for global rules when it fits. Larger scopes select relevant complete files and explicitly disclose omitted files. Oversized source is explicitly fragmented and cannot receive a passing verdict. No requirements are dropped.

Groups contain at most 64 requirements, with a default 128,000-byte input ceiling and 8,192-token report ceiling. Only a Groq-only installation subdivides groups to fit its smaller allowance. With independent fallback credentials, grouping is not forced to fit Groq. The saved 187-requirement candidate plans seven groups with no repeated requirements or fragmented source; this measures planning, not end-to-end model latency.

Findings carry `category` (`defect`, `uncertain`, or `suggestion`), exact `requirement_quote`, exact `code_quote`, and a concrete `counterexample`. Local checks downgrade defects without matching quotes or a counterexample. Missing/partial claims without a supported defect become unclear. Missing source is not proof of missing behavior. Unverified project-wide enforcement remains unclear when source is omitted. These checks improve traceability but cannot prove that an LLM's interpretation is correct. Static analysis is not a substitute for runtime tests.

The prompt explicitly distinguishes permitted behavior from required implementation: permitted eventual consistency does not demand an artificial delay, unspecified ranking does not require a ranking algorithm, and completed items already excluded from a dashboard cannot dominate it. A public synthetic live Gemini check completed in 5.4 seconds, detected an email-normalization defect, and accepted those three correct behaviors. This does not verify private project code.

**Run Critic** sends one POST and automatically displays saved progress through read-only GETs. **Repair findings** uses the same button for generation and automatic re-review, including recovery after an external pause. Up to two user-triggered repair rounds remain allowed. A short quota pause of at most 20 seconds can be resumed once within the same operation, without repeating successful work or consuming another repair round. Long quota waits and permanent errors are surfaced honestly rather than retried indefinitely.

Repairs use the existing coder only for affected files. A project-wide missing/uncertain item no longer fans out into repairs of every file. Unchanged independent review scopes are reused after repair; changed files, their detected dependents, and project-wide checks are re-reviewed. Candidate promotion still requires a complete re-review. Repair never writes repository source or runs project tests. **Approve & save** remains an explicit final action.

The existing keys (`groq_api_key`/`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`) remain unchanged. Review selection is in `phase1_rag/providers.py`, Gemini's native JSON adapter is `backend/agent/router.py::LLMRouter.review_json`, and repair providers are in `backend/stages/repair.py`. Phase 1 and ordinary Phase 2 routing are unchanged. OpenRouter critic/repair routes must be explicitly free; paid overrides are rejected. Google/Groq account billing tiers still apply.

Groq is considered only when the complete envelope fits. OpenRouter reviews prefer `nvidia/nemotron-3-ultra-550b-a55b:free` or an explicit free override. For oversized Groq requests with Gemini configured, one preferred OpenRouter attempt precedes Gemini, avoiding a second slow generic-router attempt. Without that independent fallback, bounded `openrouter/free` fallback remains available. Repairs try the existing Groq route when it fits, then a configured free OpenRouter override, verified `cohere/north-mini-code:free`, the generic free router, and configured Gemini candidates within one shared attempt budget. A timed-out provider is cooled down for ten minutes across review calls, repairs, and button clicks within the backend process; truncated/invalid/unavailable routes cool down for five minutes. Expired models become eligible again. Cooldowns reset on backend restart. Quota retry hints are honored separately. Temporary HTTP 5xx errors use a shorter 30-second cooldown.

Native Gemini JSON requests adapt its supported schema subset and disable tool calling. Strict local Pydantic, coverage, file-path and line checks remain mandatory. Gemini reserves up to 2,048 extra completion tokens within an 8,192-token ceiling. Nemotron Ultra uses native low reasoning effort and a separate total completion allowance: the report estimate plus at least 6,144 tokens of reasoning headroom (or another full report allowance for larger reports), with an 8,192-token floor and 16,384-token ceiling. The observed 4,864-token report therefore receives 11,008 total tokens instead of 5,888. Internal reasoning is not numerically capped; this allowance leaves room for it without reducing the report estimate. Token estimates include the larger reservation, and logs show report/headroom/total allowances. Timeout, attempt, and operation deadlines remain unchanged. Its active free endpoint does not support response_format, so the complete JSON schema is included in the system prompt instead. Strict local Pydantic, coverage, citation and finish-reason validation still applies, and malformed reports use the existing bounded fallbacks. Ordinary requests have a 45-second cancellable timeout; Nemotron JSON reviews retain their dedicated 90-second allowance. A public two-requirement Ultra probe through the updated adapter completed in 13.3 seconds, passed schema and semantic validation, detected the planted arithmetic defect, and marked the correct function as met. A separate synthetic 29 KB repair completed in 55.5 seconds; larger-request latency and availability remain provider-dependent.

Repair recovery counts actual inference requests, not unavailable or cooled candidates. At most three calls are allowed per file, including an optional one-second retry of a server-failing route after alternatives. Verified North Mini Code is attempted before Gemini to reduce dependence on the Google key; cached public endpoint metadata must show zero prompt/completion pricing. The primary Flash Lite model can fall back to the existing alternate Flash Lite model for temporary server errors. A Google quota error stops further Google attempts for that operation; changing models is not treated as a quota bypass. Review also uses a healthy configured Google fallback when its primary is cooling, with at most one extra server-error fallback or report correction.

Exhausted temporary server/transport failures now propagate as a typed provider-unavailable error through the coder and return HTTP 503 with saved work preserved, rather than becoming empty generated code and HTTP 409. The same Repair findings button resumes the pending round and skips completed files. A synthetic free North Mini Code repair completed in 4.6 seconds with valid corrected Python and no Gemini request. This small probe does not establish full-project repair quality or eliminate provider quotas.

| Setting | Default | Meaning |
| --- | --- | --- |
| `CRITIC_MAX_INPUT_BYTES` | `128000` | Serialized input budget with schema/prompt reserve; not an exact tokenizer count |
| `CRITIC_MAX_OUTPUT_TOKENS` | `8192` | Nominal report ceiling; smaller groups request less |
| `CRITIC_MAX_BATCHES` | `32` | Maximum review groups; API callers may request a smaller limit |
| `CRITIC_MODEL_TIMEOUT_SECONDS` | `45` | Ordinary provider deadline |
| `CRITIC_NVIDIA_TIMEOUT_SECONDS` | `90` | Nemotron JSON deadline, capped at 120 seconds and remaining operation time |
| `CRITIC_OPERATION_TIMEOUT_SECONDS` | `600` | Shared repair/review/recovery budget per click, capped at 660 seconds |
| `CRITIC_REVIEW_TIMEOUT_SECONDS` | `600` | Review-stage ceiling, also capped by the shared operation budget |
| `CRITIC_REPAIR_TIMEOUT_SECONDS` | `600` | Repair-stage ceiling, also capped by the shared operation budget |
| `CRITIC_MAX_WAIT_SECONDS` | `20` | Cumulative provider quota wait per group |
| `CRITIC_GROQ_REQUEST_TOKEN_LIMIT` | `7500` | Full Groq envelope allowance, configurable downward |
| `CRITIC_OPENROUTER_MODEL` | Nemotron `:free` for reviews; `openrouter/free` for repairs | Optional preferred free route |
| `CRITIC_GEMINI_MODEL` | `gemini-3.5-flash-lite` | Allowlisted Gemini override |

Cross-file JSON reviews include per-file line limits on the model-facing wire. An optional line number outside its cited file is removed rather than presented as evidence; a met assessment with no source citation becomes unclear. Unknown/duplicate/missing requirement assessments remain errors. Semantic consistency failures now expose sanitized field/category diagnostics and permit one corrective Gemini request for that failed group within the existing operation deadline. Completed groups are not regenerated. This citation correction does not change saved batch keys or the assessment-policy fingerprint.

GPT-OSS-120B remains first on Groq when the complete request fits the conservative 7,500-token guard. A native json_validate_failed response can retry the same model once in JSON-object mode within the existing attempt budget, with unchanged strict local validation. A public two-requirement probe detected its planted bug in 2.0 seconds using that mode; its native-schema probe was rejected. This does not imply it can accept the current large project groups. Public OpenRouter metadata on 2026-10-18 listed no active endpoints for openai/gpt-oss-120b:free, so no unavailable or paid route is silently substituted.

SQLite retains authoritative drafts, validated batch results, and staged repairs. The new assessment policy (`critic-v4`) intentionally invalidates earlier judgments, so an existing run needs one fresh review, not new extraction or generation. Previously staged repaired files are preserved and reviewed under the new policy before promotion. Source/specification/disk changes invalidate incompatible evidence. No full report is available until all required groups finish. Free-provider limits can still pause an operation; the application does not guarantee completion time or zero defects.

## Documentation and draft pull request

1. Finish the critic review and any repairs. Click **Approve all & continue** in Phase 3.
2. In **Phase 4: Documentation & Pull Request**, click **Generate documentation**.
3. Preview or download `AI_NATIVE_REVIEW_<run>.md`. It contains baseline file changes, added/removed/changed Python symbols, full requirement coverage, remaining findings and limitations. It explicitly states that automated testing was skipped. Documentation uses saved validated data, without another model call.
4. For repository runs, enter a PR title and click **Create draft PR**. This writes the report into the clone, commits only mapped accepted Python files plus the report, pushes the run's `ai-native/<run_id>` branch, and opens a **draft** GitHub PR. It does not merge the PR. Non-repository runs support documentation download only.

Publishing requires a GitHub personal access token in your local `.env`:

```dotenv
GITHUB_TOKEN=your_github_token
```

`GH_TOKEN` is also supported. For a fine-grained token, select the target repository and give **Contents: Read and write** and **Pull requests: Read and write** permissions. The token owner must have permission to push to that repository; forks are not created automatically. Existing Gemini/Groq/OpenRouter keys cannot authenticate GitHub. Restart the backend after adding the token. Never commit it. GitHub documents token permissions in its [pull request API reference](https://docs.github.com/en/rest/pulls/pulls).

Git also needs your commit identity. If it is not already configured, run these commands using your own details before publishing:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

The report preview, publishing checkpoint, and PR result are persisted in SQLite. Report generation does not write into the target repository until publishing. A failed push/API call keeps the local commit; retry **Create draft PR** on the same run. Existing matching draft PRs are reused to prevent duplicates. Files changed outside the reviewed candidate, unrelated staged/untracked changes, changed branches/origins, or manually edited reports stop publishing with an actionable error. Source revisions require a fresh critic review, approval, and documentation.

| Endpoint | Validated request | Result |
| --- | --- | --- |
| `POST /api/delivery/documentation` | `thread_id`, current critic `input_fingerprint` | `DeliveryView` with Markdown and typed file changes |
| `POST /api/delivery/pull-request` | Same fields plus previewed `document_id` and optional `title` | `DeliveryView` with draft PR number, URL, branches and commit |
| `GET /api/phase2/state/{thread_id}` | Existing run | Includes `documentation`, `documentation_stale`, and `pull_request` |

Contracts reject unknown request fields. All files must be accepted and match the current complete critic report. Pending repairs cannot be published. Stale documentation is rejected server-side, even if an older browser still shows the button. Only explicit publishing invokes GitHub write operations; regression tests use mocked GitHub requests. This local prototype does not implement a hosted multi-user publishing service.

## Requirements and setup

### Prerequisites

- Python 3.10 or newer
- Model API credentials for the workflow being used
- Network access to the configured model providers
- Enough disk space and network access for the Sentence Transformers model, which may be downloaded when first loaded

Phase 1's API route initializes the `BAAI/bge-small-en-v1.5` Sentence Transformers embedder and a ChromaDB client. It indexes chunks during the extraction request. The current upload extractor sends document chunks to the extraction models directly; it does not perform a separate retrieval query as part of that endpoint.

### Create an environment and install dependencies

Run these commands from the repository root.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Configure API keys

Create `.env` in the repository root. Put real secret values in your local environment file; do not commit API keys.

```env
# Phase 1: configure one or more extraction providers.
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional Phase 1 model overrides; each makes that provider try one model.
GROQ_EXTRACTION_MODEL=openai/gpt-oss-20b
OPENROUTER_EXTRACTION_MODEL=qwen/qwen3.5-35b-a3b
```

Credential requirements vary by workflow:

| Workflow | Required credentials |
| --- | --- |
| Phase 1 PDF extraction | At least one of `GEMINI_API_KEY`, `GROQ_API_KEY`, or `OPENROUTER_API_KEY` |
| Phase 2 standard draft generation | `GEMINI_API_KEY` and at least one of `GROQ_API_KEY` or `OPENROUTER_API_KEY` |
| Agent Studio | `GEMINI_API_KEY`; the CoderAgent draft path also uses the Phase 2 provider setup |

The Phase 2 CoderAgent uses Groq as its primary generation tier and Gemini as a fallback. OpenRouter is initialized when configured. The LangGraph classifier and model router use Gemini. Restart the backend after editing `.env`, since provider clients and the router are initialized by the application.

## Running the application

Run the backend and frontend in separate terminals from the repository root with the virtual environment active.

### 1. Start FastAPI

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The API root is `http://127.0.0.1:8000`. FastAPI's interactive Swagger UI is at `http://127.0.0.1:8000/docs`, and the alternative ReDoc page is at `http://127.0.0.1:8000/redoc`. Run without `--reload` during coding workflows: cloning repositories and writing generated Python files under the workspace can otherwise interrupt requests by triggering backend restarts. Completed checkpoints survive restarts. Restart the backend manually after changing application code.

### 2. Start Streamlit

In another terminal:

```bash
python -m streamlit run frontend/app.py --server.port 8501
```

The Streamlit page connects to the backend at `http://127.0.0.1:8000`. Start the backend first. The UI has a Phase 1 requirements tab, a Phase 2 code drafting and HITL tab, an Agent Studio, and a panel for execution logs.

## HTTP API

### `POST /api/phase1/extract`

Extracts a project specification from one or more uploaded PDFs. The request uses `multipart/form-data`; every document uses the form field name `files`.

```bash
curl -X POST http://127.0.0.1:8000/api/phase1/extract \
  -F "files=@requirements.pdf;type=application/pdf"
```

Successful response shape:

```json
{
  "status": "success",
  "specs": {
    "project_title": "Example Project",
    "features": [
      {
        "feature_name": "Accounts",
        "target_module": "account_management",
        "business_rules": ["Accounts must use unique email addresses."],
        "functions": [
          {
            "signature": "def create_account(email: str) -> Account",
            "purpose": "Create an account for a user."
          }
        ],
        "exceptions": ["DuplicateAccountError"]
      }
    ],
    "cross_cutting_constraints": [
      {
        "category_name": "security_and_authorization",
        "rules": ["Only authorized users may access protected account data."]
      }
    ]
  }
}
```

The actual extracted content depends on the uploaded documents. A missing provider key, unreadable PDF, provider failure, or failed document chunk is returned as an HTTP error with a `detail` message.

### `POST /api/phase2/generate-drafts`

Accepts the `specs` object from Phase 1 as the JSON request body. It returns `pending_reviews`, containing each draft's identifier, feature, target module, generated code, syntax validation result, and review status.

Example request body:

```json
{
  "project_title": "Example Project",
  "features": [
    {
      "feature_name": "Accounts",
      "target_module": "account_management",
      "business_rules": ["Accounts must use unique email addresses."],
      "functions": [
        {
          "signature": "def create_account(email: str) -> Account",
          "purpose": "Create an account for a user."
        }
      ],
      "exceptions": ["DuplicateAccountError"]
    }
  ],
  "cross_cutting_constraints": []
}
```

### `POST /api/phase2/approve-and-commit`

This legacy direct-save endpoint returns HTTP 409. Use the critic workflow below. The older `/api/phase2/feedback` approval action also enforces a current critic report.

### Agent Studio routes

`POST /api/phase2/start` begins a graph execution. Supply a unique thread identifier and the task or specification as text:

```json
{
  "thread_id": "a-unique-thread-id",
  "task_spec": "{\"project_title\": \"Example Project\", \"features\": [...]}"
}
```

The response includes the thread ID, status, generated code bundle, and module validation summaries. Submit feedback to `POST /api/phase2/feedback`:

To use repository mode, add this optional `repository` object to the start payload. Only `url` is required; omit `file_mappings` or use `{}` for automatic selection. When overriding selection, mapping keys must match all features in `task_spec`:

```json
{
  "url": "https://github.com/AbhinavAnnam/organization-demo.git",
  "base_branch": "main",
  "file_mappings": {
    "identity_and_organization": "src/organization_demo/organizations.py"
  }
}
```

Start/state/feedback responses include a repository summary with `run_id`, `repo_dir`, `base_branch`, `baseline_commit`, `work_branch`, and `file_mappings`, plus a `candidate_version` hash identifying the current specification and draft code. Preparation failures return `status: "repository_failed"` and an `error`; no generation runs. Correct the inputs and use a new thread. Changed module sets require a new thread and new mappings.

Submit revision feedback using the same thread:

```json
{
  "thread_id": "a-unique-thread-id",
  "user_feedback": "Please add input validation, then regenerate."
}
```

An approval keyword with `target_module` saves that file only after a current critic review. Other text is sent as revision guidance; supplying `target_module` limits revision to that file. Keep using the same `thread_id` to continue a workflow; start a new thread for a separate task.

### `GET /api/logs`

Returns the contents of `pipeline.log` as a JSON `logs` field. The Phase 1 UI refreshes logs after processing; the logs panel can also be refreshed manually.

For the canonical request models and interactive examples, open `http://127.0.0.1:8000/docs` while the backend is running.

## Configuration reference

| Variable | Used by | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | Phase 1, Phase 2, Agent Studio | Gemini extraction/fallback, task classification, and agent model calls |
| `GROQ_API_KEY` | Phase 1 fallback, Phase 2 CoderAgent | Groq extraction fallback and code drafting |
| `OPENROUTER_API_KEY` | Phase 1 fallback, Phase 2 CoderAgent | OpenRouter extraction fallback and optional code provider |
| `GROQ_EXTRACTION_MODEL` | Phase 1 | Replace Groq's extraction model candidates with one model |
| `OPENROUTER_EXTRACTION_MODEL` | Phase 1 | Replace OpenRouter's extraction model candidates with one model |
| `GROQ_MAX_TOKENS` | Phase 2 CoderAgent | Groq generation token limit; defaults to `8192` |
| `GEMINI_MAX_OUTPUT_TOKENS` | Phase 2 CoderAgent | Gemini generation token limit; defaults to `32768` |

## Generated files and data

- `pipeline.log` is appended to by the backend and is returned by `GET /api/logs`.
- `generated_src/` contains non-repository modules accepted through the critic approval workflow. Standard draft generation clears this directory before creating the new review batch.
- `.draft_staging/` is used to stage generated code during draft validation.
- Agent Studio checkpoints are stored in `workspace/checkpoints.sqlite`, including pending drafts and approval state. This local database is ignored by Git. The standalone HITL queue remains in memory and is lost on restart.
- `chroma_db_docs/` is the default path for the standalone `DocumentEmbedder` in `phase1_rag/embedder.py`. The FastAPI dependency in `backend/deps.py` currently uses `chromadb.Client()` rather than that persistent embedder path.
- The FastAPI extraction endpoint returns specs in its response; it does not save `extracted_specs.json` by default. The checked-in `artifacts/` folder may contain manually saved review outputs.

## Limitations and troubleshooting

### Supported upload formats

The Streamlit uploader and Phase 1 API currently accept PDF files. The PDF parser extracts selectable text with `pypdf`; it does not perform OCR. Scanned or image-only PDFs may therefore produce an empty-text error. Although `phase1_rag/parser.py` can parse PDF, DOCX, Markdown, and plain-text files from a directory, this is not the parser used by the current upload route.

### Provider keys or model errors

Check that `.env` is in the repository root, the key names match the configuration table, and the backend was restarted after the file changed. Provider errors and failed chunk numbers are recorded in `pipeline.log` and can be read from `GET /api/logs`. A configured model override may need to support the JSON response format or strict schema mode used by the provider client.

### Phase 2 credentials

The standard draft-generation endpoint checks for both `GEMINI_API_KEY` and at least one of `GROQ_API_KEY` or `OPENROUTER_API_KEY`. Phase 1 succeeding with a single provider key does not guarantee that Phase 2 has all of its required credentials.

### Frontend cannot reach the API

Start FastAPI on port `8000` before launching Streamlit. The current UI uses the fixed base URL `http://127.0.0.1:8000`; change that URL in `frontend/app.py` if your backend runs elsewhere.

### Generated module fails validation

Drafts are checked for Python syntax before commit. A syntax error is reported in the review queue, where the source can be edited before approval. Syntax validation does not establish that the generated code is correct, secure, or complete; review its behavior and requirements coverage.

### Earlier standalone runner

`run_phase1.py` is an earlier prototype script, not the supported application entry point. It references a `parse_documents` interface that is not present in the current `phase1_rag/parser.py`. Use FastAPI and Streamlit as described above.

### Local service security

The FastAPI app currently allows cross-origin requests broadly and does not provide user authentication. Run it as a local development service; add appropriate access controls and deployment configuration before exposing it beyond a trusted local environment.
