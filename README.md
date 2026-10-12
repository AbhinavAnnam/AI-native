# AI-Native Requirements Extraction and Code Generation

AI-Native is a Python prototype for moving from software requirements documents to structured specifications and reviewed Python module drafts. A FastAPI service exposes the workflows, and a Streamlit application provides a control center for document upload, specification review, code generation, human approval, and execution logs.

The project currently has two related workflows:

- **Phase 1 — Requirements extraction:** upload one or more text-based PDF documents. The backend extracts their text, divides it into manageable chunks, asks configured language-model providers to extract requirements, consolidates the chunk results, validates the final object, and returns structured JSON.
- **Phase 2 — Code generation and review:** submit the structured specification to a code-generation workflow. Review and edit each generated module before approving it for writing to `generated_src/`. An additional LangGraph workflow supports an iterative generate, review, revise, and approve loop.

This repository is a development prototype. Model-generated requirements and source code should be reviewed before being used in a production system.

## Contents

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
│   ├── extractor.py               # PDF text extraction, chunking, map/reduce
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
5. The configured model provider returns JSON for each chunk. If a chunk fails, the pipeline stops and reports the failed chunk numbers instead of presenting a partial document as a successful extraction.
6. The chunk results are consolidated into one project specification. Related functionality is grouped into a small number of domain modules, repeated requirements are merged, and cross-cutting rules are placed in their own categories.
7. The final result is validated using the Pydantic `ProjectSpecs` schema and returned to the UI.

The response object contains:

- `project_title`: the project or system name inferred from the source.
- `features`: domain feature/module objects. Each feature has a human-readable name, a snake_case `target_module`, `business_rules`, `functions`, and `exceptions`.
- `cross_cutting_constraints`: categories of system-wide rules, each with a `category_name` and a list of actionable `rules`.

The schema is defined in `backend/models/schemas.py`. Extraction aims to reflect the uploaded source; generated details should be checked against the original requirements.

#### Extraction provider behavior

When `GEMINI_API_KEY` is set, Phase 1 tries Gemini models in configured priority order. If Gemini fails, the extraction layer can fall back to Groq and then OpenRouter when those keys are configured. Provider model defaults and optional single-model overrides are implemented in `phase1_rag/providers.py` and `phase1_rag/extractor.py`.

The default Groq extraction candidates are `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. The default OpenRouter candidates are `qwen/qwen3.5-35b-a3b` and `openai/gpt-oss-20b`. Setting `GROQ_EXTRACTION_MODEL` or `OPENROUTER_EXTRACTION_MODEL` replaces that provider's default candidate list with the specified model. The configured provider must support the JSON response format used by extraction.

### Phase 2: specification to reviewed code

The standard draft queue starts with `POST /api/phase2/generate-drafts`, which accepts a `ProjectSpecs` JSON object. The code-generation service creates Python source drafts for the specification's features, performs syntax validation, and adds the drafts to an in-memory human-review queue. The Streamlit UI displays the target module, feature name, validation message, and editable source code.

For each draft, a reviewer can:

- edit the proposed Python code and approve it;
- reject the draft; or
- leave it awaiting review.

Approval calls `POST /api/phase2/approve-and-commit`. The file tool parses the approved code with Python's AST parser before writing it. A module name such as `identity.accounts` is translated to `generated_src/identity/accounts.py`. Rejected or unapproved drafts are not committed through this review queue.

Starting a new standard draft-generation request clears and recreates the `generated_src/` directory. That behavior is implemented by the Phase 2 route before draft generation begins, so save any generated files that need to be retained before starting another draft-generation request.

### Agent Studio: iterative LangGraph workflow

The Agent Studio provides an alternate interface to generation and feedback. `POST /api/phase2/start` begins a thread using a caller-provided `thread_id` and a text `task_spec`. The graph classifies the task, chooses a model tier, creates drafts through `CoderAgent` and the MCP file tools, and pauses for human feedback. `POST /api/phase2/feedback` resumes the thread.

The task specification can be JSON with a `features` array (for example, the Phase 1 result) or free-form text. Free-form text is wrapped into a single generated-module feature. Use an approval response such as `approve`, `approved`, `looks good`, `lgtm`, `yes`, `ok`, or `okay` to approve; other feedback is treated as revision guidance. On approval, the workflow writes the reviewed drafts under `generated_src/`.

The graph uses an in-memory LangGraph checkpointer, and the draft review queue is also in memory. These are development conveniences: thread state and pending review items are not durable across application restarts.

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
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API root is `http://127.0.0.1:8000`. FastAPI's interactive Swagger UI is at `http://127.0.0.1:8000/docs`, and the alternative ReDoc page is at `http://127.0.0.1:8000/redoc`.

### 2. Start Streamlit

In another terminal:

```bash
streamlit run frontend/app.py
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

Approves and writes a draft, or rejects it. The approval payload includes `draft_id`, `target_module`, `approved_code`, and an `action` of `approve` or `reject`. For example:

```json
{
  "draft_id": "draft_123",
  "target_module": "account_management",
  "approved_code": "class AccountService:\n    pass\n",
  "action": "approve"
}
```

The API validates Python syntax before writing approved code. Invalid syntax produces an error response instead of a file write.

### Agent Studio routes

`POST /api/phase2/start` begins a graph execution. Supply a unique thread identifier and the task or specification as text:

```json
{
  "thread_id": "a-unique-thread-id",
  "task_spec": "{\"project_title\": \"Example Project\", \"features\": [...]}"
}
```

The response includes the thread ID, status, generated code bundle, and module validation summaries. Submit feedback to `POST /api/phase2/feedback`:

```json
{
  "thread_id": "a-unique-thread-id",
  "user_feedback": "Please add input validation, then regenerate."
}
```

An approval keyword advances the thread to finalization. Other text is sent as revision guidance. Keep using the same `thread_id` to continue a workflow; start a new thread for a separate task.

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
- `generated_src/` contains modules approved through the Phase 2 workflow or Agent Studio. Standard draft generation clears this directory before creating the new review batch.
- `.draft_staging/` is used to stage generated code during draft validation.
- The HITL review queue and LangGraph thread checkpoint are held in memory. Restarting the backend loses pending queue entries and agent thread state.
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
