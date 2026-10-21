import json
import difflib
import uuid
from concurrent.futures import ThreadPoolExecutor, wait

import streamlit as st
import requests

st.set_page_config(page_title="AI-Native Control Center", layout="wide")
st.title("⚙️ AI-Native Core Control Center")

if "execution_logs" not in st.session_state:
    st.session_state.execution_logs = "No system logs fetched."
if "extracted_specs" not in st.session_state:
    st.session_state.extracted_specs = None
if "pending_reviews" not in st.session_state:
    st.session_state.pending_reviews = []
if "agent_thread_id" not in st.session_state:
    st.session_state.agent_thread_id = str(uuid.uuid4())
if "agent_status" not in st.session_state:
    st.session_state.agent_status = None
if "agent_output" not in st.session_state:
    st.session_state.agent_output = ""
if "agent_completed" not in st.session_state:
    st.session_state.agent_completed = False
if "agent_modules" not in st.session_state:
    st.session_state.agent_modules = []
if "agent_repository" not in st.session_state:
    st.session_state.agent_repository = None
if "agent_error" not in st.session_state:
    st.session_state.agent_error = None
if "agent_module_groups" not in st.session_state:
    st.session_state.agent_module_groups = {}
if "agent_missing_modules" not in st.session_state:
    st.session_state.agent_missing_modules = []
if "agent_task_spec" not in st.session_state:
    st.session_state.agent_task_spec = ""
if "agent_task_seed" not in st.session_state:
    st.session_state.agent_task_seed = None
for key, default in {"documentation": None, "documentation_stale": False, "pull_request": None,
                     "critic_report": None, "critic_stale": False,
                     "critic_repair_rounds": 0, "critic_previous_drafts": [],
                     "critic_progress": {}, "critic_pending_repair": False, "critic_repair_in_progress": False}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def fetch_logs():
    try:
        res = requests.get("http://127.0.0.1:8000/api/logs", timeout=5)
        if res.status_code == 200:
            st.session_state.execution_logs = res.json().get("logs", "No logs.")
    except Exception as e:
        st.session_state.execution_logs = f"Log fetch failed: {str(e)}"


def update_agent_review(data):
    st.session_state.agent_status = data.get("status")
    st.session_state.agent_output = data.get("latest_output") or data.get("generated_output") or ""
    st.session_state.agent_modules = data.get("modules") or []
    st.session_state.agent_completed = bool(data.get("is_completed"))
    st.session_state.agent_repository = data.get("repository")
    st.session_state.agent_error = data.get("error")
    st.session_state.agent_module_groups = data.get("module_groups") or {}
    st.session_state.agent_missing_modules = data.get("missing_modules") or []
    for key, default in {"documentation": None, "documentation_stale": False, "pull_request": None,
                         "critic_report": None, "critic_stale": False,
                         "critic_repair_rounds": 0, "critic_previous_drafts": [],
                         "critic_progress": {}, "critic_pending_repair": False, "critic_repair_in_progress": False}.items():
        st.session_state[key] = data.get(key, default)


def critic_action(action, target_module=None):
    st.session_state.active_phase = "Phase 3: Critic & Repair"
    payload = {"thread_id": st.session_state.agent_thread_id}
    if action != "review":
        payload["input_fingerprint"] = st.session_state.critic_report["input_fingerprint"]
    if target_module is not None:
        payload["target_module"] = target_module
    with st.spinner({"review": "Reviewing requirements and code...", "repair": "Repairing affected modules and reviewing again...", "approve": "Saving reviewed file...", "approve-all": "Saving the reviewed project..."}[action]):
        try:
            if action in {"approve", "approve-all"}:
                response = requests.post(f"http://127.0.0.1:8000/api/critic/{action}", json=payload, timeout=900)
            else:
                # One POST performs the operation; GETs only display saved progress.
                display = st.empty()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(requests.post, f"http://127.0.0.1:8000/api/critic/{action}", json=payload, timeout=900)
                    while not wait([future], timeout=2).done:
                        try:
                            current = requests.get(f"http://127.0.0.1:8000/api/phase2/state/{payload['thread_id']}", timeout=5)
                            if current.status_code == 200:
                                value = current.json()
                                progress = value.get("critic_progress") or {}
                                if value.get("critic_repair_in_progress"):
                                    display.info("Repairing confirmed defects. Completed files are being saved.")
                                elif progress.get("total"):
                                    display.progress(min(1., progress.get("completed", 0) / progress["total"]),
                                        text=f"Checking requirements: {progress.get('completed', 0)}/{progress['total']} review groups complete")
                        except requests.RequestException:
                            pass
                    response = future.result()
                display.empty()
            if response.status_code == 200:
                update_agent_review(response.json())
                if action == "approve-all":
                    st.session_state.active_phase = "Phase 4: Documentation & Pull Request"
            elif response.status_code == 404:
                mark_review_missing()
            else:
                if response.status_code == 429:
                    detail = response.json().get("detail", {})
                    delay = detail.get("retry_after", 60) if isinstance(detail, dict) else 60
                    st.warning(f"The model providers are currently rate-limited. Your code and completed review work are saved. Try Run Critic again in about {delay} seconds.")
                elif response.status_code == 503:
                    detail = response.json().get("detail", {})
                    st.warning(detail.get("message", "Providers are temporarily unavailable. Completed repairs are saved.") if isinstance(detail, dict) else str(detail))
                elif response.status_code in (402, 422):
                    detail = response.json().get("detail", {})
                    st.error(detail.get("message", "The critic returned an invalid report. Completed work is saved.") if isinstance(detail, dict) else str(detail))
                else:
                    st.error(f"Critic request failed ({response.status_code}): {response.text}")
                # A failed repair still consumes its attempt. Refresh metadata without rerunning models.
                current = requests.get(f"http://127.0.0.1:8000/api/phase2/state/{st.session_state.agent_thread_id}", timeout=30)
                if current.status_code == 200:
                    update_agent_review(current.json())
        except requests.RequestException as exc:
            st.error(f"Critic request failed: {exc}. Your drafts are preserved. Run Critic again to continue.")


def refresh_critic():
    st.session_state.active_phase = "Phase 3: Critic & Repair"
    try:
        response = requests.get(f"http://127.0.0.1:8000/api/phase2/state/{st.session_state.agent_thread_id}", timeout=30)
        if response.status_code == 200:
            update_agent_review(response.json())
        else:
            st.error(f"Could not refresh review: {response.text}")
    except requests.RequestException as exc:
        st.error(f"Could not refresh review: {exc}")


def delivery_action(action):
    st.session_state.active_phase = "Phase 4: Documentation & Pull Request"
    report = st.session_state.critic_report or {}
    payload = {"thread_id": st.session_state.agent_thread_id,
               "input_fingerprint": report.get("input_fingerprint", "")}
    if action == "pull-request":
        payload.update(document_id=st.session_state.documentation["document_id"],
                       title=st.session_state.get("delivery_title", "Implement reviewed requirements"))
    with st.spinner("Generating the change report..." if action == "documentation" else "Committing accepted files, pushing the working branch, and opening a draft PR..."):
        try:
            response = requests.post(f"http://127.0.0.1:8000/api/delivery/{action}", json=payload, timeout=360)
            if response.status_code == 200:
                for key, value in response.json().items():
                    st.session_state[key] = value
            else:
                detail = response.json().get("detail", "Delivery failed. Completed work is preserved.")
                st.error(str(detail))
        except (requests.RequestException, ValueError):
            st.error("Could not complete the delivery request. Saved work is preserved; retry the same action.")


def mark_review_missing():
    st.session_state.agent_status = "review_missing"
    st.session_state.agent_error = (
        "This review is unavailable on the backend. Your visible drafts are preserved; "
        "download them before starting a New Thread."
    )


def submit_agent_feedback(feedback, target_module=None, rerun=True):
    payload = {"thread_id": st.session_state.agent_thread_id, "user_feedback": feedback}
    if target_module is not None:
        payload["target_module"] = target_module
    operation = "Saving" if feedback.strip().lower() == "approve" else "Updating"
    label = "Retrying missing module drafts..." if feedback.strip().lower() == "retry missing modules" else (
        f"{operation} {target_module}..." if target_module else "Applying project feedback...")
    with st.spinner(label):
        try:
            response = requests.post("http://127.0.0.1:8000/api/phase2/feedback", json=payload, timeout=900)
            if response.status_code == 200:
                update_agent_review(response.json())
                if rerun:
                    st.rerun()
            elif response.status_code == 404:
                mark_review_missing()
                if rerun:
                    st.rerun()
            else:
                st.error(f"Feedback error ({response.status_code}): {response.text}")
        except Exception as exc:
            st.error(f"Feedback request failed: {exc}")
        finally:
            fetch_logs()


col_main, col_logs = st.columns([2, 1])
if "requested_phase" in st.session_state:
    st.session_state.active_phase = st.session_state.pop("requested_phase")

with col_main:
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Phase 1: Requirements", "💻 Phase 2: Coder Agent & HITL", "Phase 3: Critic & Repair", "Phase 4: Documentation & Pull Request"], key="active_phase", on_change="rerun")

    # ------------------------------------------------------------------
    # TAB 1: Phase 1
    # ------------------------------------------------------------------
    with tab1:
        st.subheader("Document Processing & Requirements Extraction")
        uploaded_files = st.file_uploader("Upload Technical PRD PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("🚀 Process Documents & Extract Specs", type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file.")
            else:
                # A failed new extraction must not leave an older JSON ready for a new run.
                st.session_state.extracted_specs = None
                with st.spinner("Executing Phase 1 extraction..."):
                    files_payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                    try:
                        resp = requests.post("http://127.0.0.1:8000/api/phase1/extract", files=files_payload, timeout=180)
                        if resp.status_code == 200:
                            st.session_state.extracted_specs = resp.json().get("specs")
                            st.success("Extracted specifications successfully!")
                        else:
                            st.error(f"API Error ({resp.status_code}): {resp.text}")
                    except Exception as e:
                        st.error(f"Backend Request Failed: {e}")
                    finally:
                        fetch_logs()

        if st.session_state.extracted_specs:
            st.subheader("Extracted Specifications JSON")
            extracted = st.session_state.extracted_specs
            features = extracted.get("features", [])
            st.caption(f"{len(features)} modules, "
                       f"{sum(len(f.get('business_rules', [])) for f in features)} business rules, "
                       f"{sum(len(f.get('functions', [])) for f in features)} functions, "
                       f"{sum(len(c.get('rules', [])) for c in extracted.get('cross_cutting_constraints', []))} cross-cutting rules. "
                       "Review these against your document before generating code.")
            st.json(st.session_state.extracted_specs)

    # ------------------------------------------------------------------
    # TAB 2: Phase 2 (HITL)
    # ------------------------------------------------------------------
    with tab2:
        st.subheader("Coder Agent — Draft Generation & Revision")

        # ------------------------------------------------------------------
        # Agent Studio (LangGraph Human-in-the-Loop Stream)
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🤖 Agent Studio — LangGraph Human-in-the-Loop")
        st.caption(
            "Review each draft file below before approval. Use a file's feedback box to update only that file, "
            "or use project feedback to revise all files. Updated drafts appear here for another review. "
            "Run the critic in Phase 3, review its findings, and approve files there. "
            "Generated and repaired code remains draft until approval. "
            "New-code runs clear generated_src/. Repository runs use a separate clone and working branch."
        )

        st.caption(f"Thread ID (for resuming): {st.session_state.agent_thread_id}")

        # Seed the text area from the extracted specs. A widget with a fixed `key`
        # ignores `value=` on every later rerun, which previously meant the agent
        # received an EMPTY spec (and synthesised a generic `generated_module`).
        spec_seed = (
            json.dumps(st.session_state.extracted_specs, indent=2)
            if st.session_state.extracted_specs
            else ""
        )
        if st.session_state.get("agent_task_seed") != spec_seed:
            st.session_state.agent_task_seed = spec_seed
            st.session_state.agent_task_spec = spec_seed

        task_spec = st.text_area(
            "Task specification for the agent (auto-filled from Phase 1 extraction)",
            height=160,
            key="agent_task_spec",
        )

        use_repository = st.checkbox(
            "Modify an existing repository", key="agent_use_repository",
            disabled=st.session_state.agent_status is not None,
        )
        repository_url, repository_branch, mapping_text = "", "", "{}"
        if use_repository:
            repository_url = st.text_input(
                "GitHub repository URL", key="agent_repository_url",
                placeholder="https://github.com/owner/repository.git",
                disabled=st.session_state.agent_status is not None,
            )
            repository_branch = st.text_input(
                "Base branch (optional)", key="agent_repository_branch",
                help="Leave blank to use the repository's default branch.",
                disabled=st.session_state.agent_status is not None,
            )
            st.caption("Files are selected automatically from the repository. Review the proposed file paths before approving changes.")
            with st.expander("Advanced: choose target files (optional)"):
                mapping_text = st.text_area(
                    "Module-to-file mappings (JSON)", value="{}", key="agent_file_mappings",
                    placeholder='{"organization_demo.organizations": "src/organization_demo/organizations.py"}',
                    help="Leave {} for automatic file selection. To override it, map every target_module to a different repository-relative Python file.",
                    disabled=st.session_state.agent_status is not None,
                )

        col_start, col_clear = st.columns(2)
        with col_start:
            if st.button("▶️ Start Agent", type="primary", disabled=st.session_state.agent_status is not None):
                if not task_spec.strip():
                    st.warning(
                        "The task specification is empty. Run Phase 1 extraction first, "
                        "or paste a spec JSON here."
                    )
                else:
                    with st.spinner("Preparing the workspace and generating draft files..."):
                        try:
                            payload = {
                                "thread_id": st.session_state.agent_thread_id,
                                "task_spec": task_spec,
                            }
                            if use_repository:
                                mappings = json.loads(mapping_text)
                                if not repository_url.strip() or not isinstance(mappings, dict):
                                    raise ValueError("Provide a repository URL. Optional file mappings must be a JSON object; leave {} for automatic selection.")
                                payload["repository"] = {
                                    "url": repository_url.strip(),
                                    "base_branch": repository_branch.strip() or None,
                                    "file_mappings": mappings,
                                }
                            resp = requests.post(
                                "http://127.0.0.1:8000/api/phase2/start",
                                json=payload,
                                timeout=900,
                            )
                            if resp.status_code == 200:
                                update_agent_review(resp.json())
                                st.session_state.pending_reviews = []
                                st.rerun()
                            else:
                                st.error(f"Agent start error ({resp.status_code}): {resp.text}")
                        except Exception as e:
                            st.error(f"Agent start request failed: {e}")
                        finally:
                            fetch_logs()
        with col_clear:
            if st.button("🔄 New Thread"):
                st.session_state.agent_thread_id = str(uuid.uuid4())
                st.session_state.agent_status = None
                st.session_state.agent_output = ""
                st.session_state.agent_modules = []
                st.session_state.agent_completed = False
                st.session_state.agent_repository = None
                st.session_state.agent_error = None
                st.session_state.agent_module_groups = {}
                st.session_state.agent_missing_modules = []
                st.session_state.critic_report = None
                st.session_state.critic_previous_drafts = []
                st.session_state.critic_repair_rounds = 0
                st.session_state.critic_progress = {}
                st.session_state.critic_pending_repair = False
                st.session_state.critic_repair_in_progress = False
                st.session_state.documentation = None
                st.session_state.documentation_stale = False
                st.session_state.pull_request = None
                st.rerun()

        if st.session_state.agent_status or st.session_state.agent_modules or st.session_state.agent_output:
            st.write(f"**Status:** `{st.session_state.agent_status}`")
            groups = st.session_state.agent_module_groups
            if any(len(sources) > 1 for sources in groups.values()):
                st.caption(f"Grouped {sum(len(sources) for sources in groups.values())} extracted features into {len(groups)} code modules; extracted requirements are retained.")
                with st.expander("Merged feature groups"):
                    for module, sources in groups.items():
                        if len(sources) > 1:
                            st.write(f"{module}: {', '.join(sources)}")
            repository = st.session_state.agent_repository
            if repository:
                st.caption(f"Repository branch: {repository['work_branch']} | Base: {repository['base_branch']} @ {repository['baseline_commit'][:12]}")
            if st.session_state.agent_error:
                st.error(st.session_state.agent_error)
            if st.session_state.agent_missing_modules and st.session_state.agent_status not in {"repository_failed", "review_missing"}:
                st.warning("Missing drafts: " + ", ".join(st.session_state.agent_missing_modules))
                if st.button("Retry missing modules", key="agent_retry_missing"):
                    submit_agent_feedback("Retry missing modules")
            if st.button("Refresh draft files"):
                try:
                    response = requests.get(
                        f"http://127.0.0.1:8000/api/phase2/state/{st.session_state.agent_thread_id}",
                        timeout=15,
                    )
                    if response.status_code == 200:
                        update_agent_review(response.json())
                        st.rerun()
                    elif response.status_code == 404:
                        mark_review_missing()
                        st.rerun()
                    else:
                        st.error(f"Could not refresh drafts: {response.text}")
                except Exception as exc:
                    st.error(f"Could not refresh drafts: {exc}")

            if st.session_state.agent_modules:
                st.markdown("**File review**")
                for mod in st.session_state.agent_modules:
                    ok = "✅" if mod.get("syntax_valid") else "⚠️"
                    file_path = mod.get("file_path") or mod["target_module"].replace(".", "/") + ".py"
                    with st.expander(f"{ok} {file_path}", expanded=True):
                        st.caption(mod.get("validation_msg") or "Validation not run")
                        saved = mod.get("status") == "COMMITTED"
                        if saved:
                            destination = f"repository/{file_path}" if repository else f"generated_src/{file_path}"
                            st.success(f"Approved and saved: {destination}")
                        elif mod.get("commit_error"):
                            st.error(f"Save failed: {mod['commit_error']}. Retry approval below.")
                        else:
                            st.caption("Draft — awaiting your approval")
                        if "code" not in mod:
                            st.warning("This cached draft is missing its code. Click Refresh draft files to load it.")
                            continue
                        st.code(mod.get("code", ""), language="python")
                        st.download_button(
                            "Download file", data=mod.get("code", ""),
                            file_name=file_path.rsplit("/", 1)[-1], mime="text/x-python",
                            key=f"agent_download_{st.session_state.agent_thread_id}_{mod['target_module']}",
                        )
                        if not st.session_state.agent_completed and st.session_state.agent_status != "review_missing":
                            module_key = f"{st.session_state.agent_thread_id}_{mod['target_module']}"
                            file_feedback = st.text_area(
                                f"Feedback for {file_path}",
                                placeholder="Describe the changes you want in this file...",
                                key=f"file_feedback_{module_key}",
                            )
                            if st.button(
                                "Update this file", key=f"revise_{module_key}",
                                disabled=not file_feedback.strip(),
                            ):
                                submit_agent_feedback(file_feedback, target_module=mod["target_module"])
                            if saved:
                                st.caption("Updating this file creates a new draft that needs approval again.")


            if st.session_state.agent_status == "approved":
                destination = "the repository working branch" if repository else "`generated_src/`"
                st.success(f"✅ All files approved and saved to {destination}.")
            elif st.session_state.agent_status == "repository_failed":
                st.info("Repository preparation failed before code generation. Correct the inputs and start a New Thread.")
            elif st.session_state.agent_status == "commit_failed":
                st.error("A file could not be saved. Retry approval on that file's card.")
            elif st.session_state.agent_status == "review_missing":
                st.info("Save copies of the drafts using Download file, then use New Thread to begin another review.")
            else:
                st.info("⏸️ Agent is paused and waiting for your feedback.")
            if not st.session_state.agent_modules and st.session_state.agent_status != "repository_failed":
                st.warning("No individual draft files are loaded. Click Refresh draft files to restore the saved review.")

            if not st.session_state.agent_completed and st.session_state.agent_status not in {"repository_failed", "review_missing"}:
                feedback = st.text_input(
                    "Project-wide revision (all updated files will need approval again)",
                    key="agent_feedback",
                )
                if st.button("📨 Submit Feedback", disabled=not feedback.strip()):
                    submit_agent_feedback(feedback)


    with tab3:
        st.subheader("Requirements Critic & Repair")
        st.caption("Review findings and repairs, then approve the project and continue to documentation. Remaining findings will be included in the report.")
        available = bool(st.session_state.agent_modules) and st.session_state.agent_status not in {"repository_failed", "review_missing"}
        progress = st.session_state.critic_progress
        pending = st.session_state.critic_pending_repair
        repairing = st.session_state.critic_repair_in_progress
        if progress.get("status") in {"paused", "failed"}:
            st.info("The review is incomplete. Run Critic will continue using your saved progress.")
        if repairing:
            st.info("Repair paused by a provider or time limit. Repair findings will resume saved work.")
        elif pending:
            st.info("Repaired code is saved. Repair findings will resume its automatic review.")
        st.button("Run Critic", key="critic_review", type="primary", disabled=not available or repairing,
                  on_click=critic_action, args=("review",))
        report = st.session_state.critic_report
        if not available:
            st.info("Generate module drafts in Phase 2, then click Run Critic here.")
        if report:
            stale = st.session_state.critic_stale
            if stale:
                st.warning("This report is outdated. Run Critic again before repairing or saving.")
            else:
                st.caption("Report is current for the last fetched candidate. The backend rechecks before repair and approval.")
            st.write(f"**Verdict:** `{report['verdict']}`")
            st.write(report["summary"])
            if report["verdict"] != "pass":
                st.warning("Unresolved or uncertain items remain. They will be disclosed in the documentation and draft PR.")
            st.dataframe([{"Requirement": c["requirement_id"], "Module": c.get("target_module") or "Project-wide",
                           "Description": c["description"], "Status": c["status"], "Evidence": c["evidence"]}
                          for c in report["coverage"]], hide_index=True)
            for path in sorted({f["file_path"] for f in report["findings"]}):
                with st.expander(f"Findings: {path}", expanded=True):
                    for finding in report["findings"]:
                        if finding["file_path"] != path:
                            continue
                        category = {"defect": "Confirmed defect", "uncertain": "Needs verification", "suggestion": "Optional improvement"}.get(finding.get("category", "defect"))
                        st.write(f"**{category} · {finding['severity']} · {finding['id']}** — line {finding.get('line') or 'unspecified'}")
                        st.write(finding["explanation"])
                        st.write(f"Suggested correction: {finding['suggested_fix']}")
                        if finding.get("requirement_quote"):
                            st.caption("Requirement evidence: " + finding["requirement_quote"])
                        if finding.get("code_quote"):
                            st.code(finding["code_quote"], language="python")
                        if finding.get("counterexample"):
                            st.write("Example: " + finding["counterexample"])
                        st.caption("Requirements: " + ", ".join(finding["requirement_ids"]))
            rounds = st.session_state.critic_repair_rounds
            repair_needed = any(f.get("category", "defect") == "defect" for f in report["findings"]) or any(c["status"] in {"partial", "missing"} and c.get("target_module") for c in report["coverage"])
            st.caption(f"Repair attempts: {rounds}/2")
            st.button("Repair findings", key="critic_repair", disabled=not available or (stale and not pending) or (not pending and (not repair_needed or rounds >= 2)),
                      on_click=critic_action, args=("repair",))
            if rounds >= 2 and not pending:
                st.info("Repair limit reached. Use Phase 2 revision feedback for further changes, then run Critic again.")
            with st.expander("Review limitations"):
                for limitation in report["limitations"]:
                    st.write(limitation)
            st.download_button("Download critic report", json.dumps(report, indent=2),
                               file_name="critic_report.json", mime="application/json")
            st.button("Approve all & continue", key="critic_approve_all", type="primary",
                      disabled=not available or stale or pending or bool(st.session_state.agent_missing_modules)
                      or any(not m.get("syntax_valid") for m in st.session_state.agent_modules),
                      on_click=critic_action, args=("approve-all",))
            previous = {d["target_module"]: d["code"] for d in st.session_state.critic_previous_drafts}
            for module in st.session_state.agent_modules:
                name = module["target_module"]
                path = module.get("file_path") or name.replace(".", "/") + ".py"
                with st.expander(f"Review & save: {path}"):
                    if name in previous and previous[name] != module["code"]:
                        st.code("".join(difflib.unified_diff(previous[name].splitlines(True), module["code"].splitlines(True),
                                                          fromfile="previous/" + path, tofile="repaired/" + path)), language="diff")
                    st.code(module.get("code", ""), language="python")
                    if module.get("commit_error"):
                        st.error(module["commit_error"])
                    saved = module.get("status") == "COMMITTED"
                    if saved:
                        st.success("Approved and saved locally.")
                    st.button("Approved & saved" if saved else "Approve & save this file",
                              key=f"approve_{st.session_state.agent_thread_id}_{name}", type="primary",
                              disabled=not available or stale or pending or saved or not module.get("syntax_valid") or not module.get("code", "").strip(),
                              on_click=critic_action, args=("approve", name))

        with st.expander("Recovery options (only after losing your browser session)"):
            st.caption(f"Current run ID: {st.session_state.agent_thread_id}")
            st.caption("Your current drafts are selected automatically. Use this only to reopen a different saved session.")
            saved_thread = st.text_input("Previous run ID", key="critic_resume_thread")
            if st.button("Load saved run", disabled=not saved_thread.strip()):
                try:
                    response = requests.get(f"http://127.0.0.1:8000/api/phase2/state/{saved_thread.strip()}", timeout=30)
                    if response.status_code == 200:
                        st.session_state.agent_thread_id = saved_thread.strip()
                        update_agent_review(response.json())
                        st.session_state.requested_phase = "Phase 3: Critic & Repair"
                        st.rerun()
                    else:
                        st.error(f"Could not load run: {response.text}")
                except requests.RequestException as exc:
                    st.error(f"Could not load run: {exc}")
            st.button("Reload saved status", key="critic_refresh", on_click=refresh_critic, disabled=not available)


    with tab4:
        st.subheader("Documentation & Pull Request")
        st.caption("Generate the Markdown change report, preview it, then publish the accepted candidate as a draft pull request.")
        ready = (st.session_state.agent_status == "approved" and bool(st.session_state.critic_report)
                 and not st.session_state.critic_stale and not st.session_state.critic_pending_repair)
        if not ready:
            st.info("Complete the critic review, then select Approve all & continue in Phase 3.")
        st.button("Generate documentation", key="delivery_generate", type="primary", disabled=not ready,
                  on_click=delivery_action, args=("documentation",))
        document = st.session_state.documentation
        if document:
            stale = st.session_state.documentation_stale
            if stale:
                st.warning("This report is outdated. Generate documentation again before publishing.")
            st.caption(f"{document['requirements_assessed']} requirements assessed; {document['findings']} remaining findings.")
            st.warning("Automated tests were not run. The testing phase was skipped.")
            with st.expander("Preview change report", expanded=True):
                st.markdown(document["markdown"])
            st.download_button("Download documentation", document["markdown"], file_name=document["filename"], mime="text/markdown")
            if st.session_state.agent_repository:
                st.text_input("Pull request title", value="Implement reviewed requirements", key="delivery_title", max_chars=200)
                st.caption("Create draft PR commits the approved files and this report, pushes the run's working branch, and opens a GitHub draft pull request.")
                st.button("Create draft PR", key="delivery_publish", disabled=not ready or stale
                          or not st.session_state.get("delivery_title", "").strip() or bool(st.session_state.pull_request),
                          on_click=delivery_action, args=("pull-request",))
            else:
                st.info("Documentation is ready to download. PR creation is available for repository runs.")
        if st.session_state.pull_request:
            pr = st.session_state.pull_request
            st.success(f"Draft PR #{pr['number']} created.")
            st.link_button("Open pull request", pr["url"])


with col_logs:
    st.subheader("System Execution Logs")
    if st.button("🔄 Refresh Logs"):
        fetch_logs()
    st.code(st.session_state.execution_logs, language="text")
