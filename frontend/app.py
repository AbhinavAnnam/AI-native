import json
import uuid

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
if "agent_task_spec" not in st.session_state:
    st.session_state.agent_task_spec = ""
if "agent_task_seed" not in st.session_state:
    st.session_state.agent_task_seed = None


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


def submit_agent_feedback(feedback, target_module=None, rerun=True):
    payload = {"thread_id": st.session_state.agent_thread_id, "user_feedback": feedback}
    if target_module is not None:
        payload["target_module"] = target_module
    operation = "Saving" if feedback.strip().lower() == "approve" else "Updating"
    with st.spinner(f"{operation} {target_module}..." if target_module else "Applying project feedback..."):
        try:
            response = requests.post("http://127.0.0.1:8000/api/phase2/feedback", json=payload, timeout=900)
            if response.status_code == 200:
                update_agent_review(response.json())
                if rerun:
                    st.rerun()
            else:
                st.error(f"Feedback error ({response.status_code}): {response.text}")
        except Exception as exc:
            st.error(f"Feedback request failed: {exc}")
        finally:
            fetch_logs()


col_main, col_logs = st.columns([2, 1])

with col_main:
    tab1, tab2 = st.tabs(["📋 Phase 1: Requirements", "💻 Phase 2: Coder Agent & HITL"])

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
            st.json(st.session_state.extracted_specs)

    # ------------------------------------------------------------------
    # TAB 2: Phase 2 (HITL)
    # ------------------------------------------------------------------
    with tab2:
        st.subheader("Coder Agent Synthesis & Human-In-The-Loop Approval")

        if not st.session_state.extracted_specs:
            st.info("Execute Phase 1 extraction in Tab 1 first.")
        else:
            st.caption("Starting a new coder run clears files from the previous run in generated_src/.")
            if st.button("🤖 Run Coder Agent (Synthesize Code)", type="primary"):
                with st.spinner("Coder Agent generating python modules..."):
                    try:
                        resp = requests.post(
                            "http://127.0.0.1:8000/api/phase2/generate-drafts",
                            json=st.session_state.extracted_specs,
                            timeout=180
                        )
                        if resp.status_code == 200:
                            st.session_state.pending_reviews = resp.json().get("pending_reviews", [])
                            update_agent_review({})
                            st.session_state.agent_thread_id = str(uuid.uuid4())
                            for key in list(st.session_state):
                                if key.startswith("code_editor_"):
                                    del st.session_state[key]
                            st.success("Module drafts generated and ready for review!")
                        else:
                            st.error(f"Error ({resp.status_code}): {resp.text}")
                    except Exception as e:
                        st.error(f"Backend Request Failed: {e}")
                    finally:
                        fetch_logs()

        if st.session_state.pending_reviews:
            st.markdown("---")
            st.subheader("HITL Approval Queue")

            for item in st.session_state.pending_reviews:
                draft_id = item["draft_id"]
                module = item["target_module"]
                status = item["status"]

                with st.expander(f"📦 Module: {module} | Status: {status}", expanded=(status == "AWAITING_HUMAN_REVIEW")):
                    st.write(f"**Feature:** {item['feature_name']}")
                    st.caption(f"MCP Syntax Check: {item['validation_msg']}")

                    if status == "COMMITTED":
                        st.success("✅ Module approved and committed to disk via MCP tool!")
                        st.code(item["code"], language="python")
                    elif status == "REJECTED":
                        st.error("❌ Draft rejected by operator.")
                    else:
                        edited_code = st.text_area(
                            f"Review & Edit Code for `{module}`",
                            value=item["code"],
                            height=280,
                            key=f"code_editor_{draft_id}"
                        )

                        col_app, col_rej = st.columns(2)

                        with col_app:
                            if st.button(f"✅ Approve & Deploy ({module})", key=f"app_btn_{draft_id}"):
                                payload = {
                                    "draft_id": draft_id,
                                    "target_module": module,
                                    "approved_code": edited_code,
                                    "action": "approve"
                                }
                                commit_res = requests.post("http://127.0.0.1:8000/api/phase2/approve-and-commit", json=payload)
                                if commit_res.status_code == 200:
                                    item["status"] = "COMMITTED"
                                    item["code"] = edited_code
                                    st.success(f"Committed to {commit_res.json().get('file_path')}")
                                    st.rerun()
                                else:
                                    st.error(f"MCP Deployment Error: {commit_res.text}")
                                fetch_logs()

                        with col_rej:
                            if st.button(f"❌ Reject Draft ({module})", key=f"rej_btn_{draft_id}"):
                                payload = {
                                    "draft_id": draft_id,
                                    "target_module": module,
                                    "approved_code": "",
                                    "action": "reject"
                                }
                                requests.post("http://127.0.0.1:8000/api/phase2/approve-and-commit", json=payload)
                                item["status"] = "REJECTED"
                                st.rerun()

        # ------------------------------------------------------------------
        # Agent Studio (LangGraph Human-in-the-Loop Stream)
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🤖 Agent Studio — LangGraph Human-in-the-Loop")
        st.caption(
            "Review each draft file below before approval. Use a file's feedback box to update only that file, "
            "or use project feedback to revise all files. Updated drafts appear here for another review. "
            "Click Approve & save this file to save that file to generated_src/. "
            "You can keep reviewing the remaining files afterward. "
            "Starting a new run clears the previous run's saved files."
        )

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

        col_start, col_clear = st.columns(2)
        with col_start:
            if st.button("▶️ Start Agent", type="primary", disabled=st.session_state.agent_status is not None):
                if not task_spec.strip():
                    st.warning(
                        "The task specification is empty. Run Phase 1 extraction first, "
                        "or paste a spec JSON here."
                    )
                else:
                    with st.spinner("Generating module files via the CoderAgent + MCP tools..."):
                        try:
                            resp = requests.post(
                                "http://127.0.0.1:8000/api/phase2/start",
                                json={
                                    "thread_id": st.session_state.agent_thread_id,
                                    "task_spec": task_spec,
                                },
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
                st.rerun()

        if st.session_state.agent_status or st.session_state.agent_modules or st.session_state.agent_output:
            st.write(f"**Status:** `{st.session_state.agent_status}`")
            if st.button("Refresh draft files"):
                try:
                    response = requests.get(
                        f"http://127.0.0.1:8000/api/phase2/state/{st.session_state.agent_thread_id}",
                        timeout=15,
                    )
                    if response.status_code == 200:
                        update_agent_review(response.json())
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
                            st.success(f"Approved and saved: generated_src/{file_path}")
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
                        if not st.session_state.agent_completed:
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
                            st.button(
                                "Approved & saved" if saved else "Approve & save this file",
                                key=f"approve_{module_key}", type="primary",
                                disabled=saved or not mod.get("syntax_valid") or not mod.get("code", "").strip(),
                                on_click=submit_agent_feedback,
                                args=("approve", mod["target_module"]),
                                kwargs={"rerun": False},
                            )

            if st.session_state.agent_status == "approved":
                st.success("✅ All files approved and saved to `generated_src/`.")
            elif st.session_state.agent_status == "commit_failed":
                st.error("A file could not be saved. Retry approval on that file's card.")
            else:
                st.info("⏸️ Agent is paused and waiting for your feedback.")
            if not st.session_state.agent_modules:
                st.warning("No individual draft files are loaded. Click Refresh draft files; if the backend restarted, start a New Thread.")

            if not st.session_state.agent_completed:
                feedback = st.text_input(
                    "Project-wide revision (all updated files will need approval again)",
                    key="agent_feedback",
                )
                if st.button("📨 Submit Feedback", disabled=not feedback.strip()):
                    submit_agent_feedback(feedback)


with col_logs:
    st.subheader("System Execution Logs")
    if st.button("🔄 Refresh Logs"):
        fetch_logs()
    st.code(st.session_state.execution_logs, language="text")
