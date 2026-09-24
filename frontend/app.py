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
            "This drives the real LangChain/LangGraph agent. It creates a draft, pauses, then waits for your "
            "feedback. Type an approval keyword (e.g. 'approve', 'looks good') to accept it, or type revision "
            "instructions to regenerate the output."
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
            if st.button("▶️ Start Agent", type="primary"):
                if not task_spec.strip():
                    st.warning(
                        "The task specification is empty. Run Phase 1 extraction first, "
                        "or paste a spec JSON here."
                    )
                else:
                    with st.spinner("Agent is classifying the spec, then synthesizing code via the CoderAgent + MCP tools..."):
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
                                data = resp.json()
                                st.session_state.agent_status = data.get("status")
                                st.session_state.agent_output = data.get("generated_output") or ""
                                st.session_state.agent_modules = data.get("modules") or []
                                st.session_state.agent_completed = False
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

        if st.session_state.agent_output:
            st.write(f"**Status:** `{st.session_state.agent_status}`")

            if st.session_state.agent_modules:
                st.markdown("**Modules generated this pass**")
                for mod in st.session_state.agent_modules:
                    ok = "✅" if mod.get("syntax_valid") else "⚠️"
                    st.markdown(
                        f"- {ok} `{mod.get('target_module')}` — {mod.get('validation_msg')}"
                    )

            if st.session_state.agent_completed:
                st.success("✅ Agent approved — modules committed to `generated_src/`.")
            else:
                st.info("⏸️ Agent is paused and waiting for your feedback.")
            st.code(st.session_state.agent_output, language="python")

            if not st.session_state.agent_completed:
                feedback = st.text_input(
                    "Your feedback (e.g. 'approve' / 'looks good', or revision notes)",
                    key="agent_feedback",
                )
                if st.button("📨 Submit Feedback"):
                    with st.spinner("Applying feedback and continuing the agent..."):
                        try:
                            resp = requests.post(
                                "http://127.0.0.1:8000/api/phase2/feedback",
                                json={
                                    "thread_id": st.session_state.agent_thread_id,
                                    "user_feedback": feedback,
                                },
                                timeout=900,
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                st.session_state.agent_status = data.get("status")
                                st.session_state.agent_output = data.get("latest_output") or st.session_state.agent_output
                                st.session_state.agent_modules = data.get("modules") or st.session_state.agent_modules
                                st.session_state.agent_completed = bool(data.get("is_completed"))
                                st.rerun()
                            else:
                                st.error(f"Feedback error ({resp.status_code}): {resp.text}")
                        except Exception as e:
                            st.error(f"Feedback request failed: {e}")
                        finally:
                            fetch_logs()


with col_logs:
    st.subheader("System Execution Logs")
    if st.button("🔄 Refresh Logs"):
        fetch_logs()
    st.code(st.session_state.execution_logs, language="text")
