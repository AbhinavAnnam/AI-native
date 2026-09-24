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

with col_logs:
    st.subheader("System Execution Logs")
    if st.button("🔄 Refresh Logs"):
        fetch_logs()
    st.code(st.session_state.execution_logs, language="text")
