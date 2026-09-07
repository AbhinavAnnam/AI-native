import streamlit as st
import requests

st.set_page_config(page_title="AI-Native Control Center", layout="wide")

st.title("⚙️ AI-Native Core Control Center")

if "execution_logs" not in st.session_state:
    st.session_state.execution_logs = "No execution logs fetched yet."
if "extracted_specs" not in st.session_state:
    st.session_state.extracted_specs = None

def fetch_logs():
    try:
        log_res = requests.get("http://127.0.0.1:8000/api/logs", timeout=5)
        if log_res.status_code == 200:
            st.session_state.execution_logs = log_res.json().get("logs", "No logs found.")
        else:
            st.session_state.execution_logs = f"Failed to fetch logs. Status: {log_res.status_code}"
    except Exception as e:
        st.session_state.execution_logs = f"Could not retrieve system logs: {str(e)}"

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Document Upload & Pipeline Execution")
    
    uploaded_files = st.file_uploader(
        "Upload PRD or Architecture PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process Documents & Extract Specs", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF document first.")
        else:
            with st.spinner("Running Phase 1 extraction pipeline..."):
                files_payload = [
                    ("files", (file.name, file.getvalue(), "application/pdf")) 
                    for file in uploaded_files
                ]
                
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/api/phase1/extract", 
                        files=files_payload,
                        timeout=180
                    )
                    
                    if response.status_code == 200:
                        st.success("Extraction Completed Successfully!")
                        st.session_state.extracted_specs = response.json().get("specs")
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                
                except Exception as e:
                    st.error(f"Failed to communicate with backend API: {e}")
                
                finally:
                    fetch_logs()

    if st.session_state.extracted_specs:
        st.subheader("Extracted Specifications JSON")
        st.json(st.session_state.extracted_specs)

with col2:
    st.subheader("System Execution Logs")
    
    if st.button("🔄 Refresh Logs"):
        fetch_logs()
    
    st.code(st.session_state.execution_logs, language="text")