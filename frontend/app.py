import streamlit as st
import requests

st.set_page_config(page_title="AI-Native Control Center", layout="wide")

st.title("⚙️ AI-Native Core Control Center")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Document Upload & Pipeline Execution")
    
    # Drag-and-drop widget for single or multiple PDFs
    uploaded_files = st.file_uploader(
        "Upload PRD or Architecture PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process Documents & Extract Specs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF document first.")
        else:
            with st.spinner("Processing uploaded documents & querying Gemini..."):
                # Prepare multipart file payload for FastAPI
                files_payload = [
                    ("files", (file.name, file.getvalue(), "application/pdf")) 
                    for file in uploaded_files
                ]
                
                try:
                    response = requests.post("http://127.0.0.1:8000/api/phase1/extract", files=files_payload)
                    if response.status_code == 200:
                        st.success("Extraction Completed!")
                        st.json(response.json()["specs"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with backend API: {e}")

with col2:
    st.subheader("System Execution Logs")
    if st.button("🔄 Refresh Logs"):
        try:
            log_res = requests.get("http://127.0.0.1:8000/api/logs")
            if log_res.status_code == 200:
                st.code(log_res.json().get("logs", "No logs found."))
        except Exception:
            st.error("Could not retrieve system logs.")