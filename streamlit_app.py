import os
import json
import time
import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

# Utilities
from utils.parse_pdf import parse_single_pdf
from utils.chunker import chunk_text
from langgraph_workflow.main_graph import app
from utils.sheets_logger import log_to_gsheet

load_dotenv()

# ---------------- UI Setup ----------------
st.set_page_config(page_title="SRH Assistant", layout="wide")
st.title("🎓 AI-Powered University Assistant")

# ---------------- Sidebar ----------------
st.sidebar.header("Curriculum Source")
curriculum_mode = st.sidebar.radio(
    "Select Curriculum Type:",
    ["SRH Curriculum", "Upload Your Curriculum PDF"]
)

uploaded_docs = None

if curriculum_mode == "Upload Your Curriculum PDF":
    uploaded_file = st.sidebar.file_uploader("Upload Curriculum PDF", type=["pdf"])
    if uploaded_file:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        parsed = parse_single_pdf("temp_uploaded.pdf")
        chunks = chunk_text(parsed[0]["content"], source=parsed[0]["filename"])
        uploaded_docs = [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in chunks
        ]
        st.sidebar.success(f"✅ Uploaded {parsed[0]['filename']} with {len(chunks)} chunks.")

curriculum_mode_flag = "uploaded" if uploaded_docs else "srh"

# ---------------- Chat Interface ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "logs" not in st.session_state:
    st.session_state.logs = ""

user_query = st.chat_input("Ask a question about courses, jobs, skills, or books...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))

    start_time = time.time()
    with st.spinner("🤖 Thinking..."):
        result = app.invoke({
            "query": user_query,
            "curriculum_mode": curriculum_mode_flag,
            "uploaded_docs": uploaded_docs
        })
    end_time = time.time()
    latency = end_time - start_time

    agent_used = result.get("agent", "unknown")
    is_fallback = agent_used == "fallback"
    timestamp = datetime.datetime.now().isoformat()
    answer = result.get("result", "")

    # Build log entry string
    log_entry = "\n" + "=" * 60 + "\n"
    log_entry += f"🕒 Timestamp: {timestamp}\n"
    log_entry += f"❓ Query: {user_query}\n"
    log_entry += f"📂 Curriculum Mode: {curriculum_mode_flag}\n"
    log_entry += f"📌 Routed Agent: {agent_used}\n"
    log_entry += f"⏱️ Latency: {latency:.2f} seconds\n"
    log_entry += f"🛡️ Fallback Used: {'Yes' if is_fallback else 'No'}\n"
    log_entry += f"📘 Final Answer:\n{answer}\n"
    log_entry += "=" * 60 + "\n"

    # 1️⃣ Save to session state
    st.session_state.logs += log_entry

    # 2️⃣ Save to local file (in dev or cloud)
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/workflow_logs.txt", "a", encoding="utf-8") as f:
            f.write(log_entry)
    except:
        pass

    # 3️⃣ Save to Google Sheets
    try:
        log_to_gsheet(
            timestamp=timestamp,
            query=user_query,
            agent=agent_used,
            curriculum_mode=curriculum_mode_flag,
            latency=round(latency, 2),
            is_fallback=is_fallback,
            result=answer
        )
    except Exception as e:
        st.warning(f"⚠️ Google Sheets logging failed: {e}")

    # Store response
    st.session_state.chat_history.append(("assistant", answer))
    st.session_state.chat_history.append(("system", f"📌 Routed to: `{agent_used}` agent"))

# ---------------- Render Chat ----------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
