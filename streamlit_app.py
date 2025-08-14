import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

# Utilities
from utils.parse_pdf import parse_single_pdf
from utils.chunker import chunk_text
from langgraph_workflow.main_graph import app

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

user_query = st.chat_input("Ask a question about courses, jobs, skills, or books...")

if user_query:
    # Log user message
    st.session_state.chat_history.append(("user", user_query))

    with st.spinner("🤖 Thinking..."):
        result = app.invoke({
            "query": user_query,
            "curriculum_mode": curriculum_mode_flag,
            "uploaded_docs": uploaded_docs
        })

    # Log assistant answer
    st.session_state.chat_history.append(("assistant", result["result"]))

    # Show which agent was used
    agent_used = result.get("agent", "unknown")
    st.session_state.chat_history.append(("system", f"📌 Routed to: `{agent_used}` agent"))

    # ---------------- Show job listings ----------------
    if agent_used == "job_market":
        listings_path = "data/job_listings.json"
        if os.path.exists(listings_path):
            with open(listings_path, "r", encoding="utf-8") as f:
                listings = json.load(f)

            job_display = "### 💼 Top Job Listings\n"
            for job in listings[:5]:
                job_display += f"**{job.get('title', 'No Title')}**\n"
                job_display += f"{job.get('snippet', '')}\n"
                job_display += f"[Apply Here]({job.get('link', '#')})\n\n"

            st.session_state.chat_history.append(("system", job_display.strip()))

# ---------------- Render Chat ----------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
