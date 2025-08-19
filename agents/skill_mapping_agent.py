import os
import sys
import json
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.skills_kg import extract_unique_skills, build_skill_graph, analyze_skill_gap, explain_gap

load_dotenv()


def fetch_curriculum_chunks():
    print("📘 Fetching curriculum chunks from Qdrant...")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL") or st.secrets.get("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY")
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    qdrant = Qdrant(
        client=client,
        collection_name="srh-curriculum",
        embeddings=embeddings
    )

    docs = qdrant.similarity_search("Course modules and skills", k=50)
    return docs


def load_job_listings():
    print("📄 Loading job listings from local file...")
    with open("data/job_listings.json", "r", encoding="utf-8") as f:
        return json.load(f)


def extract_skills_from_texts(curriculum_docs, job_listings):
    curriculum_skills = set()
    job_skills = set()

    for doc in curriculum_docs:
        for line in doc.page_content.splitlines():
            if "-" in line or "•" in line:
                curriculum_skills.add(line.lower().strip())

    for job in job_listings:
        snippet = job.get("snippet", "").lower()
        for word in ["python", "sql", "excel", "machine learning", "power bi", "etl", "data warehousing", "deep learning"]:
            if word in snippet:
                job_skills.add(word)

    return curriculum_skills, job_skills


def analyze_skill_match(curriculum_docs=None, job_listings=None):
    print("🤖 Analyzing skill overlap using GPT-4 and Knowledge Graph...")

    if curriculum_docs is None:
        curriculum_docs = fetch_curriculum_chunks()
    if job_listings is None:
        job_listings = load_job_listings()

    # --------- LLM Analysis ---------
    curriculum_text = "\n".join([doc.page_content for doc in curriculum_docs])
    job_text = "\n".join([
        f"{job['title']}\n{job['snippet']}" for job in job_listings
    ])

    prompt = f"""
You are a career guidance AI.
Compare the following university curriculum and job market listings.
Your goal is to:
- Identify skill overlap
- Point out missing skills in the curriculum
- Recommend external learning paths

### Curriculum Content:
{curriculum_text}

### Job Listings:
{job_text}

Respond with a structured breakdown including:
1. Skill Match Percentage
2. Matched Skills
3. Missing Skills
4. Suggested Online Courses or Topics
"""
    chat = ChatOpenAI(model="gpt-4", temperature=0.2)
    gpt_result = chat.invoke(prompt)

    # --------- KG Analysis ---------
    curriculum_skills, job_skills = extract_skills_from_texts(curriculum_docs, job_listings)
    G = build_skill_graph(curriculum_skills, job_skills)
    missing_skills, related_paths = analyze_skill_gap(G)
    kg_explanation = explain_gap(missing_skills, related_paths)

    return gpt_result.content + "\n\n" + kg_explanation


if __name__ == "__main__":
    print("🔍 Welcome to Skill Mapping CLI (SRH Curriculum vs Job Market)\n")
    input("Press Enter to start skill analysis...\n")

    curriculum_docs = fetch_curriculum_chunks()
    job_listings = load_job_listings()
    result = analyze_skill_match(curriculum_docs, job_listings)

    print("\n✅ Final Result:\n")
    print(result)
