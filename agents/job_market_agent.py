import os
import json
from utils.google_search import GoogleSearch
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# ------------------ Fetch Jobs from Web ------------------
def search_jobs(query: str):
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY") or st.secrets.get("SERPAPI_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    jobs = results.get("organic_results", [])
    if not jobs:
        print("❌ No job listings found.")
        return []

    top_links = []
    for i, job in enumerate(jobs[:5], 1):
        title = job.get("title", "No title")
        link = job.get("link", "No link")
        snippet = job.get("snippet", "")
        top_links.append({
            "title": title,
            "link": link,
            "snippet": snippet
        })

    return top_links

# ------------------ Summarize the Jobs ------------------
def summarize_jobs(job_listings):
    if not job_listings:
        return "⚠️ No job data to summarize."

    chat = ChatOpenAI(model="gpt-4", temperature=0.3)
    prompt = f"""
You are an expert job market analyst. Summarize the following job listings in terms of:
- Common job titles
- Typical skills required
- Key tools or technologies mentioned
- Any emerging trends you notice

Here are the listings:
{json.dumps(job_listings, indent=2)}
"""
    response = chat.invoke(prompt)
    return response.content

# ------------------ Unified Runner (used in main_graph) ------------------
def run_job_market_agent(query: str):
    listings = search_jobs(query)
    if not listings:
        return "❌ No job listings found for your query."

    # ✅ Save fresh listings to file every time
    os.makedirs("data", exist_ok=True)
    with open("data/job_listings.json", "w", encoding="utf-8") as f:
        json.dump(listings, f, indent=2, ensure_ascii=False)

    # 🔍 Summarize the job listings
    summary = summarize_jobs(listings)

    # ✅ Build display output
    jobs_markdown = "\n\n".join([
        f"**{i+1}. {job['title']}**\n{job['snippet']}\n🔗 [Apply Here]({job['link']})"
        for i, job in enumerate(listings[:5])
    ])

    return f"### 📊 Job Market Summary:\n\n{summary}\n\n---\n### 💼 Top Job Listings:\n\n{jobs_markdown}"

