import os
import json
from serpapi import GoogleSearch
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def search_jobs(query: str):
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    jobs = results.get("organic_results", [])
    if not jobs:
        print("❌ No job listings found.")
        return []

    top_links = []
    print(f"✅ Found {len(jobs)} listings:\n")
    for i, job in enumerate(jobs[:5], 1):
        title = job.get("title", "No title")
        link = job.get("link", "No link")
        snippet = job.get("snippet", "")
        print(f"{i}. {title}\n   {link}\n   {snippet}\n")
        top_links.append({
            "title": title,
            "link": link,
            "snippet": snippet
        })

    return top_links

def summarize_jobs(job_listings):
    if not job_listings:
        print("⚠️ No job data to summarize.")
        return

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
    print("\n🔍 Summary of Job Market:\n")
    print(response.content)

def save_job_listings(listings):
    os.makedirs("data", exist_ok=True)
    with open("data/job_listings.json", "w", encoding="utf-8") as f:
        json.dump(listings, f, indent=2, ensure_ascii=False)
    print("💾 Saved job listings to data/job_listings.json")

if __name__ == "__main__":
    query = "Data Scientist jobs in Heidelberg site:linkedin.com"
    listings = search_jobs(query)
    summarize_jobs(listings)
    save_job_listings(listings)
