import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agents"))

# Import agent modules
from curriculum_agent import qa_chain
from job_market_agent import search_jobs, summarize_jobs
from skill_mapping_agent import (
    fetch_curriculum_chunks,
    load_job_listings,
    analyze_skill_match,
)
from fallback_agent import FallbackAgent

load_dotenv()

# Load GPT model
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ------------ WRAPPER FUNCTIONS FOR EACH AGENT ------------

def run_curriculum_agent(query):
    result = qa_chain.invoke({"question": query})
    print("\n🤖 Curriculum Answer:\n", result["answer"])
    return result["answer"]

def run_job_market_agent(query):
    listings = search_jobs(query)
    if not listings:
        return "❌ No job listings found."
    return summarize_jobs(listings)

def run_skill_mapping_agent(_):
    curriculum_docs = fetch_curriculum_chunks()
    job_listings = load_job_listings()
    return analyze_skill_match(curriculum_docs, job_listings)

def run_books_agent(query):
    from books_agent import run_books_agent
    return run_books_agent(query)

def run_fallback_agent(query):
    fallback = FallbackAgent()
    return fallback.run(query)

# ------------ SUPERVISOR LOGIC ------------

def supervisor_router(query: str):
    query_lower = query.lower()

    if any(word in query_lower for word in ["course", "subject", "module"]):
        print("🧠 Routing to Curriculum Agent...")
        return run_curriculum_agent(query)

    elif any(word in query_lower for word in ["job", "career", "hiring"]):
        print("💼 Routing to Job Market Agent...")
        return run_job_market_agent(query)

    elif any(word in query_lower for word in ["match", "skills", "gap"]):
        print("📊 Routing to Skill Mapping Agent...")
        return run_skill_mapping_agent(query)

    elif any(word in query_lower for word in ["book", "resource", "learn"]):
        print("📚 Routing to Books Agent...")
        return run_books_agent(query)

    else:
        print("🤖 Unknown query type. Using Fallback Agent...")
        return run_fallback_agent(query)

# ------------ CLI EXECUTION ------------

if __name__ == "__main__":
    query = input("\n🔎 Enter your query: ")
    result = supervisor_router(query)
    print("\n✅ Final Answer:\n", result)
