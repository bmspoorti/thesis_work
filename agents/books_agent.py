import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# -----------------------
# 1. Search for books
# -----------------------
def search_books(query: str):
    params = {
        "engine": "google",
        "q": f"{query} site:goodreads.com",
        "api_key": os.getenv("SERPAPI_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    books = results.get("organic_results", [])
    if not books:
        return "❌ No relevant books found."

    suggestions = []
    for book in books[:5]:
        title = book.get("title", "No title")
        snippet = book.get("snippet", "")
        link = book.get("link", "")
        suggestions.append(f"{title}\n{snippet}\n{link}")

    return suggestions

# -----------------------
# 2. Summarize results
# -----------------------
def summarize_books(book_list):
    if not book_list:
        return "⚠️ No book data to summarize."

    prompt = f"""
You are a university assistant. Recommend a few relevant books or learning resources based on the following search results:

{book_list}

Just give concise bullet-point suggestions.
"""
    return llm.invoke(prompt).content

# -----------------------
# 3. Entry point for LangGraph
# -----------------------
def run_books_agent(query: str):
    results = search_books(query)
    return summarize_books(results)
