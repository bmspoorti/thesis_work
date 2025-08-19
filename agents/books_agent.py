import os
from dotenv import load_dotenv
from utils.google_search import GoogleSearch
from langchain_openai import ChatOpenAI
import streamlit as st

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
        "api_key": os.getenv("SERPAPI_API_KEY") or st.secrets.get("SERPAPI_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    books = results.get("organic_results", [])
    if not books:
        return []

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
        return "❌ No relevant books found."

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

# -----------------------
# 4. Simple Interactive CLI
# -----------------------
if __name__ == "__main__":
    serp_key = os.getenv("SERPAPI_API_KEY") or st.secrets.get("SERPAPI_API_KEY")
    if not serp_key:
        print("❌ SERPAPI_API_KEY not found in environment. Please set it in your .env file.")
    else:
        print("📚 Books Agent — type your query (e.g., 'book recommendations for deep learning').")
        print("Press Enter on an empty line or type 'exit' to quit.\n")

        while True:
            try:
                query = input("🔎 Query: ").strip()
                if query.lower() == "exit" or query == "":
                    print("👋 Bye!")
                    break

                answer = run_books_agent(query)
                print("\n✅ Suggestions:\n")
                print(answer)
                print("\n" + "-" * 60 + "\n")
            except KeyboardInterrupt:
                print("\n👋 Bye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
