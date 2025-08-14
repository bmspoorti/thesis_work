import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

class FallbackAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)

    def run(self, query: str) -> str:
        fallback_prompt = f"""
You are a helpful university assistant chatbot. A user has asked a question that doesn't match any known category like curriculum, jobs, skill matching, or books.

Try your best to either:
- Answer the query using general knowledge, OR
- Politely let them know the system can't help with that yet.

Query:
\"\"\"{query}\"\"\"
"""
        response = self.llm.invoke(fallback_prompt)

        # Log the query to a file for future analysis
        log_entry = {
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        os.makedirs("logs", exist_ok=True)
        with open("logs/unhandled_queries.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return response.content

# -----------------------
# Simple CLI to test fallback agent
# -----------------------
if __name__ == "__main__":
    print("💬 Fallback Agent CLI")
    user_query = input("\n🔎 Enter your query: ")
    agent = FallbackAgent()
    result = agent.run(user_query)
    print("\n✅ Final Answer:\n")
    print(result)
