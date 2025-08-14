import os
import sys
import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langsmith import traceable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Agent imports
from agents.curriculum_agent import qa_chain
from agents.job_market_agent import run_job_market_agent
from agents.skill_mapping_agent import (
    fetch_curriculum_chunks,
    load_job_listings,
    analyze_skill_match,
)
from agents.books_agent import run_books_agent
from agents.fallback_agent import FallbackAgent

load_dotenv()

from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class GraphState(TypedDict):
    query: str
    agent: str
    result: str
    curriculum_mode: Optional[str]
    uploaded_docs: Optional[List[Document]]

# ------------ ROUTING -------------
def route_agent(state: GraphState):
    query = state["query"].lower()

    if any(word in query for word in ["course", "module", "subject"]):
        agent = "curriculum"
    elif any(word in query for word in ["job", "hiring", "career"]):
        agent = "job_market"
    elif any(word in query for word in ["match", "skills", "gap"]):
        agent = "skill_mapping"
    elif any(word in query for word in ["book", "reference", "resource"]):
        agent = "books"
    else:
        agent = "fallback"

    print(f"🧭 Routing to: {agent} agent")
    return {
        "agent": agent,
        "query": state["query"],
        "curriculum_mode": state.get("curriculum_mode", "srh"),
        "uploaded_docs": state.get("uploaded_docs")
    }

# ------------ NODE DEFINITIONS -------------
@traceable(name="curriculum_node")
def curriculum_node(state: GraphState):
    result = qa_chain.invoke({"question": state["query"]})
    return {"result": result["answer"], "agent": "curriculum"}

@traceable(name="job_market_node")
def job_market_node(state: GraphState):
    result = run_job_market_agent(state["query"])  # ✅ Use single entry point
    return {"result": result, "agent": "job_market"}

@traceable(name="skill_mapping_node")
def skill_mapping_node(state: GraphState):
    if state.get("curriculum_mode") == "uploaded" and state.get("uploaded_docs"):
        curriculum = state["uploaded_docs"]
    else:
        curriculum = fetch_curriculum_chunks()

    jobs = load_job_listings()
    analysis = analyze_skill_match(curriculum, jobs)
    return {"result": analysis, "agent": "skill_mapping"}

@traceable(name="books_node")
def books_node(state: GraphState):
    result = run_books_agent(state["query"])
    return {"result": result, "agent": "books"}

@traceable(name="fallback_node")
def fallback_node(state: GraphState):
    fallback = FallbackAgent()
    result = fallback.run(state["query"])
    return {"result": result, "agent": "fallback"}

# ------------ GRAPH SETUP -------------
graph = StateGraph(GraphState)

graph.add_node("router", RunnableLambda(route_agent))
graph.add_node("curriculum", RunnableLambda(curriculum_node))
graph.add_node("job_market", RunnableLambda(job_market_node))
graph.add_node("skill_mapping", RunnableLambda(skill_mapping_node))
graph.add_node("books", RunnableLambda(books_node))
graph.add_node("fallback", RunnableLambda(fallback_node))

graph.add_conditional_edges(
    "router",
    lambda state: state["agent"],
    {
        "curriculum": "curriculum",
        "job_market": "job_market",
        "skill_mapping": "skill_mapping",
        "books": "books",
        "fallback": "fallback",
    },
)

graph.add_edge("curriculum", END)
graph.add_edge("job_market", END)
graph.add_edge("skill_mapping", END)
graph.add_edge("books", END)
graph.add_edge("fallback", END)

graph.set_entry_point("router")
app = graph.compile()

# ------------ CLI EXECUTION -------------
if __name__ == "__main__":
    query = input("\n🔎 Enter your query: ")
    final_state = app.invoke({
        "query": query,
        "curriculum_mode": "srh",
        "uploaded_docs": None
    })

    print(f"\n✅ Final Answer from {final_state['agent']} Agent:\n{final_state['result']}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/workflow_logs.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"🕒 Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"🔍 Query: {query}\n")
        f.write(f"🤖 Routed Agent: {final_state['agent']}\n")
        f.write("📤 Final Answer:\n")
        f.write(final_state["result"] + "\n")
        f.write("=" * 60 + "\n")
