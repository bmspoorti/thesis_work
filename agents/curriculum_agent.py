import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
import streamlit as st

load_dotenv()

# Qdrant client
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL") or st.secrets.get("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY"),
)

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vector store
vectorstore = Qdrant(
    client=qdrant,
    collection_name="srh-curriculum",
    embeddings=embeddings,
)

# Retriever
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)

# QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer",
)

# Simple CLI
if __name__ == "__main__":
    query = input("\n🔎 Enter your query: ")
    result = qa_chain.invoke({"question": query})

    print("\n🤖 Answer:")
    print(result["answer"])
