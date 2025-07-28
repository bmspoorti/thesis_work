import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient

load_dotenv()

# 1. Set up Qdrant client
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# 2. Load OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Connect to existing collection
vectorstore = Qdrant(
    client=qdrant,
    collection_name="srh-curriculum",
    embeddings=embeddings
)

# 4. Set up retriever
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# 5. Set up memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 6. Build conversational QA chain with GPT-4 and memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# 7. Example interaction
if __name__ == "__main__":
    query = "What courses are taught in the Data Science?"
    result = qa_chain.invoke({"question": query})

    print("\n🤖 Answer:")
    print(result["answer"])
    print("\n📚 Source Documents:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['source']} | Chunk #{doc.metadata['chunk_index']}")
