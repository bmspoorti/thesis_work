import sys
import os
from dotenv import load_dotenv
import streamlit as st

# Add root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables
load_dotenv()

from utils.parse_pdf import load_all_pdfs
from utils.chunker import chunk_text

from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def create_qdrant_collection():
    """
    Creates the Qdrant collection if it doesn't exist.
    Returns a QdrantClient instance.
    """
    client = QdrantClient(
        url=os.getenv("QDRANT_URL") or st.secrets.get("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY"),
        prefer_grpc=False  # more reliable for HTTP fallback
    )

    collection_name = "srh-curriculum"

    existing_collections = [c.name for c in client.get_collections().collections]

    if collection_name not in existing_collections:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE
            )
        )

    return client


def store_embeddings():
    """
    Loads PDFs, chunks text, embeds, and stores in Qdrant vector store.
    """
    client = create_qdrant_collection()
    collection_name = "srh-curriculum"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    all_docs = []
    for pdf in load_all_pdfs("data/curriculum_pdfs"):
        chunks = chunk_text(pdf["content"], source=pdf["filename"])
        docs = [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in chunks
        ]
        all_docs.extend(docs)

    Qdrant.from_documents(
        documents=all_docs,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL") or st.secrets.get("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY"),
        collection_name=collection_name,
        prefer_grpc=False
    )

    print(f"✅ Stored {len(all_docs)} chunks in Qdrant.")


if __name__ == "__main__":
    store_embeddings()
