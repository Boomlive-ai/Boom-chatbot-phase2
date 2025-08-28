import os
import uuid
from datetime import datetime
from typing import Dict, Any

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()


def get_vector_store(index_name: str = "boom-faq-index") -> PineconeVectorStore:
    """
    Initialize Pinecone vector store using API key from environment.
    """
    api_key = os.getenv("PINECONE_API_KEY_ABHINANDAN")
    if not api_key:
        raise ValueError("PINECONE_API_KEY_ABHINANDAN is not set in environment")

    # Set the expected env var for LangChain
    os.environ["PINECONE_API_KEY"] = api_key

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(index_name=index_name, embedding=embeddings)


def store_faq(question: str, answer: str, index_name: str = "boom-faq-index") -> Dict[str, Any]:
    """
    Store a single FAQ in the vector store.
    """
    doc = Document(
        page_content=f"Question: {question}\n\nAnswer: {answer}",
        metadata={
            "id": str(uuid.uuid4()),
            "question": question,
            "answer": answer,
            "created_at": datetime.now().isoformat()
        }
    )
    vector_store = get_vector_store(index_name)
    vector_store.add_documents([doc])
    return {"success": True, "id": doc.metadata["id"]}


def search_faq(query: str, top_k: int = 3, index_name: str = "boom-faq-index") -> Dict[str, Any]:
    """
    Search FAQs in the vector store.
    """
    if not query.strip():
        return {"success": False, "error": "Query cannot be empty"}

    vector_store = get_vector_store(index_name)
    results = vector_store.similarity_search(query, k=top_k)

    return {
        "success": True,
        "query": query,
        "results": [
            {
                "id": r.metadata.get("id", ""),
                "question": r.metadata.get("question", ""),
                "answer": r.metadata.get("answer", ""),
                "created_at": r.metadata.get("created_at", ""),
                "score": getattr(r, "score", None)
            }
            for r in results
        ]
    }
