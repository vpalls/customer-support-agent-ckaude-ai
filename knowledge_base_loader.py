"""
Knowledge Base Loader — ChromaDB + HuggingFace Embeddings
Replaces OpenAI embeddings with free, local sentence-transformers.
"""
import json
import os
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Paths ──
DATA_DIR = os.environ.get("DATA_DIR", "./data")
KB_FILE = os.path.join(DATA_DIR, "router_agent_documents.json")
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")


def get_embedding_model():
    """
    Use HuggingFace sentence-transformers (free, no API key needed).
    all-MiniLM-L6-v2 is fast and high-quality for retrieval.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_index_knowledge_base() -> Chroma:
    """
    Load the knowledge base JSON and index into ChromaDB.
    If a persisted index already exists, load from disk.
    """
    embed_model = get_embedding_model()

    # Check if persisted DB exists
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print(f"📂 Loading existing ChromaDB from {PERSIST_DIR}...")
        kbase_db = Chroma(
            collection_name="knowledge_base",
            embedding_function=embed_model,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=PERSIST_DIR,
        )
        # Verify collection has docs
        try:
            count = kbase_db._collection.count()
            if count > 0:
                print(f"   ✅ Loaded {count} documents from persisted index")
                return kbase_db
            else:
                print("   ⚠️  Persisted index is empty, re-indexing...")
        except Exception:
            print("   ⚠️  Could not read persisted index, re-indexing...")

    # Load from JSON and index
    if not os.path.exists(KB_FILE):
        raise FileNotFoundError(
            f"Knowledge base file not found at {KB_FILE}.\n"
            f"Please download it from: "
            f"https://drive.google.com/file/d/1CWHutosAcJ6fiddQW5ogvg7NgLstZJ9j/view\n"
            f"and place it in the ./data/ directory."
        )

    print(f"📄 Loading knowledge base from {KB_FILE}...")
    with open(KB_FILE, "r") as f:
        knowledge_base = json.load(f)

    print(f"   Found {len(knowledge_base)} documents")

    # Convert to LangChain Document objects
    processed_docs = []
    for doc in knowledge_base:
        metadata = doc.get("metadata", {})
        data = doc.get("text", "")
        processed_docs.append(Document(page_content=data, metadata=metadata))

    print(f"📊 Indexing {len(processed_docs)} documents into ChromaDB...")
    kbase_db = Chroma.from_documents(
        documents=processed_docs,
        collection_name="knowledge_base",
        embedding=embed_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=PERSIST_DIR,
    )
    print(f"   ✅ Indexing complete! Persisted to {PERSIST_DIR}")

    return kbase_db


if __name__ == "__main__":
    # Quick test
    db = load_and_index_knowledge_base()
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.2},
    )
    results = retriever.invoke("what is your refund policy?")
    print(f"\n🔍 Test query results ({len(results)} docs):")
    for r in results:
        print(f"   - [{r.metadata.get('category', 'N/A')}] {r.page_content[:100]}...")
