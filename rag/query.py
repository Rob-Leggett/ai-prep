# rag/query.py
import os
from typing import List, Dict, Tuple
from chromadb import PersistentClient
from langchain_ollama import OllamaLLM

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
CHROMA_PATH = os.getenv("CHROMA_PATH", "rag/chroma_store")
COLLECTION = os.getenv("CHROMA_COLLECTION", "legal_docs")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))

_llm = OllamaLLM(model=OLLAMA_MODEL)  # <- instantiate

def _unique_sources(mds: List[Dict]) -> List[Dict]:
    seen = {}
    for md in mds:
        src = md.get("source", "unknown")
        if src not in seen:
            seen[src] = {"source": src}
    return list(seen.values())

def query_rag(question: str) -> Tuple[str, List[Dict]]:
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION)
    res = collection.query(query_texts=[question], n_results=TOP_K)

    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []

    if not docs:
        return ("I couldn’t find anything relevant in the index.", [])

    context = "\n\n".join(docs)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # .invoke returns a string for OllamaLLM
    answer = _llm.invoke(prompt)
    sources = _unique_sources(metas)
    return answer, sources