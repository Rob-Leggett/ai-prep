# rag/query.py
import os

from langchain_ollama import OllamaLLM

from rag import store

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))

_llm = OllamaLLM(model=OLLAMA_MODEL)


def _unique_sources(sources: list[str]) -> list[dict]:
    return [{"source": src} for src in dict.fromkeys(s or "unknown" for s in sources)]


def query_rag(question: str) -> tuple[str, list[dict]]:
    hits = store.query(store.embed([question])[0], TOP_K)

    if not hits:
        return ("I couldn’t find anything relevant in the index.", [])

    context = "\n\n".join(doc for doc, _ in hits)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # .invoke returns a string for OllamaLLM
    answer = _llm.invoke(prompt)
    return answer, _unique_sources([src for _, src in hits])
