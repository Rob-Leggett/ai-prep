from langchain_ollama import OllamaLLM
import chromadb
from sentence_transformers import SentenceTransformer

def query_rag(question: str):
    chroma = chromadb.PersistentClient(path="rag/chroma_store")
    collection = chroma.get_or_create_collection("legal_docs")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    q_emb = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=3)

    context = "\n\n".join([doc for doc in results["documents"][0]])
    llm = OllamaLLM(model="llama3")
    prompt = f"Answer the question based on context:\n\n{context}\n\nQuestion: {question}"
    answer = llm.invoke(prompt)
    return answer, results["metadatas"][0]