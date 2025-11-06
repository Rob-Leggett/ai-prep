from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb, os

chroma = chromadb.PersistentClient(path="rag/chroma_store")
collection = chroma.get_or_create_collection("legal_docs")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        docs = PyPDFLoader(f"data/{file}").load()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            emb = embedder.encode(chunk.page_content).tolist()
            collection.add(documents=[chunk.page_content], embeddings=[emb], metadatas=[{"source": file}])
print("Indexed all PDFs.")