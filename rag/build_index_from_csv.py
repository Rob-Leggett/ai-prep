import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

CSV_PATH = os.getenv("CSV_PATH", "data/sample.csv")
CHROMA_PATH = os.getenv("CHROMA_PATH", "rag/chroma_store")
COLLECTION = os.getenv("CHROMA_COLLECTION", "legal_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"

df = pd.read_csv(CSV_PATH)
if df.empty:
    raise SystemExit("CSV has no rows.")

docs = [
    f"Person age {row.age:.0f} has income {row.income:.0f} and target value {row.target:.3f}."
    for _, row in df.iterrows()
]
ids = [f"row-{i}" for i in range(len(docs))]

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(COLLECTION)

embedder = SentenceTransformer(EMBED_MODEL)  # device auto-detect; set with SENTENCE_TRANSFORMERS_HOME if needed
embeddings = embedder.encode(docs, show_progress_bar=False).tolist()

collection.add(
    ids=ids,
    documents=docs,
    embeddings=embeddings,
    metadatas=[{"source": os.path.basename(CSV_PATH)} for _ in docs],
)

print(f"✅ Indexed {len(docs)} rows from {CSV_PATH} into '{COLLECTION}'.")