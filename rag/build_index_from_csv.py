import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load structured data
df = pd.read_csv("data/sample.csv")

# Turn each row into text
docs = [
    f"Person age {row.age:.0f} has income {row.income:.0f} and target value {row.target:.3f}."
    for _, row in df.iterrows()
]

# Create unique IDs for each row
ids = [f"row-{i}" for i in range(len(docs))]

# Connect to Chroma
chroma = chromadb.PersistentClient(path="rag/chroma_store")
collection = chroma.get_or_create_collection("legal_docs")

# Encode & add
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(docs)

collection.add(
    ids=ids,  # 👈 new requirement
    documents=docs,
    embeddings=embeddings.tolist(),
    metadatas=[{"source": "sample.csv"} for _ in docs],
)

print(f"✅ Indexed {len(docs)} rows from sample.csv into 'legal_docs'.")