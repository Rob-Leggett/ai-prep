import os

import pandas as pd

from rag import store

CSV_PATH = os.getenv("CSV_PATH", "data/sample.csv")

assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"

df = pd.read_csv(CSV_PATH)
if df.empty:
    raise SystemExit("CSV has no rows.")

docs = [
    f"Person age {row.age:.0f} has income {row.income:.0f} and target value {row.target:.3f}."
    for _, row in df.iterrows()
]
ids = [f"row-{i}" for i in range(len(docs))]

store.add(
    ids=ids,
    documents=docs,
    embeddings=store.embed(docs),
    sources=[os.path.basename(CSV_PATH)] * len(docs),
)

print(f"✅ Indexed {len(docs)} rows from {CSV_PATH} into '{store.COLLECTION}'.")
