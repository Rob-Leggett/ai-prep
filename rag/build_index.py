import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag import store

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        reader = PdfReader(f"data/{file}")
        for page_no, page in enumerate(reader.pages):
            chunks = splitter.split_text(page.extract_text() or "")
            if not chunks:
                continue
            store.add(
                ids=[f"{file}-p{page_no}-c{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=store.embed(chunks),
                sources=[file] * len(chunks),
            )
print("Indexed all PDFs.")
