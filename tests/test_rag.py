from rag import query, store


def test_store_returns_most_similar_documents(tmp_path):
    path = str(tmp_path / "store.duckdb")
    store.add(
        ids=["a", "b", "c"],
        documents=["alpha", "beta", "gamma"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        sources=["x.csv", "y.pdf", "x.csv"],
        path=path,
    )
    # Re-adding an existing id replaces it rather than duplicating it.
    store.add(ids=["b"], documents=["beta2"], embeddings=[[0.0, 1.0]], sources=["y.pdf"], path=path)

    assert store.query([1.0, 0.1], top_k=2, path=path) == [("alpha", "x.csv"), ("gamma", "x.csv")]
    assert store.query([0.0, 1.0], top_k=1, path=path) == [("beta2", "y.pdf")]
    assert store.query([1.0, 0.0], path=path, collection="other") == []


def test_query_on_missing_store_is_empty(tmp_path):
    assert store.query([1.0], path=str(tmp_path / "missing.duckdb")) == []


def test_query_rag_builds_context_and_dedupes_sources(monkeypatch):
    monkeypatch.setattr(store, "embed", lambda texts: [[1.0, 0.0]])
    monkeypatch.setattr(
        store, "query", lambda emb, k: [("doc1", "x.csv"), ("doc2", "x.csv"), ("doc3", "y.pdf")]
    )
    prompts = []

    class FakeLLM:
        def invoke(self, prompt):
            prompts.append(prompt)
            return "answer"

    monkeypatch.setattr(query, "_llm", FakeLLM())

    answer, sources = query.query_rag("question?")

    assert answer == "answer"
    assert sources == [{"source": "x.csv"}, {"source": "y.pdf"}]
    assert "doc1\n\ndoc2\n\ndoc3" in prompts[0]


def test_query_rag_without_hits_skips_llm(monkeypatch):
    monkeypatch.setattr(store, "embed", lambda texts: [[1.0]])
    monkeypatch.setattr(store, "query", lambda emb, k: [])
    answer, sources = query.query_rag("question?")
    assert sources == []
    assert "couldn’t find" in answer
