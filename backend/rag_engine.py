"""
RAG Engine — builds a FAISS vector index over the interview corpus
and retrieves the top-k most relevant Q&A entries for a given query.
"""

import json
import os
import numpy as np

# Lazy-loaded to avoid slow imports at startup
_model = None
_index = None
_corpus = []


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _build_index(corpus: list[dict]):
    """Encode all corpus entries and build a FAISS flat-L2 index."""
    import faiss

    model = _load_model()
    texts = [
        f"{entry['role']} | {entry['category']} | {entry['question']}"
        for entry in corpus
    ]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def get_engine():
    """Return (index, corpus), building them on first call."""
    global _index, _corpus

    if _index is not None:
        return _index, _corpus

    corpus_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "interview_corpus.json"
    )
    with open(corpus_path, "r", encoding="utf-8") as f:
        _corpus = json.load(f)

    _index = _build_index(_corpus)
    return _index, _corpus


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve the top_k most relevant corpus entries for the given query.

    Parameters
    ----------
    query   : natural language query (e.g. role + topic description)
    top_k   : number of results to return

    Returns
    -------
    List of corpus dicts sorted by relevance (most relevant first).
    """
    import faiss  # noqa: F401 — ensures faiss available

    index, corpus = get_engine()
    model = _load_model()

    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, min(top_k, len(corpus)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        entry = dict(corpus[idx])
        entry["_score"] = float(dist)
        results.append(entry)
    return results
