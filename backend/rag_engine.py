"""
RAG Engine — builds a FAISS vector index over the interview corpus
using IBM watsonx.ai cloud embeddings, and retrieves the top-k most 
relevant Q&A entries for a given query.
"""

import json
import os
import numpy as np

# Lazy-loaded to avoid slow imports at startup
_embeddings_client = None
_index = None
_corpus = []


def _get_corpus():
    global _corpus
    if not _corpus:
        corpus_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "interview_corpus.json"
        )
        with open(corpus_path, "r", encoding="utf-8") as f:
            _corpus = json.load(f)
    return _corpus


def _build_index(corpus: list[dict]):
    """Encode all corpus entries via watsonx.ai and build a FAISS index."""
    import faiss
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import Embeddings

    WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY") or os.getenv("WATSONX_API_kEY") or ""
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")

    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        raise EnvironmentError("Watsonx credentials are required to build the index.")

    credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    client = Embeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID
    )

    texts = [
        f"{entry['role']} | {entry['category']} | {entry['question']}"
        for entry in corpus
    ]
    
    # Generate embeddings via watsonx.ai cloud API
    embeddings_list = client.embed_documents(texts=texts)
    embeddings = np.array(embeddings_list).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    return index, client


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve the top_k most relevant corpus entries for the given query.
    Bypasses vector search and returns first top_k elements if credentials are not set.
    """
    corpus = _get_corpus()
    if not corpus:
        return []

    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY") or os.getenv("WATSONX_API_kEY") or ""
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")

    # Fallback: if credentials are not configured, run in Demo Mode
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        # Return first top_k items from the corpus as a mock retrieve result
        return [dict(entry, _score=0.0) for entry in corpus[:min(top_k, len(corpus))]]

    global _index, _embeddings_client
    try:
        if _index is None:
            _index, _embeddings_client = _build_index(corpus)

        # Generate query embedding
        query_vec = _embeddings_client.embed_documents(texts=[query])[0]
        query_vec = np.array([query_vec]).astype("float32")

        import faiss  # noqa: F401
        distances, indices = _index.search(query_vec, min(top_k, len(corpus)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(corpus[idx])
            entry["_score"] = float(dist)
            results.append(entry)
        return results

    except Exception as exc:
        print(f"[RAG Engine Error: {exc}]. Falling back to mock retrieve.")
        # Graceful fallback: return top_k items
        return [dict(entry, _score=999.0) for entry in corpus[:min(top_k, len(corpus))]]
