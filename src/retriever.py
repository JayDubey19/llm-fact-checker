import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)

def load_index(path="data/embeddings.npz"):
    data = np.load(path, allow_pickle=True)
    return data

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

def retrieve_top_k(claim: str, k: int = 5):
    idx = load_index()
    emb_db = idx["embeddings"]
    ids = idx["ids"]
    stmts = idx["statements"]
    srcs = idx["sources"]
    dates = idx["dates"]

    q_vec = _model.encode([claim], convert_to_numpy=True)[0]

    sims = [(i, cosine(q_vec, v)) for i, v in enumerate(emb_db)]
    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:k]

    results = []
    for i, sim in top:
        results.append({
            "id": int(ids[i]),
            "statement": str(stmts[i]),
            "source": str(srcs[i]),
            "date": str(dates[i]),
            "similarity": float(sim)
        })
    return results
