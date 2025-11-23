import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"

def load_facts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def embed_facts(df: pd.DataFrame):
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(df["statement"].tolist(), show_progress_bar=True, convert_to_numpy=True)
    return emb

def save_index(embeddings, df, out_path):
    Path(out_path).parent.mkdir(exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        ids=df["id"].to_numpy(),
        statements=df["statement"].to_numpy(),
        sources=df["source"].to_numpy(),
        dates=df["date"].to_numpy(),
    )

if __name__ == "__main__":
    df = load_facts("data/facts.csv")
    embeddings = embed_facts(df)
    save_index(embeddings, df, "data/embeddings.npz")
    print("Index built successfully.")
