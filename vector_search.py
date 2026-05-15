import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

index = faiss.read_index(os.path.join(BASE_DIR, "food_index.faiss"))

with open(os.path.join(BASE_DIR, "food_metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

with open("food_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def search_food(query, k=3):

    emb = embed_model.encode([query])
    D, I = index.search(np.array(emb).astype("float32"), k)

    results = [metadata[i] for i in I[0]]

    # ⭐ Nutrition fusion (average)
    avg = {
        "food_name": " + ".join([r["food_name"] for r in results]),
        "calories": np.mean([r["calories"] for r in results]),
        "protein": np.mean([r["protein"] for r in results]),
        "carbs": np.mean([r["carbs"] for r in results]),
        "fat": np.mean([r["fat"] for r in results])
    }

    return avg