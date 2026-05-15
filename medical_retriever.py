import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("medical.faiss")

with open("medical.pkl", "rb") as f:
    docs = pickle.load(f)

def get_medical_advice(disease):
    emb = model.encode([disease])
    D, I = index.search(np.array(emb).astype("float32"), 1)

    return docs[I[0][0]]