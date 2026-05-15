#build_medical_db.py
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

df = pd.read_excel("disease_diet_dataset.xlsx")
model = SentenceTransformer("all-MiniLM-L6-v2")
docs = []
embeddings = []

for _, row in df.iterrows():
    text = f"Disease: {row['Disease']}. Eat: {row['Food to be Consumed']}. Avoid: {row['Food NOT to be Consumed']}"
    docs.append(
        {
            "disease": row["Disease"],
            "eat": row["Food to be Consumed"],
            "avoid": row["Food NOT to be Consumed"],
        }
    )
    embeddings.append(model.encode(text))

embeddings = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "medical.faiss")

with open("medical.pkl", "wb") as f:
    pickle.dump(docs, f)

print("Medical RAG built ✅")