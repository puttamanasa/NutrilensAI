# import pandas as pd
# import faiss
# import numpy as np
# import pickle
# from sentence_transformers import SentenceTransformer

# df = pd.read_csv("fndds.xlsx")

# documents = []
# texts = []

# for _, row in df.iterrows():

#     text = f"""
#     Food: {row['food_description']}
#     Calories: {row['energy_kcal']}
#     Protein: {row['protein_g']}
#     Carbs: {row['carbs_g']}
#     Fat: {row['fat_g']}
#     """

#     documents.append(text)
#     texts.append(row["food_description"])

# model = SentenceTransformer("all-MiniLM-L6-v2")

# embeddings = model.encode(texts)

# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(np.array(embeddings).astype("float32"))

# faiss.write_index(index, "fndds_rag.faiss")

# with open("fndds_docs.pkl", "wb") as f:
#     pickle.dump(documents, f)

# print("RAG DB built successfully")

import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def build_db(file_path, file_type="fndds"):

    # =========================
    # Load file safely
    # =========================
    try:
        if file_path.endswith(".xlsx"):
            print(f"📂 Loading Excel: {file_path}")
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            print(f"📂 Loading CSV: {file_path}")
            df = pd.read_csv(file_path)
    except:
        print("⚠️ Using robust CSV loader...")
        df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")

    print("Columns:", df.columns)
    print("Total rows:", len(df))

    documents = []
    texts = []

    for _, row in df.iterrows():
        try:
            # -------- FNDDS --------
            if file_type == "fndds":
                food = str(row["food_description"])
                calories = float(row["energy_kcal"])
                protein = float(row["protein_g"])
                carbs = float(row["carbs_g"])
                fat = float(row["fat_g"])

            # -------- IFND (YOUR DATASET) --------
            else:
                food = str(row["Food Name"])
                calories = float(row["Calories (kcal)"])
                protein = float(row["Protein (g)"])
                carbs = float(row["Carbohydrates (g)"])
                fat = float(row["Fats (g)"])

            text = f"""
            Food: {food}
            Calories: {calories}
            Protein: {protein}
            Carbs: {carbs}
            Fat: {fat}
            """

            documents.append(text)
            texts.append(food)

        except:
            continue

    print(f"✅ Processed {len(documents)} documents")

    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    return index, documents


# =========================
# BUILD FNDDS
# =========================
fndds_index, fndds_docs = build_db("fndds.xlsx", "fndds")
faiss.write_index(fndds_index, "fndds_rag.faiss")

with open("fndds_docs.pkl", "wb") as f:
    pickle.dump(fndds_docs, f)

print("✅ FNDDS DB built")


# =========================
# BUILD IFND
# =========================
ifnd_index, ifnd_docs = build_db("ifnd_dataset.csv", "ifnd")
faiss.write_index(ifnd_index, "ifnd_rag.faiss")

with open("ifnd_docs.pkl", "wb") as f:
    pickle.dump(ifnd_docs, f)

print("✅ IFND DB built")

print("🎉 DONE")