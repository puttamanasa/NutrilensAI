# import os
# import faiss
# import pickle
# import numpy as np
# import re
# from sentence_transformers import SentenceTransformer

# # =========================
# # Paths
# # =========================
# BASE = os.path.dirname(os.path.abspath(__file__))

# index_path = os.path.join(BASE, "fndds_rag.faiss")
# docs_path = os.path.join(BASE, "fndds_docs.pkl")

# if not os.path.exists(index_path):
#     raise Exception("RAG index not found. Run build_rag_db.py first.")

# if not os.path.exists(docs_path):
#     raise Exception("RAG documents not found. Run build_rag_db.py first.")

# # =========================
# # Load Model + Index
# # =========================
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# index = faiss.read_index(index_path)

# with open(docs_path, "rb") as f:
#     docs = pickle.load(f)

# # =========================
# # Parse nutrition document
# # =========================
# def parse_doc(doc):

#     try:
#         food = re.search(r"Food:\s*(.*)", doc).group(1)

#         calories = float(re.search(r"Calories:\s*(.*)", doc).group(1))
#         protein = float(re.search(r"Protein:\s*(.*)", doc).group(1))
#         carbs = float(re.search(r"Carbs:\s*(.*)", doc).group(1))
#         fat = float(re.search(r"Fat:\s*(.*)", doc).group(1))

#     except:
#         # fallback if parsing fails
#         return {
#             "food_name": "unknown",
#             "calories": 0,
#             "protein": 0,
#             "carbs": 0,
#             "fat": 0
#         }

#     return {
#         "food_name": food,
#         "calories": calories,
#         "protein": protein,
#         "carbs": carbs,
#         "fat": fat
#     }

# # =========================
# # RAG Retrieval Function
# # =========================
# def retrieve_food_docs(query, k=3):

#     emb = embed_model.encode([query])
#     D, I = index.search(np.array(emb).astype("float32"), k)

#     results = [parse_doc(docs[i]) for i in I[0]]

#     # Nutrition fusion (average)
#     fused = {
#         "food_name": " + ".join([r["food_name"] for r in results]),
#         "calories": round(float(np.mean([r["calories"] for r in results])), 2),
#         "protein": round(float(np.mean([r["protein"] for r in results])), 2),
#         "carbs": round(float(np.mean([r["carbs"] for r in results])), 2),
#         "fat": round(float(np.mean([r["fat"] for r in results])), 2)
#     }

#     return fused

# import os
# import faiss
# import pickle
# import numpy as np
# import re
# from sentence_transformers import SentenceTransformer

# BASE = os.path.dirname(os.path.abspath(__file__))


# # =========================
# # Load DB
# # =========================
# def load_db(index_name, docs_name):
#     index = faiss.read_index(os.path.join(BASE, index_name))

#     with open(os.path.join(BASE, docs_name), "rb") as f:
#         docs = pickle.load(f)

#     return index, docs


# fndds_index, fndds_docs = load_db("fndds_rag.faiss", "fndds_docs.pkl")
# ifnd_index, ifnd_docs = load_db("ifnd_rag.faiss", "ifnd_docs.pkl")

# embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# # =========================
# # Normalize text
# # =========================
# def normalize(text):
#     return re.sub(r'[^a-z0-9 ]', '', str(text).lower()).strip()


# # =========================
# # Parse document
# # =========================
# def parse_doc(doc):
#     try:
#         food = re.search(r"Food:\s*(.*)", doc).group(1).strip()
#         calories = float(re.search(r"Calories:\s*(.*)", doc).group(1))
#         protein = float(re.search(r"Protein:\s*(.*)", doc).group(1))
#         carbs = float(re.search(r"Carbs:\s*(.*)", doc).group(1))
#         fat = float(re.search(r"Fat:\s*(.*)", doc).group(1))

#         return {
#             "food_name": food,
#             "calories": calories,
#             "protein": protein,
#             "carbs": carbs,
#             "fat": fat
#         }
#     except:
#         return None


# # =========================
# # Build lookup
# # =========================
# def build_lookup(docs):
#     lookup = {}

#     for doc in docs:
#         parsed = parse_doc(doc)
#         if parsed:
#             key = normalize(parsed["food_name"])
#             lookup[key] = parsed

#     print("✅ Sample keys:", list(lookup.keys())[:10])
#     return lookup


# fndds_lookup = build_lookup(fndds_docs)
# ifnd_lookup = build_lookup(ifnd_docs)


# # =========================
# # Matching logic
# # =========================
# def find_match(query, lookup):

#     query = normalize(query)
#     print("🔍 Searching:", query)

#     # Exact
#     if query in lookup:
#         return lookup[query]

#     # Contains
#     for key in lookup:
#         if query in key:
#             return lookup[key]

#     # Token match
#     query_tokens = set(query.split())
#     for key in lookup:
#         if len(query_tokens & set(key.split())) >= 1:
#             return lookup[key]

#     return None


# # =========================
# # MAIN FUNCTION
# # =========================
# def retrieve_food_docs(query):

#     print("\n====================")
#     print("QUERY:", query)

#     # 1️⃣ FNDDS
#     res = find_match(query, fndds_lookup)
#     if res:
#         print("✅ FROM FNDDS")
#         return res

#     # 2️⃣ IFND
#     res = find_match(query, ifnd_lookup)
#     if res:
#         print("✅ FROM IFND")
#         return res

#     print("❌ NOT FOUND")

#     return {
#         "food_name": "Not found",
#         "calories": 0,
#         "protein": 0,
#         "carbs": 0,
#         "fat": 0
#     }

# rag_retriever.py  —  Dual FAISS + lookup food retrieval

import os, faiss, pickle, re, numpy as np
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))

_embed_model  = None
_fndds_index  = _ifnd_index = None
_fndds_docs   = _ifnd_docs  = None
_fndds_lookup = _ifnd_lookup = None


def _load():
    global _embed_model, _fndds_index, _ifnd_index
    global _fndds_docs, _ifnd_docs, _fndds_lookup, _ifnd_lookup

    if _embed_model is not None:
        return

    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_db(idx_name, doc_name):
        idx  = faiss.read_index(os.path.join(BASE, idx_name))
        with open(os.path.join(BASE, doc_name), "rb") as f:
            docs = pickle.load(f)
        return idx, docs

    _fndds_index, _fndds_docs = load_db("fndds_rag.faiss", "fndds_docs.pkl")
    _ifnd_index,  _ifnd_docs  = load_db("ifnd_rag.faiss",  "ifnd_docs.pkl")

    _fndds_lookup = _build_lookup(_fndds_docs)
    _ifnd_lookup  = _build_lookup(_ifnd_docs)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower()).strip()


def _parse_doc(doc: str) -> dict | None:
    try:
        food     = re.search(r"Food:\s*(.*)",     doc).group(1).strip()
        calories = float(re.search(r"Calories:\s*(.*)", doc).group(1))
        protein  = float(re.search(r"Protein:\s*(.*)",  doc).group(1))
        carbs    = float(re.search(r"Carbs:\s*(.*)",    doc).group(1))
        fat      = float(re.search(r"Fat:\s*(.*)",      doc).group(1))
        return {"food_name": food, "calories": calories,
                "protein": protein, "carbs": carbs, "fat": fat, "portion": 100}
    except Exception:
        return None


def _build_lookup(docs):
    lookup = {}
    for doc in docs:
        p = _parse_doc(doc)
        if p:
            lookup[_normalize(p["food_name"])] = p
    return lookup


def _find_match(query: str, lookup: dict) -> dict | None:
    q = _normalize(query)
    if q in lookup:
        return lookup[q]
    for key in lookup:
        if q in key:
            return lookup[key]
    q_tokens = set(q.split())
    for key in lookup:
        if q_tokens & set(key.split()):
            return lookup[key]
    return None


def _faiss_search(query: str, index, docs) -> dict | None:
    vec = _embed_model.encode([query])
    _, I = index.search(np.array(vec).astype("float32"), 1)
    if I[0].size:
        return _parse_doc(docs[I[0][0]])
    return None


def retrieve_food_docs(query: str) -> dict:
    _load()

    for lookup in (_fndds_lookup, _ifnd_lookup):
        res = _find_match(query, lookup)
        if res:
            return res

    for idx, docs in ((_fndds_index, _fndds_docs), (_ifnd_index, _ifnd_docs)):
        res = _faiss_search(query, idx, docs)
        if res:
            return res

    return {"food_name": query, "calories": 0,
            "protein": 0, "carbs": 0, "fat": 0, "portion": 100}