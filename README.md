<div align="center">

# 🥗 NutriLens AI

### *See your food. Know your nutrition. Eat smarter.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge)
![CLIP](https://img.shields.io/badge/CLIP-Fine--tuned-412991?style=for-the-badge&logo=openai&logoColor=white)
![LLaMA](https://img.shields.io/badge/LLaMA-LLM-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Vector+DB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)

<br/>

> An end-to-end multimodal AI system that identifies food items from images (including Indian cuisine), retrieves detailed nutritional information via embedding-based vector search, and delivers personalised dietary recommendations through an LLM-powered assistant.

<br/>

**🏆 Project Expo Winner — Keshav Memorial Engineering College**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-system-architecture)
- [Pipeline](#-recognition-pipeline)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-datasets)
- [Installation](#-installation)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Future Work](#-future-work)

---

## 🔍 Overview

NutriLens AI solves a real problem: **most nutrition trackers fail on Indian food**. Existing solutions are trained on Western datasets and can't recognise dal, biryani, or dosas — let alone provide accurate macros for them.

NutriLens combines:
- **Computer vision** (YOLOv8) for multi-object food detection
- **Contrastive vision-language understanding** (fine-tuned CLIP) for food identification, including Indian cuisine
- **Embedding-based vector search** over FNDDS & IFND nutritional datasets
- **LLaMA LLM** for personalised, conversational dietary recommendations

---

## 🏗 System Architecture

```
User Image
    │
    ▼
┌─────────────────────┐
│  Stage 1: CLIP      │  ← Binary food / non-food filter
│  Food Filter        │     (zero-shot contrastive inference)
└────────┬────────────┘
         │ (food detected)
         ▼
┌─────────────────────┐
│  Stage 2: YOLOv8    │  ← Multi-object detection
│  Object Detection   │     Returns bounding boxes per item
└────────┬────────────┘
         │ (cropped objects)
         ▼
┌─────────────────────────────┐
│  Stage 3: Fine-tuned CLIP   │  ← Trained on Food-20 dataset
│  Food Labelling             │     Identifies Indian + global food
└────────┬────────────────────┘
         │ (food label embeddings)
         ▼
┌─────────────────────────────┐
│  Vector DB Similarity Search│  ← FNDDS + IFND as embeddings
│  Nutrition Retrieval        │     Cosine similarity matching
└────────┬────────────────────┘
         │ (nutritional profile)
         ▼
┌─────────────────────┐
│  LLaMA LLM          │  ← RAG-style recommendations
│  Recommendations    │     Personalised dietary coaching
└────────┬────────────┘
         │
         ▼
    FastAPI Backend  ──►  MongoDB (users, meals, history)
```

---

## 🔬 Recognition Pipeline

### Stage 1 — CLIP Food Filter
Before running expensive object detection, every image is first passed through CLIP with prompts like `"a photo of food"` vs `"a photo of something other than food"`. Non-food images are rejected early, reducing unnecessary compute.

### Stage 2 — YOLOv8 Object Detection
YOLOv8 detects and localises all food items in the image, producing bounding box crops for each detected object. This handles multi-item plates (e.g. a thali with rice, dal, and sabzi as separate detections).

### Stage 3 — Fine-tuned CLIP for Food Labelling
Each crop is passed through CLIP fine-tuned on the **Food-20 dataset**, which includes Indian cuisine categories. The model produces an embedding for each crop, which is then matched via **cosine similarity** against pre-indexed food label embeddings — returning the closest food label, including Indian items like biryani, idli, dosa, paneer dishes, etc.

### Stage 4 — Vector DB Nutrition Retrieval
Identified food labels are converted to embeddings and searched against a **vector database** built from the FNDDS (US) and IFND (Indian) nutritional datasets. This retrieves per-item macro and micronutrient profiles (calories, protein, carbs, fats, fibre, vitamins, minerals).

### Stage 5 — LLaMA Recommendations
The retrieved nutritional context is passed to **LLaMA** which generates personalised dietary recommendations, answers nutrition Q&A, and provides conversational coaching based on the user's BMI, goals, and meal history.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🍛 Indian Food Support | Fine-tuned CLIP on Food-20 dataset for Indian cuisine recognition |
| 🔍 Multi-item Detection | YOLOv8 detects multiple food items in a single plate image |
| 🧬 Embedding-based Nutrition | Vector similarity search over FNDDS & IFND datasets |
| 🤖 AI Recommendations | LLaMA-powered personalised dietary coaching |
| 📊 Nutrition Dashboard | Per-meal macro/micronutrient breakdowns and analytics |
| ⚖️ BMI Tracking | Tracks BMI over time with trend visualisation |
| 👤 User Profiles | Persistent meal history and dietary preferences via MongoDB |
| ⚡ REST API | Full FastAPI backend with documented endpoints |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Vision-Language Model | CLIP (OpenAI, fine-tuned on Food-20) |
| LLM | LLaMA (Meta) |
| Vector Search | Vector DB (embedding similarity) |
| Backend | FastAPI |
| Database | MongoDB |
| ML Libraries | PyTorch, Hugging Face Transformers, Scikit-learn |
| Language | Python 3.10+ |

---

## 📦 Datasets

| Dataset | Source | Usage |
|---|---|---|
| **FNDDS** | USDA Food and Nutrient Database for Dietary Studies | Nutritional profiles for global foods |
| **IFND** | Indian Food Nutrient Database | Nutritional profiles for Indian cuisine |
| **Food-20** | Custom curated | CLIP fine-tuning for Indian food recognition |

> FNDDS and IFND records are converted to text embeddings and indexed in a vector database for fast similarity-based retrieval.

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/puttamanasa1/nutrilens-ai.git
cd nutrilens-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Fill in: MONGODB_URI, LLAMA_MODEL_PATH, VECTOR_DB_PATH

# 5. Run the FastAPI server
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API docs.

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/detect` | Upload image → food detection + nutrition |
| `POST` | `/recommend` | Get LLaMA-powered dietary recommendations |
| `GET` | `/nutrition/{food_name}` | Retrieve nutritional profile by food name |
| `POST` | `/user/bmi` | Log BMI entry |
| `GET` | `/user/dashboard` | Fetch analytics and meal history |
| `POST` | `/user/profile` | Create/update user profile |

Full docs available at `/docs` (Swagger UI) after running the server.

---

## 📁 Project Structure

```
nutrilens-ai/
├── app/
│   ├── main.py               # FastAPI app entry point
│   ├── routes/               # API endpoint handlers
│   ├── models/               # Pydantic schemas
│   ├── services/
│   │   ├── clip_filter.py    # Stage 1: food/non-food filter
│   │   ├── yolo_detector.py  # Stage 2: object detection
│   │   ├── clip_labeller.py  # Stage 3: fine-tuned CLIP labelling
│   │   ├── vector_search.py  # Stage 4: nutrition retrieval
│   │   └── llama_advisor.py  # Stage 5: LLM recommendations
│   └── db/
│       ├── mongodb.py        # User data & meal history
│       └── vector_db.py      # Embedding index (FNDDS + IFND)
├── data/
│   ├── fndds/                # FNDDS dataset + embeddings
│   ├── ifnd/                 # IFND dataset + embeddings
│   └── food20/               # Food-20 fine-tuning data
├── models/
│   ├── clip_finetuned/       # Fine-tuned CLIP weights
│   └── yolov8/               # YOLOv8 weights
├── notebooks/                # Training & evaluation notebooks
├── requirements.txt
└── .env.example
```

---

## 📈 Results

- **Food Detection**: YOLOv8 accurately localises multiple food items per image including mixed-plate Indian meals
- **Indian Food Recognition**: Fine-tuned CLIP on Food-20 significantly improves labelling accuracy for Indian cuisine vs. zero-shot baseline
- **Nutrition Retrieval**: Embedding similarity search retrieves correct nutritional profiles across FNDDS & IFND datasets
- **🏆 Won Project Expo** at Keshav Memorial Engineering College

---

## 🔮 Future Work

- [ ] Mobile app (React Native) for on-device food scanning
- [ ] Expand Food-20 to Food-101 with more Indian categories
- [ ] Real-time video stream food detection
- [ ] Calorie goal setting and weekly meal planning
- [ ] Integration with fitness trackers (steps, activity)
- [ ] Multi-language support for regional Indian languages

---

## 👩‍💻 Author

**Putta Manasa**
B.Tech CSE (AI & ML) — Keshav Memorial Engineering College

[![LinkedIn](https://www.linkedin.com/in/manasa-putta-64541437b?utm_source=share_via&utm_content=profile&utm_medium=member_android)


---

<div align="center">
<sub>Built with ❤️ and a lot of late nights | If this helped you, give it a ⭐</sub>
</div>
