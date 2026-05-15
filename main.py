#main.py
"""
NutriLens — FastAPI Backend
Replaces the Streamlit app.py with a proper REST API.
Run with:  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import io
from PIL import Image

# ── Your existing modules (unchanged) ──────────────────────────────────────
from blip_model import process_image
from rag_retriever import retrieve_food_docs
from gemini_recommender import rag_diet_reasoning, weekly_diet_recommendation
from portion_estimator import estimate_portion
from mongo_db import register_user, login_user, save_food_history, get_food_history
from medical_retriever import get_medical_advice

# ──────────────────────────────────────────────────────────────────────────
app = FastAPI(title="NutriLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend HTML from the same directory
# app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pydantic models ────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class LogMealRequest(BaseModel):
    username: str
    food_name: str
    caption: str
    calories: float
    protein: float
    carbs: float
    fat: float
    portion: float

class WeeklyInsightRequest(BaseModel):
    username: str


# ── Helper functions (ported from app.py) ─────────────────────────────────

FOOD_EMOJI_MAP = {
    "salad": "🥗", "caesar": "🥗", "biryani": "🍚", "rice": "🍚",
    "curry": "🍛", "chicken": "🍗", "egg": "🥚", "scrambled": "🥚",
    "banana": "🍌", "mango": "🥭", "lassi": "🥛", "pizza": "🍕",
    "burger": "🍔", "pasta": "🍝", "noodles": "🍜", "dosa": "🫓",
    "idli": "⚪", "samosa": "🥟", "paneer": "🧀", "dal": "🥣",
    "roti": "🫓", "naan": "🫓", "cake": "🍰", "ice cream": "🍨",
    "chole": "🥘", "bhature": "🥘", "puri": "🫓",
}

def get_food_emoji(food_name: str) -> str:
    food_lower = food_name.lower()
    for key, emoji in FOOD_EMOJI_MAP.items():
        if key in food_lower:
            return emoji
    return "🍽️"

def get_status_for_food(food_name: str, disease: str, nutrition: dict) -> str:
    if disease == "None":
        return "safe"
    if disease == "Diabetes" and nutrition.get("carbs", 0) > 50:
        return "caution"
    elif disease == "Hypertension" and nutrition.get("fat", 0) > 20:
        return "caution"
    elif disease == "Obesity" and nutrition.get("calories", 0) > 400:
        return "caution"
    return "safe"

def scale_nutrition(base_nut: dict, base_portion: float, current_portion: float) -> dict:
    scale = current_portion / max(base_portion, 1)
    return {
        "calories": round(base_nut["calories"] * scale),
        "protein":  round(base_nut["protein"]  * scale),
        "carbs":    round(base_nut["carbs"]    * scale),
        "fat":      round(base_nut["fat"]      * scale),
    }

def build_tags(nut: dict) -> list:
    tags = []
    if nut["carbs"] < 30:
        tags.append({"label": "Low glycemic index", "color": "green"})
    if nut["fat"] < 15:
        tags.append({"label": "Low fat", "color": "green"})
    elif nut["fat"] < 25:
        tags.append({"label": "Moderate fat", "color": "yellow"})
    if nut["protein"] > 10:
        tags.append({"label": "Good protein", "color": "green"})
    return tags


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


# Auth
@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    success, msg = register_user(req.username, req.password)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    success, msg = login_user(req.username, req.password)
    if not success:
        raise HTTPException(status_code=401, detail=msg)
    return {"username": req.username, "message": msg}


# Food Analysis
@app.post("/api/analyze")
async def analyze_food(
    file: UploadFile = File(...),
    disease: str = Form(default="None"),
    username: str = Form(...),
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = process_image(image)

    if result["status"] == "not_food":
        raise HTTPException(status_code=422, detail="This doesn't appear to be a food image.")

    # De-duplicate multi-food detections (same logic as app.py)
    raw_foods = result.get("foods", [])
    unique_foods = {}
    for f in raw_foods:
        label = f.get("label", "").lower().strip()
        conf  = f.get("confidence", 0)
        if not label:
            continue
        if label not in unique_foods or conf > unique_foods[label]["confidence"]:
            unique_foods[label] = {"label": label, "confidence": conf, "caption": f.get("caption", label),"box": f.get("box")}

    multi_foods = sorted(unique_foods.values(), key=lambda x: x["confidence"], reverse=True)[:5]

    estimated_portion = estimate_portion(image)
    medical           = get_medical_advice(disease)

    # per_food_data = []
    # for idx, fd in enumerate(multi_foods):
    #     fd_caption    = fd.get("caption", fd["label"])
    #     fd_nut_base   = retrieve_food_docs(fd_caption)
    #     fd_base_portion = max(fd_nut_base.get("portion", 100), 1)
    #     fd_portion    = min(max(round(estimated_portion / max(len(multi_foods), 1)), 25), 500)
    #     fd_scale      = fd_portion / fd_base_portion
    #     fd_nut_scaled = {
    #         "calories": round(fd_nut_base["calories"] * fd_scale),
    #         "protein":  round(fd_nut_base["protein"]  * fd_scale),
    #         "carbs":    round(fd_nut_base["carbs"]    * fd_scale),
    #         "fat":      round(fd_nut_base["fat"]      * fd_scale),
    #     }
    #     ai_rec = rag_diet_reasoning(fd["label"], fd_nut_scaled, medical)
    #     per_food_data.append({
    #         "index":          idx,
    #         "label":          fd["label"],
    #         "caption":        fd_caption,
    #         "confidence":     fd["confidence"],
    #         "emoji":          get_food_emoji(fd["label"]),
    #         "base_nutrition": fd_nut_base,
    #         "base_portion":   fd_base_portion,
    #         "detected_portion": fd_portion,
    #         "nutrition":      fd_nut_scaled,
    #         "ai_rec":         ai_rec,
    #         "tags":           build_tags(fd_nut_scaled),
    #         "status":         get_status_for_food(fd["label"], disease, fd_nut_scaled),
    #     })

    # return {
    #     "primary_food":    result["primary"],
    #     "estimated_portion": estimated_portion,
    #     "disease":         disease,
    #     "count":           result.get("count", 1),
    #     "tier":            result.get("tier", 2),
    #     "per_food_data":   per_food_data,
    # }

# ─────────────────────────────
# Calculate area-based portion
# ─────────────────────────────
    # ── Pre-compute all box areas so ratios always sum to 1.0 ──────────────────
    image_area = image.width * image.height

    def get_box(fd, image):
        box = fd.get("box")
        if not box or len(box) != 4:
            return [0, 0, image.width, image.height]
        return box

    def box_area(box, image):
        x1, y1, x2, y2 = box
        return max((x2 - x1) * (y2 - y1), 1)

    boxes        = [get_box(fd, image) for fd in multi_foods]
    areas        = [box_area(b, image) for b in boxes]

# Detect CLIP fallback: all boxes are identical full-image boxes
    all_identical = len(set(map(tuple, boxes))) == 1

    if all_identical:
    # Equal split — avoids N×100% problem
        portions = [round(estimated_portion / len(multi_foods))] * len(multi_foods)
    else:
    # Normalize by sum of actual box areas so portions sum to estimated_portion
        total_area = sum(areas)
        portions   = [round(estimated_portion * (a / total_area)) for a in areas]

# Clamp each portion to [25, 500]
    portions = [min(max(p, 25), 500) for p in portions]

# ── Build per-food data ────────────────────────────────────────────────────
    per_food_data = []

    for idx, fd in enumerate(multi_foods):
        fd_caption      = fd.get("caption", fd["label"])
        fd_nut_base     = retrieve_food_docs(fd_caption)
        fd_base_portion = max(fd_nut_base.get("portion", 100), 1)
        fd_portion      = portions[idx]

        fd_scale = fd_portion / fd_base_portion

        fd_nut_scaled = {
            "calories": round(fd_nut_base["calories"] * fd_scale),
            "protein":  round(fd_nut_base["protein"]  * fd_scale),
            "carbs":    round(fd_nut_base["carbs"]    * fd_scale),
            "fat":      round(fd_nut_base["fat"]      * fd_scale),
    }

        ai_rec = rag_diet_reasoning(fd["label"], fd_nut_scaled, medical)

        per_food_data.append({
            "index":            idx,
            "label":            fd["label"],
            "caption":          fd_caption,
            "confidence":       fd["confidence"],
            "emoji":            get_food_emoji(fd["label"]),
            "base_nutrition":   fd_nut_base,
            "base_portion":     fd_base_portion,
            "detected_portion": fd_portion,
            "nutrition":        fd_nut_scaled,
            "ai_rec":           ai_rec,
            "tags":             build_tags(fd_nut_scaled),
            "status":           get_status_for_food(fd["label"], disease, fd_nut_scaled),
    })
    #image_area = image.width * image.height

    # per_food_data = []

    # for idx, fd in enumerate(multi_foods):

    #     fd_caption = fd.get("caption", fd["label"])
    #     fd_nut_base = retrieve_food_docs(fd_caption)

    #     fd_base_portion = max(fd_nut_base.get("portion", 100), 1)

    # # 🔥 AREA-BASED PORTION
    #     # box = fd.get("box", [0, 0, image.width, image.height])
    #     # x1, y1, x2, y2 = box
    #     box = fd.get("box")

    #     if not box or len(box) != 4:
    #         x1, y1, x2, y2 = 0, 0, image.width, image.height
    #     else:
    #         x1, y1, x2, y2 = box
    #     all_same_box = (x1 == 0 and y1 == 0 and x2 == image.width and y2 == image.height)
    #     if all_same_box or len(multi_foods) > 2:
    #         fd_portion = round(estimated_portion / len(multi_foods))
    #     else:
    #         box_area = max((x2 - x1) * (y2 - y1), 1)
    #         portion_ratio = box_area / image_area

    #         fd_portion = round(estimated_portion * portion_ratio)

    # # clamp
    #     fd_portion = min(max(fd_portion, 25), 500)

    # # scale nutrition
    #     fd_scale = fd_portion / fd_base_portion

    #     fd_nut_scaled = {
    #         "calories": round(fd_nut_base["calories"] * fd_scale),
    #         "protein":  round(fd_nut_base["protein"]  * fd_scale),
    #         "carbs":    round(fd_nut_base["carbs"]    * fd_scale),
    #         "fat":      round(fd_nut_base["fat"]      * fd_scale),
    # }

    #     ai_rec = rag_diet_reasoning(fd["label"], fd_nut_scaled, medical)

    #     per_food_data.append({
    #         "index": idx,
    #         "label": fd["label"],
    #         "caption": fd_caption,
    #         "confidence": fd["confidence"],
    #         "emoji": get_food_emoji(fd["label"]),
    #         "base_nutrition": fd_nut_base,
    #         "base_portion": fd_base_portion,
    #         "detected_portion": fd_portion,
    #         "nutrition": fd_nut_scaled,
    #         "ai_rec": ai_rec,
    #         "tags": build_tags(fd_nut_scaled),
    #         "status": get_status_for_food(fd["label"], disease, fd_nut_scaled),
    # })
    return {
        "primary_food":    result["primary"],
        "estimated_portion": estimated_portion,
        "disease":         disease,
        "count":           result.get("count", 1),
        "tier":            result.get("tier", 2),
        "per_food_data":   per_food_data,
    }
# Dynamic nutrition rescaling endpoint (called when user moves the portion slider)
@app.get("/api/scale")
async def scale_nutrition_endpoint(
    base_calories: float,
    base_protein:  float,
    base_carbs:    float,
    base_fat:      float,
    base_portion:  float,
    current_portion: float,
    disease: str = "None",
    food_name: str = "",
):
    base_nut = {"calories": base_calories, "protein": base_protein,
                "carbs": base_carbs, "fat": base_fat}
    scaled = scale_nutrition(base_nut, base_portion, current_portion)
    return {
        **scaled,
        "tags":   build_tags(scaled),
        "status": get_status_for_food(food_name, disease, scaled),
    }


# Meal logging
@app.post("/api/meals/log")
async def log_meal(req: LogMealRequest):
    save_food_history(
        req.username, req.food_name, req.caption,
        req.calories, req.protein, req.carbs, req.fat, req.portion
    )
    return {"message": "Meal logged successfully"}


# History & analytics
@app.get("/api/meals/history/{username}")
async def get_history(username: str):
    data = get_food_history(username)
    enriched = []
    for row in data:
        food_name = row.get("food_name") or row.get("food") or "Unknown"
        enriched.append({
            **row,
            "food_name": food_name,
            "emoji":     get_food_emoji(food_name),
        })
    return enriched


@app.get("/api/meals/analytics/{username}")
async def get_analytics(username: str, disease: str = "None"):
    import pandas as pd
    data = get_food_history(username)
    if not data:
        return {"empty": True}

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "food_name" not in df.columns and "food" in df.columns:
        df["food_name"] = df["food"]

    # Stats
    avg_calories = round(df["calories"].mean())
    avg_protein  = round(df["protein"].mean())
    avg_fat      = round(df["fat"].mean())
    avg_carbs    = round(df["carbs"].mean())
    total_meals  = len(df)
    score        = min(100, int((avg_protein * 2) - avg_fat + 50))

    # Last 7 days bar chart
    df_recent = df.tail(7).copy()
    df_recent["day"] = df_recent["timestamp"].dt.strftime("%a")
    calorie_trend = df_recent[["day", "calories"]].to_dict("records")

    # Macro totals for donut
    macros = {
        "carbs":   int(df["carbs"].sum()),
        "protein": int(df["protein"].sum()),
        "fat":     int(df["fat"].sum()),
    }

    # Daily trend
    daily_df = df.copy()
    daily_df["date"] = daily_df["timestamp"].dt.date.astype(str)
    daily_df = daily_df.groupby("date")[["calories", "protein", "carbs", "fat"]].sum().reset_index()
    daily_trend = daily_df.to_dict("records")

    # Top foods
    top_foods = df["food_name"].value_counts().head(5)
    top_foods_list = [{"food": k, "count": int(v)} for k, v in top_foods.items()]

    # Recent meals (last 10, newest first)
    recent_rows = df.tail(10).iloc[::-1].copy()
    recent_meals = []
    for _, row in recent_rows.iterrows():
        food_name = row.get("food_name") or row.get("food") or "Unknown"
        recent_meals.append({
            "food_name":  food_name,
            "emoji":      get_food_emoji(food_name),
            "calories":   row["calories"],
            "protein":    row["protein"],
            "carbs":      row["carbs"],
            "fat":        row["fat"],
            "timestamp":  str(row["timestamp"]),
            "status":     get_status_for_food(food_name, disease, {
                "calories": row["calories"],
                "carbs":    row["carbs"],
                "fat":      row["fat"],
            }),
        })

    return {
        "empty":         False,
        "stats": {
            "avg_calories": avg_calories,
            "avg_protein":  avg_protein,
            "avg_fat":      avg_fat,
            "avg_carbs":    avg_carbs,
            "total_meals":  total_meals,
            "health_score": score,
        },
        "calorie_trend":  calorie_trend,
        "macros":         macros,
        "daily_trend":    daily_trend,
        "top_foods":      top_foods_list,
        "recent_meals":   recent_meals,
    }


@app.post("/api/meals/weekly-insight")
async def weekly_insight(req: WeeklyInsightRequest):
    import pandas as pd
    data = get_food_history(req.username)
    if not data:
        raise HTTPException(status_code=404, detail="No meal history found.")
    df = pd.DataFrame(data)
    insight = weekly_diet_recommendation(df)
    return {"insight": insight}