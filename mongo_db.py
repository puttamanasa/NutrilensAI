from pymongo import MongoClient
import bcrypt
from datetime import datetime

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
MONGO_URI = "mongodb+srv://puttamanasa2006_db_user:DDZLFaL2cm21eFSX@cluster0.mpiv5k8.mongodb.net/?appName=Cluster0"

client = MongoClient(MONGO_URI)
db     = client["nutrilens"]

users   = db["users"]
history = db["food_history"]


# ─────────────────────────────────────────────
# REGISTER
# ─────────────────────────────────────────────
def register_user(username: str, password: str):
    if users.find_one({"username": username}):
        return False, "User already exists"

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    users.insert_one({
        "username":   username,
        "password":   hashed_pw,
        "created_at": datetime.utcnow()
    })
    return True, "Account created successfully ✅"


# ─────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────
def login_user(username: str, password: str):
    user = users.find_one({"username": username})

    if not user:
        return False, "User not found"

    if bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return True, "Login successful"

    return False, "Incorrect password"


# ─────────────────────────────────────────────
# SAVE FOOD HISTORY  (called after each analysis)
# ─────────────────────────────────────────────
def save_food_history(username: str, food_name: str, caption: str,
                      calories: float, protein: float,
                      carbs: float, fat: float, portion: float):
    history.insert_one({
        "username":  username,
        "food_name": food_name,
        "caption":   caption,
        "calories":  calories,
        "protein":   protein,
        "carbs":     carbs,
        "fat":       fat,
        "portion":   portion,
        "timestamp": datetime.utcnow()
    })


# ─────────────────────────────────────────────
# GET FOOD HISTORY  (returns list of dicts)
# ─────────────────────────────────────────────
def get_food_history(username: str):
    records = list(
        history.find(
            {"username": username},
            {"_id": 0}          # exclude Mongo internal _id
        ).sort("timestamp", -1) # newest first
    )
    return records


# ─────────────────────────────────────────────
# DELETE ONE ENTRY  (optional helper)
# ─────────────────────────────────────────────
def delete_history_entry(username: str, timestamp: datetime):
    history.delete_one({"username": username, "timestamp": timestamp})