# auth.py  —  JWT creation + verification + user register/login

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from pymongo import MongoClient
import bcrypt

# ── CONFIG ────────────────────────────────────────────────────────────────────
SECRET_KEY  = "nutrilens-super-secret-key-change-in-production"
ALGORITHM   = "HS256"
TOKEN_EXPIRE_HOURS = 24

# ── MONGO ─────────────────────────────────────────────────────────────────────
MONGO_URI = "mongodb+srv://puttamanasa2006_db_user:DDZLFaL2cm21eFSX@cluster0.mpiv5k8.mongodb.net/?appName=Cluster0"
_client = MongoClient(MONGO_URI)
_db     = _client["nutrilens"]
_users  = _db["users"]


# ─────────────────────────────────────────────────────────────────────────────
# JWT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# USER REGISTER / LOGIN
# ─────────────────────────────────────────────────────────────────────────────
def register_user(username: str, password: str):
    if len(password) < 6:
        return False, "Password must be at least 6 characters"

    if _users.find_one({"username": username}):
        return False, "Username already taken"

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    _users.insert_one({
        "username":   username,
        "password":   hashed,
        "created_at": datetime.utcnow(),
    })
    return True, "Account created successfully"


def login_user(username: str, password: str):
    user = _users.find_one({"username": username})
    if not user:
        return False, "User not found"
    if bcrypt.checkpw(password.encode(), user["password"]):
        return True, "Login successful"
    return False, "Incorrect password"