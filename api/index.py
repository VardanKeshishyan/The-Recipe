from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from indexing import preprocess
from indexing import indexer
from indexing.recommender import recommend, get_similar_recipes, UserProfile
import os
import json
import uuid
import ast
import re
import hashlib
from datetime import datetime
import pandas as pd
from groq_helper import enhance_recipe
import asyncio
from vercel.blob import AsyncBlobClient
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TMP_DIR = "/tmp/recipe-data"
os.makedirs(TMP_DIR, exist_ok=True)

async def _download_blob_once(pathname: str):
    client = AsyncBlobClient()
    result = await client.get(pathname, access="private")
    if result is None or result.status_code != 200:
        raise FileNotFoundError(f"Blob not found: {pathname}")

    local_path = os.path.join(TMP_DIR, os.path.basename(pathname))
    urllib.request.urlretrieve(result.blob.download_url, local_path)
    return local_path
    
def get_blob_file(pathname: str):
    local_path = os.path.join(TMP_DIR, os.path.basename(pathname))
    if os.path.exists(local_path):
        return local_path
    return asyncio.run(_download_blob_once(pathname))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = os.environ.get("SECRET_KEY", "recipe-search-dev-key-change-in-prod")

recipes_csv = get_blob_file("recipes.csv")
inv_path = get_blob_file("inverted_index.pkl")
tfidf_path = get_blob_file("tfidf_index.pkl")

df = preprocess.load_recipes(recipes_csv)
df = preprocess.preprocess_recipes(df)

if not os.path.exists(inv_path):
    print("Building inverted index...")
    inverted_index = indexer.build_inverted_index(df)
    indexer.save_index(inverted_index, inv_path)
else:
    inverted_index = indexer.load_index(inv_path)

if not os.path.exists(tfidf_path):
    print("Building TF-IDF index...")
    vectorizer, tfidf_matrix = indexer.build_tfidf_index(df)
    indexer.save_tfidf_index(vectorizer, tfidf_matrix, tfidf_path)
else:
    vectorizer, tfidf_matrix = indexer.load_tfidf_index(tfidf_path)

rid_to_row = {rid: i for i, rid in enumerate(df["RecipeId"].to_numpy())}
print(f"Ready! {len(df)} recipes indexed.")

PROFILE_DIR = os.path.join(BASE_DIR, "data", "profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

USERS_FILE = os.path.join(BASE_DIR, "data", "users.json")

def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def _set_user_session(username: str):
    """Log a user in: bind their session sid to their username so the
    profile system automatically uses their persistent profile file."""
    session["user"] = username
    session["sid"] = username
    session.permanent = True

# ─────────────────────────────────────────────
#  PROFILE HELPERS  (unchanged logic)
# ─────────────────────────────────────────────

def _profile_path(session_id: str) -> str:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    return os.path.join(PROFILE_DIR, f"{safe}.json")

def get_profile() -> UserProfile:
    sid = session.get("sid")
    if not sid:
        return UserProfile()
    path = _profile_path(sid)
    if not os.path.exists(path):
        return UserProfile()
    try:
        with open(path) as f:
            return UserProfile.from_dict(json.load(f))
    except Exception:
        return UserProfile()

def save_profile(profile: UserProfile):
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
        session.permanent = True
    with open(_profile_path(sid), "w") as f:
        json.dump(profile.to_dict(), f)

# ─────────────────────────────────────────────
#  SAFE CONVERSION HELPERS  (unchanged)
# ─────────────────────────────────────────────

def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default

def safe_str(value, default: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return default
        return str(value)
    except Exception:
        return default

def safe_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() == "nan":
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return []


def clean_estimated_ingredient(item: str) -> str:
    s = safe_str(item, "").strip()
    if not s:
        return ""

    s = re.sub(r"\s+", " ", s).strip()

    patterns = [
        r"^([a-zA-Z]+\s*\([^)]+\))\s+(.+)$",   # g (1 tsp) salt
        r"^(teaspoons?)\s+(.+)$",
        r"^(tablespoons?)\s+(.+)$",
        r"^(cups?)\s+(.+)$",
        r"^(cloves? of)\s+(.+)$",
    ]

    for pattern in patterns:
        m = re.match(pattern, s, flags=re.I)
        if m:
            front = m.group(1).strip()
            rest = m.group(2).strip()
            s = f"{rest} ({front})"
            break

    s = re.sub(r"\b(.+?)\s+\1\b", r"\1", s, flags=re.I)
    return s.strip()

def normalize_ingredient_list(items) -> list:
    out = []
    seen = set()

    for item in items or []:
        s = clean_estimated_ingredient(item)
        if not s:
            continue

        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)

    return out

def instructions_to_text(value) -> str:
    if isinstance(value, list):
        return "\n".join([str(x).strip() for x in value if str(x).strip()])
    return safe_str(value, "")

def split_instruction_text(text: str) -> list:
    text = safe_str(text, "")
    if not text:
        return []

    lines = [x.strip() for x in text.splitlines() if x.strip()]

    if len(lines) <= 1:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        lines = [x.strip() for x in parts if x.strip()]

    cleaned = []
    for line in lines:
        line = re.sub(r"^\d+[\).\s-]*", "", line).strip()
        if line:
            cleaned.append(line)

    return cleaned

def ai_steps_are_bad(steps: list, ingredients: list) -> bool:
    if not steps:
        return True

    total_words = sum(len(str(step).split()) for step in steps)
    if total_words < 18:
        return True

    if len(steps) == 1 and len(ingredients) >= 3:
        return True

    giant_step = any(len(str(step).split()) > 55 for step in steps)
    if giant_step and len(steps) <= 2:
        return True

    return False

def build_fallback_steps(original_steps: list, ingredients: list) -> list:
    if len(original_steps) >= 2:
        return original_steps

    if len(original_steps) == 1:
        return original_steps

    if ingredients:
        return [
            "Gather the listed ingredients.",
            "Combine the ingredients using the order suggested by the recipe title and source notes.",
            "The source recipe does not provide enough detail to generate reliable full cooking steps."
        ]

    return ["The source recipe does not provide enough detail to generate reliable instructions."]

def normalize_step_list(value) -> list:
    if isinstance(value, list):
        out = []
        for item in value:
            s = safe_str(item, "").strip()
            s = re.sub(r"^\d+[\).\s-]*", "", s)
            if s:
                out.append(s)
        return out

    if isinstance(value, str):
        return split_instruction_text(value)

    return []

def get_best_ingredient_display_list(row) -> list:
    quantity_candidates = [
        "parsed_ingredient_quantities",
        "parsed_quantities",
        "RecipeIngredientQuantities",
        "ingredient_quantities",
    ]
    part_candidates = [
        "parsed_ingredients",
        "RecipeIngredientParts",
        "ingredient_parts",
    ]

    qtys = []
    parts = []

    for key in quantity_candidates:
        vals = safe_list(row.get(key, []))
        if vals:
            qtys = vals
            break

    for key in part_candidates:
        vals = safe_list(row.get(key, []))
        if vals:
            parts = vals
            break

    if not parts:
        return safe_list(row.get("parsed_ingredients", []))

    combined = []

    for i, part in enumerate(parts):
        part_str = safe_str(part, "").strip()
        qty_str = safe_str(qtys[i], "").strip() if i < len(qtys) else ""

        if not part_str:
            continue

        if not qty_str or qty_str.lower() == "nan":
            combined.append(part_str)
            continue

        qty_lower = qty_str.lower()

        starts_with_number = bool(re.match(r"^\d", qty_str))
        looks_like_amount = any(ch.isdigit() for ch in qty_str)

        if starts_with_number or looks_like_amount:
            combined.append(f"{qty_str} {part_str}".strip())
        elif qty_lower in ["teaspoon", "teaspoons", "tablespoon", "tablespoons", "cup", "cups", "clove", "cloves"]:
            combined.append(f"{part_str} ({qty_str})".strip())
        elif qty_lower.endswith("of"):
            combined.append(f"{part_str} ({qty_str})".strip())
        else:
            combined.append(f"{part_str} ({qty_str})".strip())

    return combined

    return safe_list(row.get("parsed_ingredients", []))

# ─────────────────────────────────────────────
#  AUTH ROUTES
# ─────────────────────────────────────────────

@app.route("/login", methods=["GET"])
def login_page():
    if "user" in session:
        return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/register", methods=["GET"])
def register_page():
    if "user" in session:
        return redirect(url_for("home"))
    return render_template("register.html")

@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    confirm  = data.get("confirm") or ""

    if not username:
        return jsonify({"error": "Username is required."}), 400
    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters."}), 400
    if not re.match(r"^[a-z0-9_]+$", username):
        return jsonify({"error": "Username can only contain letters, numbers, and underscores."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    if password != confirm:
        return jsonify({"error": "Passwords do not match."}), 400

    users = _load_users()
    if username in users:
        return jsonify({"error": "That username is already taken."}), 400

    users[username] = {
        "password_hash": _hash_password(password),
        "created_at": datetime.utcnow().isoformat(),
    }
    _save_users(users)
    _set_user_session(username)
    return jsonify({"success": True})

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    users = _load_users()
    user = users.get(username)
    if not user or user.get("password_hash") != _hash_password(password):
        return jsonify({"error": "Incorrect username or password."}), 401

    _set_user_session(username)
    return jsonify({"success": True})

@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/api/auth/status", methods=["GET"])
def api_auth_status():
    return jsonify({
        "logged_in": "user" in session,
        "username": session.get("user"),
    })

# ─────────────────────────────────────────────
#  EXISTING ROUTES  (unchanged)
# ─────────────────────────────────────────────

@app.route("/")
def home():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
        session.permanent = True
    return render_template("index.html")

@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    ingredients = data.get("ingredients", [])
    k = int(data.get("k", 20))

    profile = get_profile()

    results = recommend(
        query=query,
        query_ingredients=ingredients,
        df=df,
        inverted_index=inverted_index,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        profile=profile,
        k=k,
    )

    return jsonify({"results": results})

@app.route("/api/recipe/<int:recipe_id>", methods=["GET"])
def api_recipe_detail(recipe_id: int):
    profile = get_profile()
    profile.record_view(recipe_id)
    save_profile(profile)

    if recipe_id not in rid_to_row:
        return jsonify({"error": "Recipe not found"}), 404

    row = df.iloc[rid_to_row[recipe_id]]

    display_ingredients = get_best_ingredient_display_list(row)
    original_text = instructions_to_text(row.get("parsed_instructions", []))
    original_steps = split_instruction_text(original_text)

    recipe_for_ai = {
        "title": safe_str(row.get("Name", "")),
        "category": safe_str(row.get("RecipeCategory", "")),
        "description": safe_str(row.get("Description", "")),
        "ingredients": display_ingredients,
        "instructions": original_text,
        "time": safe_str(row.get("TotalTime", "")),
        "calories": safe_float(row.get("Calories", 0)),
        "servings": safe_str(row.get("RecipeServings", "")),
    }

    ai_result = enhance_recipe(recipe_for_ai)

    estimated_ingredients = display_ingredients[:]
    ai_steps = []
    instruction_note = ""
    ai_used = False

    if isinstance(ai_result, dict):
        if ai_result.get("ingredient_amounts"):
            estimated_ingredients = normalize_ingredient_list(ai_result.get("ingredient_amounts", []))
        elif ai_result.get("ingredients"):
            estimated_ingredients = normalize_ingredient_list(ai_result.get("ingredients", []))

        ai_steps = normalize_step_list(ai_result.get("steps", []))
        instruction_note = safe_str(
            ai_result.get("notes", ai_result.get("warning", "")),
            ""
        ).strip()

    elif isinstance(ai_result, str):
        ai_steps = split_instruction_text(ai_result)

    if ai_steps_are_bad(ai_steps, estimated_ingredients):
        final_steps = build_fallback_steps(original_steps, estimated_ingredients)
        if not instruction_note:
            instruction_note = "This recipe was incomplete, so the app used limited source details."
        ai_used = False
    else:
        final_steps = ai_steps
        ai_used = True
        if not instruction_note:
            instruction_note = "Some ingredient amounts or steps may be estimated from incomplete source data."

    try:
        similar = get_similar_recipes(
            recipe_id,
            df,
            tfidf_matrix,
            rid_to_row,
            k=6,
        )
    except Exception as e:
        print(f"similar-recipes error for {recipe_id}: {e}")
        similar = []

    return jsonify({
        "RecipeId": safe_int(row.get("RecipeId", recipe_id), recipe_id),
        "Name": safe_str(row.get("Name", "")),
        "Description": safe_str(row.get("Description", "")),
        "RecipeCategory": safe_str(row.get("RecipeCategory", "")),
        "AggregatedRating": safe_float(row.get("AggregatedRating", 0)),
        "ReviewCount": safe_int(row.get("ReviewCount", 0)),
        "TotalTime": safe_str(row.get("TotalTime", "")),
        "Calories": safe_float(row.get("Calories", 0)),
        "RecipeServings": safe_str(row.get("RecipeServings", "")),
        "parsed_ingredients": estimated_ingredients,
        "parsed_instructions": final_steps,
        "original_instructions": original_steps,
        "parsed_keywords": safe_list(row.get("parsed_keywords", [])),
        "is_favorited": recipe_id in profile.favorited,
        "is_made": recipe_id in profile.made,
        "similar": similar,
        "instruction_note": instruction_note,
        "ai_used": ai_used,
    })

@app.route("/api/profile/favorite/<int:recipe_id>", methods=["POST"])
def api_toggle_favorite(recipe_id: int):
    profile = get_profile()
    now_favorited = profile.toggle_favorite(recipe_id)
    save_profile(profile)
    return jsonify({"favorited": now_favorited, "recipe_id": recipe_id})

@app.route("/api/profile/made/<int:recipe_id>", methods=["POST"])
def api_toggle_made(recipe_id: int):
    profile = get_profile()
    now_made = profile.toggle_made(recipe_id)
    save_profile(profile)
    return jsonify({"made": now_made, "recipe_id": recipe_id})

def recipe_row_to_card(row, profile: UserProfile) -> dict:
    rid = safe_int(row.get("RecipeId", 0))
    return {
        "RecipeId": rid,
        "Name": safe_str(row.get("Name", "")),
        "RecipeCategory": safe_str(row.get("RecipeCategory", "")),
        "AggregatedRating": safe_float(row.get("AggregatedRating", 0)),
        "TotalTime": safe_str(row.get("TotalTime", "")),
        "Calories": safe_float(row.get("Calories", 0)),
        "score": safe_float(row.get("AggregatedRating", 0)),
        "is_favorited": rid in profile.favorited,
        "is_made": rid in profile.made,
    }

@app.route("/api/profile", methods=["GET"])
def api_profile():
    profile = get_profile()

    def enrich(recipe_ids):
        out = []
        for rid in recipe_ids:
            if rid not in rid_to_row:
                continue
            out.append(recipe_row_to_card(df.iloc[rid_to_row[rid]], profile))
        return out

    top_viewed_ids = [rid for rid, _ in sorted(profile.viewed.items(), key=lambda x: -x[1])[:10]]

    return jsonify({
        "viewed_count": len(profile.viewed),
        "favorited_count": len(profile.favorited),
        "made_count": len(profile.made),
        "viewed": profile.viewed,
        "favorited": list(profile.favorited),
        "made": list(profile.made),
        "favorited_recipes": enrich(list(profile.favorited)),
        "made_recipes": enrich(list(profile.made)),
        "viewed_recipes": enrich(top_viewed_ids),
    })

@app.route("/api/recommendations/for-you", methods=["POST"])
def api_for_you():
    data = request.get_json(force=True) or {}
    context_ingredients = data.get("ingredients", [])
    profile = get_profile()

    if not profile.get_positive_recipes():
        if context_ingredients:
            results = recommend(
                query="",
                query_ingredients=context_ingredients,
                df=df,
                inverted_index=inverted_index,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                profile=profile,
                k=20,
            )
            return jsonify({"results": results, "cold_start": True, "mode": "ingredients"})

        popular = df.nlargest(20, "AggregatedRating")
        results = [recipe_row_to_card(row, profile) for _, row in popular.iterrows()]
        return jsonify({"results": results, "cold_start": True, "mode": "popular"})

    results = recommend(
        query="",
        query_ingredients=context_ingredients,
        df=df,
        inverted_index=inverted_index,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        profile=profile,
        k=20,
        alpha=0.10,
        beta=0.85,
        gamma=0.05,
    )
    return jsonify({"results": results, "cold_start": False, "mode": "personalized"})
