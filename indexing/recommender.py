import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

try:
    from . import search
except ImportError:
    from indexing import search


class UserProfile:
    WEIGHTS = {"made": 3.0, "favorited": 2.0, "viewed": 1.0}

    def __init__(self):
        self.viewed: dict[int, int] = {}
        self.favorited: set[int] = set()
        self.made: set[int] = set()

    def record_view(self, recipe_id: int):
        self.viewed[recipe_id] = self.viewed.get(recipe_id, 0) + 1

    def toggle_favorite(self, recipe_id: int) -> bool:
        if recipe_id in self.favorited:
            self.favorited.discard(recipe_id)
            return False
        self.favorited.add(recipe_id)
        return True

    def toggle_made(self, recipe_id: int) -> bool:
        if recipe_id in self.made:
            self.made.discard(recipe_id)
            return False
        self.made.add(recipe_id)
        return True

    def get_interaction_score(self, recipe_id: int) -> float:
        score = 0.0
        score += self.viewed.get(recipe_id, 0) * self.WEIGHTS["viewed"]
        if recipe_id in self.favorited:
            score += self.WEIGHTS["favorited"]
        if recipe_id in self.made:
            score += self.WEIGHTS["made"]
        return score

    def get_positive_recipes(self) -> list[int]:
        ids = set(self.favorited) | set(self.made)
        ids |= {rid for rid, cnt in self.viewed.items() if cnt >= 2}
        return list(ids)

    def to_dict(self) -> dict:
        return {
            "viewed": self.viewed,
            "favorited": list(self.favorited),
            "made": list(self.made),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        p = cls()
        p.viewed = {int(k): v for k, v in d.get("viewed", {}).items()}
        p.favorited = set(int(x) for x in d.get("favorited", []))
        p.made = set(int(x) for x in d.get("made", []))
        return p

def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_str(value, default: str = ""):
    try:
        if value is None or pd.isna(value):
            return default
        return str(value)
    except Exception:
        return default


def _safe_list(value):
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
        return [s]
    return []

def _load_archetypes_if_available():
    if os.path.exists(search.ARCH_PATH):
        try:
            archetypes, _counts = search.load_archetypes(search.ARCH_PATH)
            return archetypes
        except Exception:
            return None
    return None


def _normalize_ingredient_list(query_ingredients: list[str]):
    out = set()
    for ing in query_ingredients:
        s = re.sub(r"[^a-z\s]", " ", str(ing).lower())
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            out.add(s)
    return out

def _build_user_taste_vector(profile: UserProfile, df: pd.DataFrame, tfidf_matrix, rid_to_row: dict[int, int],):
    positive = profile.get_positive_recipes()
    if not positive:
        return None

    vectors, weights = [], []
    for rid in positive:
        if rid not in rid_to_row:
            continue
        vec = tfidf_matrix[rid_to_row[rid]]
        w = profile.get_interaction_score(rid)
        vectors.append(vec.toarray())
        weights.append(max(w, 0.1))

    if not vectors:
        return None

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    taste_vec = np.average(np.vstack(vectors), axis=0, weights=weights)
    return taste_vec.reshape(1, -1)


def _hybrid_candidates(
    query: str,
    query_ingredients: list[str],
    df: pd.DataFrame,
    inverted_index: dict,
    vectorizer,
    tfidf_matrix,
    archetypes: Optional[dict[str, list[str]]] = None,
    topN_tfidf: int = 800,
):

    q_norm = search.normalize_query(query)
    ing_tokens = _normalize_ingredient_list(query_ingredients)

    combined_query_tokens = q_norm.split() + list(ing_tokens)
    combined_query_tokens = [t for t in combined_query_tokens if t]

    if not combined_query_tokens:
        # no text query / no ingredients -> fallback to all docs, zero base score
        candidate_rows = list(range(len(df)))
        return candidate_rows, np.zeros(len(candidate_rows), dtype=float)

    expanded_query, top_dish_set = search.expand_query(combined_query_tokens, archetypes)

    recipe_ids = df["RecipeId"].astype(int).to_numpy()
    rid_to_row = {rid: i for i, rid in enumerate(recipe_ids)}

    cand_from_ing = search.ing_candidate_rows(combined_query_tokens, inverted_index, rid_to_row)

    q_vec = vectorizer.transform([expanded_query])
    topN_rows = search.tfidf_topN_rows(q_vec, tfidf_matrix, topN_tfidf)

    candidate_rows = sorted(set(cand_from_ing) | set(topN_rows))
    if not candidate_rows:
        candidate_rows = list(range(len(df)))

    sims = cosine_similarity(q_vec, tfidf_matrix[candidate_rows]).ravel()
    ing_overlap = search.ing_overlap_scores(df, candidate_rows, combined_query_tokens)
    dish_boost = search.dish_boost_scores(df, candidate_rows, top_dish_set)

    w_text, w_ing, w_dish = 0.65, 0.20, 0.15
    base_scores = (w_text * sims) + (w_ing * ing_overlap) + (w_dish * dish_boost)

    return candidate_rows, base_scores


def recommend(query: str, query_ingredients: list[str], df: pd.DataFrame, inverted_index: dict,
    vectorizer, tfidf_matrix, profile: Optional[UserProfile] = None, k: int = 20,
    alpha: float = 0.70, beta: float = 0.25, gamma: float = 0.05,
    topN_tfidf: int = 800, archetypes: Optional[dict[str, list[str]]] = None,) -> list[dict]:

    if archetypes is None:
        archetypes = _load_archetypes_if_available()

    recipe_ids = df["RecipeId"].astype(int).to_numpy()
    rid_to_row = {rid: i for i, rid in enumerate(recipe_ids)}

    candidate_rows, base_search_scores = _hybrid_candidates(
        query=query,
        query_ingredients=query_ingredients,
        df=df,
        inverted_index=inverted_index,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        archetypes=archetypes,
        topN_tfidf=topN_tfidf,
    )

    sub_matrix = tfidf_matrix[candidate_rows]

    taste_vec = _build_user_taste_vector(profile, df, tfidf_matrix, rid_to_row) if profile else None
    if taste_vec is not None:
        user_taste = cosine_similarity(taste_vec, sub_matrix).ravel()
    else:
        user_taste = np.zeros(len(candidate_rows), dtype=float)

    ratings = df.iloc[candidate_rows]["AggregatedRating"].fillna(0).to_numpy(dtype=float)
    max_r = ratings.max() if len(ratings) and ratings.max() > 0 else 1.0
    popularity_boost = ratings / max_r

    combined = (alpha * base_search_scores) + (beta * user_taste) + (gamma * popularity_boost)

    top_local = np.argsort(-combined)[:k]

    results = []
    for j in top_local:
        row_i = candidate_rows[int(j)]
        row = df.iloc[row_i]
        rid = int(row["RecipeId"])

        results.append({
            "RecipeId": _safe_int(row.get("RecipeId", rid), rid),
            "Name": _safe_str(row.get("Name", "")),
            "score": _safe_float(combined[int(j)]),
            "search_score": _safe_float(base_search_scores[int(j)]),
            "taste_score": _safe_float(user_taste[int(j)]),
            "rating_score": _safe_float(popularity_boost[int(j)]),
            "Description": _safe_str(row.get("Description", "")),
            "RecipeCategory": _safe_str(row.get("RecipeCategory", "")),
            "AggregatedRating": _safe_float(row.get("AggregatedRating", 0)),
            "ReviewCount": _safe_int(row.get("ReviewCount", 0)),
            "TotalTime": _safe_str(row.get("TotalTime", "")),
            "Calories": _safe_float(row.get("Calories", 0)),
            "parsed_ingredients": _safe_list(row.get("parsed_ingredients", [])),
            "parsed_instructions": _safe_list(row.get("parsed_instructions", [])),
            "parsed_keywords": _safe_list(row.get("parsed_keywords", [])),
            "RecipeServings": _safe_str(row.get("RecipeServings", "")),
            "is_favorited": rid in profile.favorited if profile else False,
            "is_made": rid in profile.made if profile else False,
            "interaction_score": profile.get_interaction_score(rid) if profile else 0.0,
        })

    return results


def get_similar_recipes(recipe_id: int, df: pd.DataFrame, tfidf_matrix, rid_to_row: dict[int, int], k: int = 6,) -> list[dict]:
    if recipe_id not in rid_to_row:
        return []

    row_i = rid_to_row[recipe_id]
    sims = cosine_similarity(tfidf_matrix[row_i], tfidf_matrix).ravel()
    sims[row_i] = -1.0

    top = np.argsort(-sims)[:k]
    results = []
    for i in top:
        row = df.iloc[int(i)]
        results.append({
            "RecipeId": _safe_int(row["RecipeId"]),
            "Name": _safe_str(row["Name"]),
            "score": _safe_float(sims[int(i)]),
            "RecipeCategory": _safe_str(row.get("RecipeCategory", "")),
            "AggregatedRating": _safe_float(row.get("AggregatedRating", 0) or 0),
        })
    return results