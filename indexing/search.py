import os
import re
import pickle
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from . import indexer, preprocess
except ImportError:
    from indexing import indexer, preprocess


ARCH_PATH = "data/dish_archetypes.pkl"

def normalize_query(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^a-z\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

DISH_PATTERNS = {
    "sandwich": r"\bsandwich|panini|sub|hoagie|wrap|burger|sliders\b",
    "salad": r"\bsalad|slaw\b",
    "pasta": r"\bpasta|spaghetti|penne|lasagna|macaroni|noodle\b",
    "soup": r"\bsoup|stew|chowder|bisque\b",
    "pizza": r"\bpizza|calzone\b",
    "taco": r"\btaco|burrito|quesadilla|enchilada|fajita\b",
    "dessert": r"\bcake|cookie|brownie|pie|pudding|ice cream|dessert\b",
    "breakfast": r"\bbreakfast|omelet|pancake|waffle|muffin\b",
    "stirfry": r"\bstir fry|stirfry\b",
    "curry": r"\bcurry\b",
}

def label_recipe(row: pd.Series) -> set[str]:
    name = str(row.get("Name", ""))
    cat = str(row.get("RecipeCategory", ""))
    kws = row.get("parsed_keywords", [])
    text = (name + " " + cat + " " + " ".join(kws)).lower()

    labels = set()
    for dish, pat in DISH_PATTERNS.items():
        if re.search(pat, text):
            labels.add(dish)
    return labels

def build_dish_archetypes(df: pd.DataFrame, top_n=30, min_recipes=200):
    dish_counts = defaultdict(Counter)
    dish_recipe_counts = Counter()

    for _, row in df.iterrows():
        labels = label_recipe(row)
        if not labels:
            continue
        ings = row["parsed_ingredients"]
        for d in labels:
            dish_recipe_counts[d] += 1
            dish_counts[d].update([i for i in ings if i])

    archetypes = {}
    for dish, cnt in dish_recipe_counts.items():
        if cnt >= min_recipes:
            archetypes[dish] = [ing for ing, _ in dish_counts[dish].most_common(top_n)]

    return archetypes, dict(dish_recipe_counts)

def save_archetypes(archetypes, dish_recipe_counts, path: str = ARCH_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"archetypes": archetypes, "dish_recipe_counts": dish_recipe_counts}, f)

def load_archetypes(path: str = ARCH_PATH):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["archetypes"], obj["dish_recipe_counts"]

def top_dish_types_for_query(query_tokens: list[str], archetypes: dict[str, list[str]], topk=5):
    q = set(query_tokens)
    scored = []
    for dish, common_ings in archetypes.items():
        common = set(common_ings)
        overlap = len(q & common)
        score = overlap / (len(q) + 1e-9)
        if score > 0:
            scored.append((dish, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:topk]

def expand_query_with_archetypes(query_tokens: list[str], top_dishes: list[tuple[str, float]],
    archetypes: dict[str, list[str]], max_added=6):
    base = list(query_tokens)
    have = set(base)

    added = []
    for dish, _score in top_dishes:
        for ing in archetypes.get(dish, []):
            if ing not in have:
                added.append(ing)
                have.add(ing)
            if len(added) >= max_added:
                break
        if len(added) >= max_added:
            break

    return base + added, added

def expand_query(query_tokens, archetypes):
    expanded_query = " ".join(query_tokens)
    top_dish_set = set()

    if archetypes:
        dish_scores = top_dish_types_for_query(query_tokens, archetypes, topk=5)
        expanded_tokens, _added = expand_query_with_archetypes(
            query_tokens, dish_scores, archetypes, max_added=6
        )
        expanded_query = " ".join(expanded_tokens)
        top_dish_set = {d for d, _ in dish_scores[:3]}

    return expanded_query, top_dish_set


def ing_candidate_rows(query_tokens, inverted_index, rid_row):
    ing_candidates = set()
    for t in query_tokens:
        if t in inverted_index:
            ing_candidates |= set(inverted_index[t])

    return [rid_row[rid] for rid in ing_candidates if rid in rid_row]


def tfidf_topN_rows(q_vec, tfidf_matrix, topN):
    global_sims = cosine_similarity(q_vec, tfidf_matrix).ravel()
    return np.argsort(-global_sims)[:topN].tolist()


def ing_overlap_scores(df, candidate_rows, query_tokens):
    qset = set(query_tokens)
    scores = np.zeros(len(candidate_rows), dtype=float)

    for j, row_i in enumerate(candidate_rows):
        recipe_ings = set(df.iloc[row_i]["parsed_ingredients"])
        scores[j] = len(recipe_ings & qset)

    if scores.max() > 0:
        scores /= scores.max()
    return scores


def dish_boost_scores(df, candidate_rows, top_dish_set):
    boost = np.zeros(len(candidate_rows), dtype=float)
    if not top_dish_set:
        return boost

    for j, row_i in enumerate(candidate_rows):
        if label_recipe(df.iloc[row_i]) & top_dish_set:
            boost[j] = 1.0
    return boost

def search(query: str, df: pd.DataFrame, inverted_index: dict, vectorizer: TfidfVectorizer, tfidf_matrix,
    archetypes: dict[str, list[str]] | None = None, k: int = 10, topN_tfidf: int = 800,):
    
    q_norm = normalize_query(query)
    if not q_norm:
        return []

    query_tokens = q_norm.split()

    expanded_query, top_dish_set = expand_query(query_tokens, archetypes)

    recipe_ids = df["RecipeId"].astype(int).to_numpy()
    rid_row = {rid: i for i, rid in enumerate(recipe_ids)}

    cand_from_ing = ing_candidate_rows(query_tokens, inverted_index, rid_row)

    q_vec = vectorizer.transform([expanded_query])
    topN_rows = tfidf_topN_rows(q_vec, tfidf_matrix, topN_tfidf)

    candidate_rows = sorted(set(cand_from_ing) | set(topN_rows))

    sims = cosine_similarity(q_vec, tfidf_matrix[candidate_rows]).ravel()
    ing_overlap = ing_overlap_scores(df, candidate_rows, query_tokens)
    dish_boost = dish_boost_scores(df, candidate_rows, top_dish_set)

    w_text, w_ing, w_dish = 0.65, 0.20, 0.15
    final_scores = (w_text * sims) + (w_ing * ing_overlap) + (w_dish * dish_boost)

    top_local = np.argsort(-final_scores)[:k]
    return [
        {
            "RecipeId": int(df.iloc[candidate_rows[int(j)]]["RecipeId"]),
            "Name": str(df.iloc[candidate_rows[int(j)]]["Name"]),
            "score": float(final_scores[int(j)]),
        }
        for j in top_local
    ]