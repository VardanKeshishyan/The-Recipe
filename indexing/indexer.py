from collections import defaultdict
import pickle
import gzip
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import load_recipes, preprocess_recipes

INVERTED_PATH = "data/inverted_index.pkl"
TFIDF_PATH = "data/tfidf_index.pkl"
META_PATH = "data/doc_meta.pkl"

def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        magic = f.read(2)
        f.seek(0)

        if magic == b"\x1f\x8b":
            with gzip.open(path, "rb") as gz:
                return pickle.load(gz)
        else:
            return pickle.load(f)

def save_meta(meta: dict, path: str = META_PATH):
    save_pickle(meta, path)

def load_meta(path: str = META_PATH):
    return load_pickle(path)


def build_inverted_index(df: pd.DataFrame):
    inverted_index = defaultdict(set)

    for rid, ings in zip(df["RecipeId"].to_numpy(), df["parsed_ingredients"]):
        rid = int(rid)
        for ing in ings:
            if ing:
                inverted_index[ing].add(rid)

    return inverted_index


def build_text_corpus(df: pd.DataFrame):
    corpus = []
    for _, row in df.iterrows():
        text = " ".join([
            str(row.get("Name", "")),
            str(row.get("Description", "")),
            str(row.get("RecipeCategory", "")),
            " ".join(row.get("parsed_keywords", [])),
        ])
        corpus.append(text)
    return corpus


def build_tfidf_index(df: pd.DataFrame):
    corpus = df["document"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


def save_tfidf_index(vectorizer, tfidf_matrix, path: str = TFIDF_PATH):
    save_pickle((vectorizer, tfidf_matrix), path)

def load_tfidf_index(path: str = TFIDF_PATH):
    return load_pickle(path)

def save_index(index, path: str = INVERTED_PATH):
    save_pickle(dict(index), path)

def load_index(path: str = INVERTED_PATH):
    return load_pickle(path)


def build_and_save_all(inverted_path: str = INVERTED_PATH, tfidf_path: str = TFIDF_PATH, meta_path: str = META_PATH,):
    df = preprocess_recipes(load_recipes())

    inv = build_inverted_index(df)
    save_index(inv, inverted_path)

    vec, tfidf = build_tfidf_index(df)
    save_tfidf_index(vec, tfidf, tfidf_path)

    recipe_ids = df["RecipeId"].astype(int).to_numpy()
    docstore = df[["RecipeId", "Name", "RecipeCategory"]].copy()
    save_meta({"recipe_ids": recipe_ids, "docstore": docstore}, meta_path)

def build_and_save_index():
    df = load_recipes()
    df = preprocess_recipes(df)
    index = build_inverted_index(df)
    save_index(index, INVERTED_PATH)


if __name__ == "__main__":
    build_and_save_all()
