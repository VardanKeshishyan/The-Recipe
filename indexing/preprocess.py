import pandas as pd
# import ast
import re

# Loads recipe data
def load_recipes(path="data/recipes.csv"):
    df = pd.read_csv(path)
    return df

# Parses the list of ingredients 
# ie. c("blueberries", "granulated sugar") -> ["blueberries", "granulated sugar"]


def parse_ingredients(ingredients_str):
    if pd.isna(ingredients_str):
        return []
    s = str(ingredients_str).strip()

    if s.startswith("c(") and s.endswith(")"):
        return parse_r_list(s)

    if len(s) >= 3:
        s = s[2:-1]
    parts = s.split('", "')
    parts = [p.replace('"', "") for p in parts]
    return parts


def parse_r_list(text):
    if pd.isna(text):
        return []
    text = re.sub(r'^c\(', '', text)
    text = re.sub(r'\)$', '', text)
    
    return re.findall(r'"(.*?)"', text)

# Normalizes ingredient names by converting to lowercase and removing non-alphabetic characters
def normalize_ingredient(ingredient):
    ingredient = ingredient.lower()
    ingredient = re.sub(r"[^a-z\s]", "", ingredient)
    ingredient = re.sub(r"\s+", " ", ingredient).strip()
    return ingredient.strip()

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower()


def build_document(row):
    parts = []

    # Title (weight x3)
    title = clean_text(row["Name"])
    parts.append((title + " ") * 3)

    # Category (weight x2)
    category = clean_text(row["RecipeCategory"])
    parts.append((category + " ") * 2)

    # Ingredients
    ingredients = " ".join(row["parsed_ingredients"])
    parts.append(ingredients)

    # Keywords
    keywords = " ".join(row["parsed_keywords"])
    parts.append(keywords)

    # Description
    description = clean_text(row["Description"])
    parts.append(description)

    return " ".join(parts)

# apply preprocessing to all recipes in the dataframe
def preprocess_recipes(df):
    df["parsed_ingredients"] = df["RecipeIngredientParts"].apply(
        lambda x: [
            normalize_ingredient(i)
            for i in parse_ingredients(x)
        ]
    )
    
    df["parsed_keywords"] = df["Keywords"].apply(parse_r_list)

    # Optional: parse instructions (if you want later)
    df["parsed_instructions"] = df["RecipeInstructions"].apply(parse_r_list)

    # Build final searchable document
    df["document"] = df.apply(build_document, axis=1)

    return df

if __name__ == "__main__":
    df = load_recipes()
    df = preprocess_recipes(df)

    for i in range(5):
        print(df.loc[i, "Name"])
        print("Ingredients:", df.loc[i, "parsed_ingredients"])
        print("Keywords:", df.loc[i, "parsed_keywords"])
        print()



