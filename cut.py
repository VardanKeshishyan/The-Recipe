import pandas as pd

recipes = pd.read_csv("data/recipes.csv")
recipes.head(200000).to_csv("data/recipes_small.csv", index=False)

reviews = pd.read_csv("data/reviews.csv")
reviews.head(150000).to_csv("data/reviews_small.csv", index=False)

print("Done")