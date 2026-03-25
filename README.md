# The-Recipe
HI this is web
Clone the repository:

git clone https://github.com/VardanKeshishyan/The-Recipe.git

cd RecipeRecommendation
Download the Datasets:
Download the datasets from https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews?select=recipes.csv

After downloading, place the CSV files into a folder called data/ in the project root:


RecipeRecommendation/
├── data/
│ ├── recipes.csv
│ └── reviews.csv

Run preprocessing and indexing:
python indexing/preprocess.py
python indexing/indexer.py
