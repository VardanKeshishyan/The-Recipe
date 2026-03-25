# The-Recipe

Our project is a recipe search and recommendation system that helps people figure out what they can cook with the ingredients they already have at home. The user enters their ingredients, and the system uses search algorithms to go through a large recipe dataset and find matching recipes. Then it uses a recommendation system to rank the best options higher based on things like how many ingredients match, how fast the recipe is, and what the user has made before. It is designed for people like college students or anyone with limited time, money, or kitchen space, and the goal is to make cooking easier and more practical instead of starting with a dish and then buying ingredients.

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


# How the frontend looks

<img width="757" height="844" alt="image" src="https://github.com/user-attachments/assets/079c3d6b-0542-4314-8b61-cb7839251dc8" />

<img width="1540" height="844" alt="image" src="https://github.com/user-attachments/assets/e55b195b-3b24-4970-9677-0b6e5460b456" />

There are three simple ways of searching our system. The first one is ingredient search. The user enters ingredients that he or she already has, such as chicken, garlic, and rice, and the system finds recipes that best fit that query. The second one is dish name search. The user enters the name of one of the meals, such as Spaghetti Carbonara, and then the system identifies recipes that are closely related to that meal. The third is combined search. The user enters a dish and a few ingredients, such as pasta with tomato and basil. The system lists the recipes that match the type of dish and the ingredients provided.


# Dish search
<img width="1483" height="981" alt="image" src="https://github.com/user-attachments/assets/93fa1b16-cfb8-4cfd-9d22-f7e77ffdd537" />

# Ingredient search
<img width="1383" height="985" alt="image" src="https://github.com/user-attachments/assets/05b22258-cd1e-4864-9bda-a02382a5fdcd" />

# Both
<img width="1354" height="1002" alt="image" src="https://github.com/user-attachments/assets/fe2f982b-ceab-4e94-9f5b-b060fd95dab3" />

# Result
<img width="1239" height="844" alt="image" src="https://github.com/user-attachments/assets/17f888b2-4394-44eb-ba61-933116dd2a62" />





