import os
import json
import re
from groq import Groq

client = Groq(api_key="Groq_Token")

def _extract_json(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

def _clean_list(items):
    out = []
    for item in items or []:
        s = str(item).strip()
        s = re.sub(r"^\d+[\).\s-]*", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            out.append(s)
    return out

def enhance_recipe(recipe):
    ingredients = recipe.get("ingredients", []) or []
    instructions = recipe.get("instructions", "") or ""

    if isinstance(instructions, list):
        instructions = "\n".join([str(x).strip() for x in instructions if str(x).strip()])

    ingredient_lines = "\n".join([f"- {str(x).strip()}" for x in ingredients if str(x).strip()])

    prompt = f"""
Return JSON only in exactly this format:
{{
  "estimated": true,
  "ingredient_amounts": ["item 1", "item 2", "item 3"],
  "steps": ["step 1", "step 2", "step 3", "step 4", "step 5"],
  "notes": "short note"
}}

You are given incomplete recipe data.
Your job is to turn it into a FULL, COOKABLE, PRACTICAL recipe.

Rules:
- You ARE allowed to estimate realistic ingredient amounts.
- You ARE allowed to infer missing cooking details such as order, prep, temperature, cookware, and timing.
- Keep the recipe consistent with the title and ingredient list.
- Do not add weird unrelated ingredients.
- If needed, basic pantry assumptions are allowed: water, oil, salt, pepper.
- Prefer 4 servings unless the source clearly suggests otherwise.
- Write as many clear cooking steps (min: 12; max: 20).
- Make it sound like a real recipe a person can actually follow.
- ingredient_amounts must include estimated amounts for each main ingredient.
- notes should say briefly that some amounts/details were estimated from incomplete source data.
- The most important part is that for all ingredients (such as water, oil, salt, pepper, garlic cloves (minced), and chicken), you must specify the exact amounts, like 1/2 cup, 1 teaspoon, kilograms, grams, etc.
- In the "Ingredients" section only (not in the "Instructions"), include only things like 1/2 cup, 1 teaspoon, etc (and simular things like that). - In the "Ingredients" section only (not in the "Instructions"), don't include kg, g, etc.
Title: {recipe.get("title", "")}
Category: {recipe.get("category", "")}
Servings: {recipe.get("servings", "")}
Time: {recipe.get("time", "")}

Ingredients:
{ingredient_lines}

Original instructions:
{instructions}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You turn incomplete recipe data into full estimated recipes. Return strict JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5
        )

        content = response.choices[0].message.content.strip()
        data = _extract_json(content)

        if not data:
            return {
                "estimated": True,
                "ingredient_amounts": [],
                "steps": [],
                "notes": "Could not generate a full estimated recipe."
            }

        return {
            "estimated": True,
            "ingredient_amounts": _clean_list(data.get("ingredient_amounts", [])),
            "steps": _clean_list(data.get("steps", [])),
            "notes": str(data.get("notes", "")).strip()
        }

    except Exception as e:
        print("Groq error:", e)
        return {
            "estimated": True,
            "ingredient_amounts": [],
            "steps": [],
            "notes": "Could not generate a full estimated recipe."
        }