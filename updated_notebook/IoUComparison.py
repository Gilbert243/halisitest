import json
import re

#  Utility Functions
# Data cleaning 
def load_clean_json(filepath):
    """Load JSON and clean Mongo-style ObjectId."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    clean = re.sub(r'ObjectId\(".*?"\)', '"ObjectId"', raw)
    return json.loads(clean)

def normalize_ingredient(text):
    """Normalize ingredient name: lowercase, remove parentheses, extra spaces."""
    text = text.lower()
    text = re.sub(r"\(.*?\)", "", text)  # remove text in parentheses
    return text.strip()

def compare_iou(list1, list2):
    """Compute IoU similarity score between two ingredient lists."""
    set1 = {normalize_ingredient(i) for i in list1}
    set2 = {normalize_ingredient(i) for i in list2}
    intersection = set1 & set2
    union = set1 | set2
    return set1, set2, len(intersection) / len(union) if union else 1.0

# --- File Paths ---
file1_path = "C:/Users/Ld/Downloads/Neotex_env/halisi_cosmetics_genAI/json_output/1.json"
file2_path = "C:/Users/Ld/Downloads/Neotex_env/halisi_cosmetics_genAI/prompt_output/1_CheberElixir.json"

# --- Load Files ---
file1 = load_clean_json(file1_path)
file2 = load_clean_json(file2_path)

# --- Extract Ingredient Lists ---
ingredients1 = file1["Product Info"]["Product Sheet"].get("Key ingredients", {}).get("EN", [])
ingredients2 = file2["Product Info"]["Product Sheet"].get("Key ingredients", {}).get("EN", [])

# --- Compare ---
set1, set2, iou_score = compare_iou(ingredients1, ingredients2)

# --- Print Results ---
print("Normalized Ingredients (File 1):", set1)
print("Normalized Ingredients (File 2):", set2)
print("IoU Score:", round(iou_score, 3))
