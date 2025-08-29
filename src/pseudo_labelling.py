import os
import json
from openai import OpenAI
import time

client = OpenAI(api_key="ecret")

# Directories
root = os.path.dirname(os.path.dirname(__file__))
data_folder = os.path.join(root, "data")
training_folder = os.path.join(root, "training_data")
os.makedirs(training_folder, exist_ok=True)

prompt = """
You are a simple Spam detector for pseudo labelling.

You will receive a JSON array of Google reviews. For each review, output an array of JSON objects
that strictly follow this schema:

{
  "user_id": string | null,
  "user_name": string | null,
  "business_name": string | null,
  "time": number | null,
  "text": string | null,
  "rating": number | null,
  "sentiment_category": "positive" | "neutral" | "negative" | null,
  "rating_category": "taste" | "menu" | "indoor" | "outdoor" | null,
  "gmap_id": string | null,
  "spam_label": 0 | 1
}

Rules:
1. Keep `sentiment_category` as is, if already present and not null.
2. If `sentiment_category` is null, infer it from text/rating if possible, else leave null.
3. Infer `rating_category` only if the review clearly refers to taste, menu, indoor, or outdoor. Otherwise null.
4. Infer `spam_label`:
   - Ham = 0
   - Spam = 1
5. Always return valid JSON with exactly the above keys, no extra commentary.
"""

def gpt_label(reviews_batch):
    response = client.responses.create(
        model="gpt-4.1-mini-2025-04-14",
        input=prompt + "\n\nReviews:\n" + json.dumps(reviews_batch, ensure_ascii=False),
    )
    try:
        output_text = response.output_text.strip()
        if output_text.startswith("```"):
            output_text = output_text.split("\n", 1)[1].rsplit("\n", 1)[0]
        return json.loads(output_text)
    except Exception as e:
        print("⚠️ Error parsing GPT output:", e)
        print("Raw output:", response.output_text)
        return []

def chunk_list(lst, size=27):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def pseudo_label_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    labeled_reviews = []
    for chunk in chunk_list(reviews, size=50):
        labeled_reviews.extend(gpt_label(chunk))
    time.sleep(20)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled_reviews, f, ensure_ascii=False, indent=2)

    print("Done.")

if __name__ == "__main__":
    for filename in os.listdir(data_folder):
        if "Kaggle" in filename:
            input_path = os.path.join(data_folder, filename)
            output_path = os.path.join(training_folder, filename.replace("_meta", "_pseudo"))
            pseudo_label_file(input_path, output_path)
