import jsonlines
import random
import json

""" Handles the special UCSD JSONL format. 
Since this dataset already provides structured fields, we can skip the slower GPT-based standardization and perform faster operations instead. 
This function randomly selects up to 1000 entries from the dataset and standardizes them into the project schema. 
The sampled subset can then be used for testing or as input to generate pseudo-labels for training. """

categories = [
    "user_id", "user_name", "business_name", "time", "text",
    "rating", "sentiment_category", "rating_category", "gmap_id"
]

# Dealing with float rating (dissimilar to kaggle data)
def get_rating_category(rating):
    if rating is None:
        return None
    try:
        rating = float(rating)
    except (ValueError, TypeError):
        return None
    
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    elif rating >= 4:
        return "positive"
    return None

def handle_ucsd_json(input_file: str, output_file: str, sample_size: int, seed: int):
    random.seed(seed)
    reservoir = []
    n = 0

    with jsonlines.open(input_file) as reader:
        for review in reader:
            n += 1
            if len(reservoir) < sample_size:
                reservoir.append(review)
            else:
                j = random.randint(0, n - 1)
                if j < sample_size:
                    reservoir[j] = review

    # standardize schema
    standardized = []
    for review in reservoir:
        entry = {}
        for field in categories:
            if field == "user_name":
                entry[field] = review.get("name", None)
            elif field == "rating_category":
                entry[field] = get_rating_category(review.get("rating"))
            else:
                entry[field] = review.get(field, None)
        standardized.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(standardized, f, ensure_ascii=False, indent=2)


