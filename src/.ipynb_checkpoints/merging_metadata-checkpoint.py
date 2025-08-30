import json
import os
import jsonlines

    
"""
This is file is used for our training dataset. We incorporate the metadata from the 3 locations we chose: Alaska, California, New York into our training data.
With additional information from gmap_id and business name, our model can be more expressive and accurate.
"""

data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
reviews_file = os.path.join(data_folder, "review-California_standardized2.json")
metadata_file = os.path.join(data_folder, "meta-California.json")
output_file = os.path.join(data_folder, "review-California_meta2.json")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    with jsonlines.open(path) as reader:
        return [obj for obj in reader]

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def merge_with_metadata(reviews_file, metadata_file, output_file):
    reviews = load_json(reviews_file)
    metadata = load_jsonl(metadata_file)

    metadata_map = {m["gmap_id"]: m for m in metadata if "gmap_id" in m}

    for review in reviews:
        gmap_id = review.get("gmap_id")
        if gmap_id in metadata_map:
            review["business_name"] = metadata_map[gmap_id].get("name")
        else:
            review["business_name"] = None

    save_json(reviews, output_file)
    print(f"✅ Dataset is now populated with metadata!")

if __name__ == "__main__":
    merge_with_metadata(reviews_file, metadata_file, output_file)

