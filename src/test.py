import json
import csv
import os

# Paths
training_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training_data")
json_path = os.path.join(training_folder, "review-Kaggle_meta.json")
csv_path = os.path.join(training_folder, "2.csv")

# Load JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Ensure data is a list of dicts
if not isinstance(data, list):
    raise ValueError("JSON must be a list of objects/dicts")

# Collect all keys across all JSON objects (handles missing fields)
all_keys = set()
for entry in data:
    all_keys.update(entry.keys())

all_keys = list(all_keys)

# Write CSV
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(data)

print(f"✅ Converted {len(data)} rows into {csv_path}")
