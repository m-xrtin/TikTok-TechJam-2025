import os
import csv
import json
import argparse
from openai import OpenAI

# Initialize client
gpt_api_key = "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"
client = OpenAI(api_key=gpt_api_key)

categories = [
    "user_id", "user_name", "business_name", "time", "text",
    "rating", "sentiment_category", "rating_category", "gmap_id"
]

def gpt_extract(reviews: list[str]):
    prompt = f"""
    You are a data parser. You are given information about reviews for restaurants. 
    Prioritize runtime.
    Normalize the following reviews into the schema:

    {", ".join(categories)}
    
    Rules:
    1. Always return a valid JSON array of objects, exactly with these keys.
    2. If a field is not present, return null.
    3. user_name refers to the name of the commenter.
    4. user_id: do not invent IDs. If not provided, set to null.
    5. business_name may not exist. If gmap_id is present, include it. If both are missing, set both to null.
    6. sentiment_category:
       - rating 1-2: "negative"
       - rating 3: "neutral"
       - rating 4-5: "positive"
       - if no rating is present, infer from text sentiment if possible, otherwise null.
    7. rating_category:
       - If the dataset has a thematic category field (taste, menu, indoor_atmosphere, outdoor_atmosphere), use it.
       - If not, infer from text if possible.
       - If no theme can be determined, set to null.
    8. time may not be available. If not present, set to null.
    9. Output only the JSON array, with no extra text.
    10. Limit output to a maximum of 1000 entries.

    Review:
    {json.dumps(reviews, ensure_ascii=False)}
    """

    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=prompt,
        store=True
    )

    try:
        return safe_json_loads(response.output_text)
    except Exception as e:
        print("⚠️ Failed to parse GPT output:", e)
        print("Raw output:", response.output_text)
        return None
    
def safe_json_loads(output_text: str):
    raw = output_text.strip()

    # Solution for GPT Output failing to be recognized as Json
    if raw.startswith("```"):
        # drop first and last line
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("\n", 1)[0]

    return json.loads(raw)

# Used for chunking input for faster runtime
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]

# Parses File for GPT Cleaning
def parse_file(input_file, output_file, batch_size=10):
    ext = os.path.splitext(input_file)[1].lower()
    raw_reviews = []

    if ext == ".json":  
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f) 
            raw_reviews = [json.dumps(obj) for obj in data]

    elif ext == ".csv":
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_reviews = [json.dumps(row) for row in reader]

    elif ext in [".txt", ".jsonl"]:  
        with open(input_file, "r", encoding="utf-8") as f:
            raw_reviews = [line.strip() for line in f if line.strip()]

    else:
        raise ValueError(f"❌ Unsupported file type: {ext}")

    # Process in batches
    parsed_reviews = []
    for chunk in chunk_list(raw_reviews, batch_size):
        batch_result = gpt_extract(chunk)
        if batch_result:
            parsed_reviews.extend(batch_result)

    return parsed_reviews
