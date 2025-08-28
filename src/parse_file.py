import os
import csv
import json
import argparse
from openai import OpenAI

# Initialize client
gpt_api_key = "Your GPT_API_key"
client = OpenAI(api_key=gpt_api_key)

categories = [
    "user_id", "user_name", "business_name", "time", "text",
    "rating", "sentiment_category", "rating_category", "gmap_id"
]

def gpt_extract(line: str):
    prompt = f"""
    You are a data parser. You are given information about reviews for restaurants. 
    Normalize the following review into the schema:

    {", ".join(categories)}

    Rules:
    1. Always return valid JSON with exactly these keys.
    2. If a field is not present, return JSON null (not the string "null").
    3. user_name refers to the name of the commenter.
    4. Do not assign user_id automatically; leave it null if not provided in the input.
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
    9. Output only the JSON object, with no extra text.

    Review:
    {line}
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        store=True
    )

    try:
        return json.loads(response.output_text)
    except Exception as e:
        print("⚠️ Failed to parse GPT output:", e)
        print("Raw output:", response.output_text)
        return None

def parse_file(input_file, output_file):
    ext = os.path.splitext(input_file)[1].lower()
    parsed_reviews = []

    if ext == ".json":
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = gpt_extract(line)
                if parsed:
                    parsed_reviews.append(parsed)

    elif ext == ".csv":
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = gpt_extract(json.dumps(row))
                if parsed:
                    parsed_reviews.append(parsed)

    elif ext == ".txt":
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = gpt_extract(line)
                if parsed:
                    parsed_reviews.append(parsed)
    else:
        raise ValueError("❌ Unsupported file type. Only .json, .csv, and .txt are supported.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_reviews, f, indent=2, ensure_ascii=False)

    print(f"✅ Parsed {len(parsed_reviews)} reviews → saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse raw reviews file into standardized JSON using GPT")
    parser.add_argument("input", help="Input file (.json, .csv, .txt)")
    parser.add_argument("--output", default="parsed_reviews.json", help="Output JSON file")
    args = parser.parse_args()

    parse_file(args.input, args.output)
