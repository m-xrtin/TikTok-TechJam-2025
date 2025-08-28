import sys
import os
import json
from parse_file import parse_file

def main():
    if len(sys.argv) != 2:
        print("Usage: python Main.py <input_file>")
        sys.exit(1)

    # Allows Main to access datasets from data folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(project_root, "data")
    input_file = os.path.join(data_folder, sys.argv[1])
    ext = os.path.splitext(input_file)[1].lower()
    
    # File integrity check
    if ext not in ['.json', '.txt', '.csv']:
        print("Error: Supported file types are .json, .txt, .csv")
        sys.exit(1)

    try:
        result = parse_file(input_file, "parsed_reviews.json")
    except Exception as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)

    output_file = os.path.splitext(input_file)[0] + "_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Output written to {output_file}")

if __name__ == "__main__":
    main()