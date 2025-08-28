import sys
import os
import json
import jsonlines
import random
from parse_file import parse_file
from ucsd_json_standardization import handle_ucsd_json

def main():
    
    if len(sys.argv) != 2:
        print("________________________________________________________________________________________________________________________________\n")
        print("Hello there! Our Program is a machine learning approach dedicated to filtering irrelevant/malicious google review comments!\n\n")
        print("To use our program, we have some basic requirements: jsonlines and openAI must both be downloaded on your local computer.\n\n")
        print("For simple installation, you can run pip install jsonlines/openai!\n\n")
        print("Please also make sure the file of interest is inside the 'data' folder.\n\n")
        print("Simply type into the terminal: python src/Main.py <file_name> to begin! (Example Input: python src/Main.py test.json)")
        sys.exit(1)

    # Always reference the data folder from project root
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    input_filename = os.path.basename(sys.argv[1])
    input_file = os.path.join(data_folder, input_filename)
    ext = os.path.splitext(input_file)[1].lower()
    
    # Check if file exists
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' does not exist in 'data' folder.")
        sys.exit(1)
        
    # File integrity check
    supported_formats = ['.json', '.txt', '.csv']
    success_message = "✅ File is successfully standardized!"
    
    if ext not in supported_formats:
        print("Warning!! This file format is not officially supported (This tool works best with JSON, CSV, or TXT).\n")
        answer = input(f"Are you sure you want to proceed? (yes/no): ").strip().lower()
        if answer != "yes":
            print("Exiting program, feel free to try again with supported file types.\n")
            sys.exit(1)
        else: 
            print("Proceeding with caution, Answers may be inconsistent.\n")
    
    # Special Case, JSON in UCSD Dataset format
    if ext == ".json":
        import jsonlines

        with jsonlines.open(input_file) as reader:
            data = [obj for obj in reader]

        # UCSD dataset format check
        if isinstance(data, list) and all(
            all(k in review for k in ("user_id", "text", "time", "gmap_id"))
            for review in data[:8]
            ):
            output_file = os.path.splitext(input_filename)[0] + "_sampled.json"
            output_path = os.path.join(data_folder, output_file)
            handle_ucsd_json(input_file, output_path, 1000, 67)
            print(success_message)
            sys.exit(1)

    try:
        result = parse_file(input_file, "parsed_reviews.json")
    except Exception as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)

    # Output parsed files into data directory for future use
    output_file = os.path.splitext(input_filename)[0] + "_output.json"
    output_path = os.path.join(data_folder, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(success_message)

if __name__ == "__main__":
    main()