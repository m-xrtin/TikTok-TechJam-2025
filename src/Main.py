import sys
import os
import json
import jsonlines
import random
from parse_file import *
from ucsd_json_standardization import *
from helpers import *

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
    
    success_message = "✅ File is successfully standardized!"
    
    output_file = os.path.splitext(input_filename)[0] + "_standardized.json"
    output_path = os.path.join(data_folder, output_file)

    # Special Case, JSON in UCSD Dataset format
    if ext == ".json" and is_ucsd_format:
        handle_ucsd_json(input_file, output_path, 1000, random.randint(1, 1000))
    # Otherwise, use GPT standardization
    else:
        try:
            result = parse_file(input_file, output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)            
        except Exception as e:
            print(f"Error parsing file: {e}")
            sys.exit(1)

    print(success_message)

if __name__ == "__main__":
    main()