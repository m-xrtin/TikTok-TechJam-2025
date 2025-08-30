import os
import sys
import json
import random
from ucsd_json_standardization import *
from parse_file import *

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # script mode
except NameError:
    PROJECT_ROOT = os.path.abspath("..")
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "input")
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")

def standardize_file(input_file):
    # get basename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    ext = os.path.splitext(input_file)[1].lower()

    # rebuild full path inside /input
    full_input_path = os.path.join(INPUT_FOLDER, base_name + ext)
    output_path = os.path.join(DATA_FOLDER, base_name + "_standardized.json")

    if not os.path.exists(full_input_path):
        print(f"Error: {base_name+ext} not found in {INPUT_FOLDER}")
        sys.exit(1)

    if ext == ".json" and is_ucsd_format:
        handle_ucsd_json(full_input_path, output_path, 1000, random.randint(1, 1000))
    else:
        try:
            result = parse_file(full_input_path, output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error parsing file: {e}")
            sys.exit(1)

    print("File is successfully standardized!")
    return output_path