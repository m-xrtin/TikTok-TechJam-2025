import os
import re
import json
import jsonlines
import pandas as pd
import sys
import csv

# -------------------------
# File validation
# -------------------------
def validate_file(input_file, supported_formats=None):
    if supported_formats is None:
        supported_formats = ['.json', '.txt', '.csv']

    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' does not exist in 'data' folder.")
        sys.exit(1)

    # Check extension
    ext = os.path.splitext(input_file)[1].lower()
    if ext not in supported_formats:
        print("Warning: This file format is not officially supported "
              f"(works best with {', '.join(supported_formats)}).\n")
        answer = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
        if answer != "yes":
            print("Exiting program, please use a supported file type.\n")
            sys.exit(1)
        else:
            print("Proceeding with caution — results may be inconsistent.\n")

    return ext

def json_to_csv_from_data(json_path, csv_path=None):
    """
    Converts a standardized JSON file from the data folder into CSV.

    Args:
        json_path (str): Path to input standardized JSON.
        csv_path (str, optional): Path to save CSV. 
                                  If None, auto-generates same name with .csv extension.
    Returns:
        str: Path to saved CSV file
    """

    # Auto-generate csv_path if not given
    if csv_path is None:
        base_name = os.path.splitext(json_path)[0]
        csv_path = base_name + ".csv"

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list of dicts")

    # Collect keys across all dicts
    all_keys = set()
    for entry in data:
        all_keys.update(entry.keys())
    all_keys = list(all_keys)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(data)

    print(f"Converted {len(data)} rows into {csv_path}")
    return csv_path