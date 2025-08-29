import os
import sys

# file validation
def validate_file(input_file, supported_formats=None):
    if supported_formats is None:
        supported_formats = ['.json', '.txt', '.csv']

    if not os.path.isfile(input_file):
        print(f"❌ Error: File '{input_file}' does not exist in 'data' folder.")
        sys.exit(1)

    # Check extension
    ext = os.path.splitext(input_file)[1].lower()
    if ext not in supported_formats:
        print("⚠️ Warning: This file format is not officially supported "
              f"(works best with {', '.join(supported_formats)}).\n")
        answer = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
        if answer != "yes":
            print("Exiting program, please use a supported file type.\n")
            sys.exit(1)
        else:
            print("Proceeding with caution — results may be inconsistent.\n")

    return ext
