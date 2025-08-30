import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")


"""
Preprocess a standardized CSV file from the data folder.
    
Args:
    base_name (str): The base filename (without .csv extension).
    Example: "combined" reads "data/combined.csv"
    
Returns:
df (pd.DataFrame): Preprocessed DataFrame
output_path (str): Path to the saved preprocessed CSV
"""
def preprocess_file(base_name: str):

    input_path = os.path.join(DATA_FOLDER, f"{base_name}.csv")
    output_path = os.path.join(DATA_FOLDER, f"{base_name}_preprocessed.csv")

    df = pd.read_csv(input_path)

    categorical_cols = ["sentiment_category", "gmap_id", "user_name", "business_name", "user_id"]
    numeric_cols = ["rating", "time"]
    text_cols = ["text"]

    # Drop sparse column
    df = df.drop(columns=["rating_category"], errors="ignore")

    # Fix categoricals
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    # Fix numerics
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-999)

    # Fix text
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    # Save processed file
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Saved to {output_path}")
    print("Final columns:", df.columns.tolist())

    return df, output_path

