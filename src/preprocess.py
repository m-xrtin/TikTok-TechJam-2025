import pandas as pd

df = pd.read_csv("combined.csv")

categorical_cols = ["sentiment_category", "gmap_id", "user_name", "business_name", "user_id"]
numeric_cols = ["rating", "time"]
text_cols = ["text"]

# Drop only rating_category (too sparse)
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

df.to_csv("all_data.csv", index=False)
print("✅ Preprocessing complete. Saved to all_data.csv")
print("Final columns:", df.columns.tolist())
