import os
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

DATA_FOLDER = "data"

# Download lexicon quietly once
nltk.download('vader_lexicon', quiet=True)

"""
Run VADER sentiment scoring on a preprocessed CSV.

Args:
    base_name (str): The base filename without extension.
                     Example: "review-Kaggle" reads "data/review-Kaggle_preprocessed.csv"

Returns:
    df (pd.DataFrame): DataFrame with added VADER columns
    output_path (str): Path to the saved *_final.csv
"""
def VADER_Sentiment_Score(base_name: str, text_col="text"):
    input_path = os.path.join(DATA_FOLDER, f"{base_name}_preprocessed.csv")
    output_path = os.path.join(DATA_FOLDER, f"{base_name}_final.csv")

    df = pd.read_csv(input_path)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {list(df.columns)}")

    # Initialize analyzer
    sia = SentimentIntensityAnalyzer()
    texts = df[text_col].astype(str).fillna("")

    # Add compound score + category
    df["vader_score"] = texts.apply(lambda x: sia.polarity_scores(x)["compound"])

    def vader_sentiment_label(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["vader_category"] = df["vader_score"].apply(vader_sentiment_label)

    # Save
    df.to_csv(output_path, index=False)
    print(f"✅ VADER sentiment scoring complete. Saved to {output_path}")

    return df, output_path
