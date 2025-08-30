import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Only this is needed for VADER
nltk.download('vader_lexicon')

def VADER_Sentiment_Score(file_path, text_col="text", out_path=None):
    # Load
    df = pd.read_csv(file_path)

    # Safety checks
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV. Available: {list(df.columns)}")

    # Compute VADER scores
    sia = SentimentIntensityAnalyzer()
    texts = df[text_col].astype(str).fillna("")
    df["vader_score"] = texts.apply(lambda x: sia.polarity_scores(x)["compound"])

    # Map to categories
    def vader_sentiment_label(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["vader_category"] = df["vader_score"].apply(vader_sentiment_label)

    # Optional save
    if out_path:
        df.to_csv(out_path, index=False)
        
    print(df.head())
    
    df.to_csv("reviews_with_vader.csv", index=False)

    return df

#Just used here to test if it works
VADER_Sentiment_Score('all_data.csv')
