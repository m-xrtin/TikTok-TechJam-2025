import os
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool

DATA_FOLDER = "data"
MODEL_FOLDER = "model"
OUTPUT_FOLDER = "outputs"

def run_inference(base_name: str):
    input_path = os.path.join(DATA_FOLDER, f"{base_name}_final.csv")
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_results.csv")

    # Make sure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load input data
    df = pd.read_csv(input_path)

    # Load metadata
    feature_order = joblib.load(os.path.join(MODEL_FOLDER, "features.pkl"))
    cat_features = joblib.load(os.path.join(MODEL_FOLDER, "cat_features.pkl"))
    text_features = joblib.load(os.path.join(MODEL_FOLDER, "text_features.pkl"))
    best_threshold = joblib.load(os.path.join(MODEL_FOLDER, "threshold.pkl"))

    # Match training feature order
    X = df[feature_order]
    pool = Pool(X, cat_features=cat_features, text_features=text_features)

    # Load fold models
    models = []
    for i in range(1, 11):
        fold_path = os.path.join(MODEL_FOLDER, f"fold_{i}.cbm")
        if os.path.exists(fold_path):
            m = CatBoostClassifier()
            m.load_model(fold_path)
            models.append(m)

    if not models:
        raise RuntimeError("❌ No fold models found in model folder!")

    # Ensemble predictions
    fold_predictions = [m.predict_proba(pool)[:, 1] for m in models]
    probabilities = np.mean(fold_predictions, axis=0)
    decisions = (probabilities >= best_threshold).astype(int)

    # Add results
    df["probability"] = probabilities
    df["decision"] = decisions
    df["verdict"] = df["decision"].map({1: "likely spam", 0: "likely ham"})

    # Keep only requested columns
    keep_cols = ["vader_category", "time", "rating", "text", "user_name", "probability", "decision", "verdict"]
    result_df = df[keep_cols]

    # Save
    result_df.to_csv(output_path, index=False)
    print(f"✅ Inference complete. Results saved to {output_path}")

    # Quick summary
    n_spam = (result_df["decision"] == 1).sum()
    n_ham = (result_df["decision"] == 0).sum()
    print(f"Summary: {n_spam} likely spam, {n_ham} likely ham out of {len(result_df)} reviews.")

    return result_df, output_path
