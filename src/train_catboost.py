import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from catboost import CatBoostClassifier, Pool
import joblib

# ----------------------------
# CONSTANTS
# ----------------------------
DATA_FOLDER = "training_data"
MODEL_FOLDER = "model"
INPUT_FILE = "reviews_with_vader.csv"

NUMBEROFKFOLDS = 10
MAXITERS = 2000
RATE = 0.3
MAXTREEDEPTH = 8
RANDOMIZATION = 42
CHANCES = 200
UPDATEFREQUENCY = 200

# ----------------------------
# LOAD DATA
# ----------------------------
file_path = os.path.join(DATA_FOLDER, INPUT_FILE)
df = pd.read_csv(file_path)

# Drop duplicates & optional columns
df = df.drop_duplicates()
if "sentiment_category" in df.columns:
    df = df.drop(columns=["sentiment_category"])

# Target column
y = df["spam_label"].values

# Features (everything except label)
feature_cols = [col for col in df.columns if col != "spam_label"]
X = df[feature_cols].copy()

# Ensure group labels are consistent
groups = df["business_name"].fillna("unknown").astype(str).values

# Identify categorical + text features
cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object" and col != "text"]
text_features = [X.columns.get_loc("text")]

print("Categorical features:", [X.columns[i] for i in cat_features])
print("Text features:", [X.columns[i] for i in text_features])

# ----------------------------
# CROSS-VALIDATION TRAINING
# ----------------------------
os.makedirs(MODEL_FOLDER, exist_ok=True)

gkf = GroupKFold(n_splits=NUMBEROFKFOLDS)
oof_preds = np.zeros(len(df))
best_iterations = []

# Class weights for imbalance
genuine_count = (y == 0).sum()
spam_count = (y == 1).sum()
pos_weight = genuine_count / max(spam_count, 1)
class_weights = [1.0, float(pos_weight)]

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\nFold {fold+1}/{NUMBEROFKFOLDS}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_pool = Pool(X_train, label=y_train, cat_features=cat_features, text_features=text_features)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_features, text_features=text_features)

    model = CatBoostClassifier(
        iterations=MAXITERS,
        learning_rate=RATE,
        depth=MAXTREEDEPTH,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOMIZATION,
        early_stopping_rounds=CHANCES,
        verbose=UPDATEFREQUENCY,
        class_weights=class_weights
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    val_preds = model.predict_proba(val_pool)[:, 1]
    oof_preds[val_idx] = val_preds

    best_iterations.append(model.get_best_iteration() or MAXITERS)

    # Save each fold model
    fold_model_path = os.path.join(MODEL_FOLDER, f"fold_{fold+1}.cbm")
    model.save_model(fold_model_path)
    print(f"✅ Saved fold {fold+1} model → {fold_model_path}")
    print("Fold AP:", average_precision_score(y_val, val_preds))

print("\nOOF Average Precision:", average_precision_score(y, oof_preds))
print("OOF ROC-AUC:", roc_auc_score(y, oof_preds))

# ----------------------------
# THRESHOLD SELECTION
# ----------------------------
precision, recall, thresholds = precision_recall_curve(y, oof_preds)
f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
best_idx = np.nanargmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\nBest F1 = {best_f1:.4f} at threshold {best_threshold:.4f}")

# Save metadata
joblib.dump(feature_cols, os.path.join(MODEL_FOLDER, "features.pkl"))
joblib.dump(cat_features, os.path.join(MODEL_FOLDER, "cat_features.pkl"))
joblib.dump(text_features, os.path.join(MODEL_FOLDER, "text_features.pkl"))
joblib.dump(best_threshold, os.path.join(MODEL_FOLDER, "threshold.pkl"))

print("\n✅ Metadata saved to model/")

# ----------------------------
# ENSEMBLE INFERENCE (on training set for sanity check)
# ----------------------------
all_fold_preds = []
for fold in range(1, NUMBEROFKFOLDS+1):
    model = CatBoostClassifier()
    model.load_model(os.path.join(MODEL_FOLDER, f"fold_{fold}.cbm"))
    pool = Pool(X, cat_features=cat_features, text_features=text_features)
    fold_probs = model.predict_proba(pool)[:, 1]
    all_fold_preds.append(fold_probs)

final_probs = np.mean(all_fold_preds, axis=0)
final_preds = (final_probs >= best_threshold).astype(int)

train_results = df.copy()
train_results["probability_spam"] = final_probs
train_results["predicted_spam_label"] = final_preds

train_results.to_csv(os.path.join(DATA_FOLDER, "train_predictions.csv"), index=False)
print(f"✅ Ensemble predictions saved to {os.path.join(DATA_FOLDER, 'train_predictions.csv')}")
