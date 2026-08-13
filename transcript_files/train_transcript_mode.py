

import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---- 1. Load your extracted features ----
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "transcript_features_clean.csv")

X_all = df.drop(columns=["label"])
y_all = df["label"]

print(f"Loaded {len(df)} samples with {X_all.shape[1]} features.")
print(df["label"].value_counts())

# ---- 2. Build the pipeline (same structure you validated with k-fold) ----
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(
        C=100,
        gamma=0.001,
        class_weight='balanced',
        probability=True,
    ))
])

# ---- 3. Train on ALL data ----
final_pipeline.fit(X_all, y_all)
print("Model trained on full dataset.")

# ---- 4. Save the model ----
model_dir = BASE_DIR / "model"
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "logistic_regression_pipeline.pkl"
joblib.dump(final_pipeline, model_path)
print(f"Model saved to {model_path}")

# ---- 5. Sanity check: reload and test ----
loaded_model = joblib.load(model_path)

sample = X_all.iloc[:5]
predictions = loaded_model.predict(sample)
probabilities = loaded_model.predict_proba(sample)


print("Predictions:", predictions)
print("Probabilities:\n", probabilities)
print("Actual labels:", y_all.iloc[:5].tolist())
