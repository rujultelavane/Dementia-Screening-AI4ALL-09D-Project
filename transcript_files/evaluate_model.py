import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load your extracted features
PROJECT_ROOT = Path(__file__).resolve().parent
df = pd.read_csv(PROJECT_ROOT / "transcript_features_clean.csv")
X_all = df.drop(columns=["label"])
y_all = df["label"]

# Same pipeline structure you validated
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(
        C=100,
        gamma=0.001,
        class_weight='balanced',
        probability=True,
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Out-of-fold predictions — each prediction is from a model that never saw that sample
y_pred = cross_val_predict(pipeline, X_all, y_all, cv=cv)

# Metrics
accuracy = accuracy_score(y_all, y_pred)
precision = precision_score(y_all, y_pred, pos_label="Dementia")
recall = recall_score(y_all, y_pred, pos_label="Dementia")
f1 = f1_score(y_all, y_pred, pos_label="Dementia")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_all, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_all, y_pred))
