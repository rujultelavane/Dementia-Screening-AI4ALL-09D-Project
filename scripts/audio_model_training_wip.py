"""
=============================================================================
 AUDIO MODEL for dementia prediction  (eGeMAPS / openSMILE features)
=============================================================================
 
WHAT THIS SCRIPT DOES, top to bottom:
  1. Loads your table of features (one row per recording, 88 numbers each).
  2. Runs SANITY CHECKS -- this is what catches the "coin-flip" bug.
  3. Splits the data so NO SPEAKER appears in both train and test
     (this is the honest-evaluation step we talked about).
  4. For each of the 3 recipes (SVM, logistic regression, random forest):
        - scales the features correctly (inside cross-validation, no leakage)
        - tries a few hyperparameter settings and keeps the best
  5. Reports each model's honest accuracy + F1 on the held-out test set.
 
The ACTUAL "training" is still just .fit() -- it happens inside sklearn's
GridSearchCV. Everything else here is the plumbing and the not-fooling-
yourself work.
 
-----------------------------------------------------------------------------
 pip install pandas scikit-learn   (that's all you need)
=============================================================================
"""
 
import re
import pandas as pd
import numpy as np
import joblib
import os
 
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
 
 
# =============================================================================
# CONFIG  -- edited to match acoustic_features.csv
# =============================================================================
 
FEATURES_CSV = "acoustic_features.csv"
 
NAME_COLUMN = "file"     # recording identifier column in your CSV
LABEL_COLUMN = "group"   # diagnosis label column in your CSV ("dementia"/"control")
 
LABELS_CSV = None
LABELS_JOIN_ON = "name"
LABELS_LABEL_COLUMN = "diagnosis"
 
def speaker_id_from_name(name: str) -> str:
    m = re.match(r"([A-Za-z0-9]+)", str(name))
    return m.group(1) if m else str(name)
 
TEST_FRACTION = 0.2
RANDOM_SEED = 42
 
 
# =============================================================================
# 1. LOAD THE DATA
# =============================================================================
print("Loading features from:", FEATURES_CSV)
df = pd.read_csv(FEATURES_CSV)
 
if LABELS_CSV is not None:
    labels_df = pd.read_csv(LABELS_CSV)
    df = df.merge(
        labels_df[[LABELS_JOIN_ON, LABELS_LABEL_COLUMN]],
        left_on=NAME_COLUMN, right_on=LABELS_JOIN_ON, how="inner"
    )
    LABEL_COLUMN = LABELS_LABEL_COLUMN
 
df["_speaker"] = df[NAME_COLUMN].apply(speaker_id_from_name)
 
non_feature = {LABEL_COLUMN, "_speaker"}
feature_columns = [
    c for c in df.columns
    if c not in non_feature
    and c != NAME_COLUMN
    and pd.api.types.is_numeric_dtype(df[c])
]
 
X = df[feature_columns].copy()
y = df[LABEL_COLUMN].copy()
groups = df["_speaker"].copy()
 
 
# =============================================================================
# 2. SANITY CHECKS
# =============================================================================
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)
print(f"Recordings (rows):        {len(df)}")
print(f"Feature columns detected: {len(feature_columns)}  (you expect 88)")
print(f"Unique speakers:          {groups.nunique()}")
print(f"Label values & counts:\n{y.value_counts()}")
 
if len(feature_columns) != 88:
    print("  !! WARNING: expected 88 features. Check your CSV columns.")
 
n_missing = X.isna().sum().sum()
print(f"Missing feature values:   {n_missing}")
if n_missing > 0:
    print("  Filling missing values with the column mean.")
    X = X.fillna(X.mean())
 
print("\nFirst 5 rows (name, speaker, label) -- check these look correct:")
print(df[[NAME_COLUMN, "_speaker", LABEL_COLUMN]].head().to_string(index=False))
 
majority = y.value_counts(normalize=True).max()
print(f"\nMajority-class baseline: {majority:.1%} "
      f"(any real model MUST beat this, not just beat 50%)")
 
 
# =============================================================================
# 3. SPEAKER-INDEPENDENT TRAIN / TEST SPLIT
# =============================================================================
splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                             random_state=RANDOM_SEED)
train_idx, test_idx = next(splitter.split(X, y, groups))
 
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]
 
overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
print(f"\nTrain recordings: {len(X_train)} | Test recordings: {len(X_test)}")
print(f"Speaker overlap between train/test: {len(overlap)}  (must be 0)")
 
 
# =============================================================================
# 4. DEFINE THE 3 RECIPES
# =============================================================================
models = {
    "Logistic Regression": {
        "pipe": Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000))]),
        "grid": {"clf__C": [0.01, 0.1, 1, 10]},
    },
    "SVM (RBF)": {
        "pipe": Pipeline([("scaler", StandardScaler()),
                          ("clf", SVC(probability=True))]),
        "grid": {"clf__C": [0.1, 1, 10],
                 "clf__gamma": ["scale", 0.01, 0.001]},
    },
    "Random Forest": {
        "pipe": Pipeline([("scaler", StandardScaler()),
                          ("clf", RandomForestClassifier(random_state=RANDOM_SEED))]),
        "grid": {"clf__n_estimators": [200, 400],
                 "clf__max_depth": [None, 5, 10]},
    },
}
 
inner_cv = GroupKFold(n_splits=min(5, groups_train.nunique()))
 
 
# =============================================================================
# 5. TRAIN + TUNE EACH MODEL, THEN TEST ON THE HELD-OUT SET
# =============================================================================
results = {}
fitted_pipelines = {}
 
for name, spec in models.items():
    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)
 
    search = GridSearchCV(
        estimator=spec["pipe"],
        param_grid=spec["grid"],
        cv=inner_cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=groups_train)
 
    preds = search.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
 
    print(f"Best settings found: {search.best_params_}")
    print(f"Cross-val F1 (on train, for tuning): {search.best_score_:.3f}")
    print(f"HELD-OUT TEST accuracy: {acc:.3f}")
    print(f"HELD-OUT TEST F1:       {f1:.3f}")
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))
 
    results[name] = {"accuracy": acc, "f1": f1, "best_params": search.best_params_}
    fitted_pipelines[name] = search.best_estimator_
 
 
# =============================================================================
# 6. SUMMARY -- pick the winner, refit on ALL data, and save it.
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY  (compare against the majority-class baseline above)")
print("=" * 60)
for name, r in sorted(results.items(), key=lambda kv: kv[1]["f1"], reverse=True):
    print(f"  {name:22s}  acc={r['accuracy']:.3f}  f1={r['f1']:.3f}")
 
best = max(results, key=lambda k: results[k]["f1"])
print(f"\nBest audio model: {best}")
 
# Refit the winning pipeline on ALL data (train+test) so the saved model
# uses every available sample, same pattern as your text model script.
best_pipeline = fitted_pipelines[best]
best_pipeline.fit(X, y)
 
os.makedirs("model", exist_ok=True)
model_path = "model/audio_model.pkl"
joblib.dump(best_pipeline, model_path)
print(f"\nSaved best audio model ({best}) to {model_path}")
print("This is the '.predict()' you'll later combine with the text model.")