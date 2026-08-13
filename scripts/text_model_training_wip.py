"""
=============================================================================
 TEXT MODEL (Route 2: transformer embeddings) for dementia prediction
=============================================================================

THE ONLY NEW IDEA vs the audio script:
  For audio, openSMILE turned each recording into 88 numbers.
  Here, a pretrained language model turns each TRANSCRIPT into ~384 numbers
  (an "embedding") that encode its meaning. Those numbers then go into the
  exact same classifier setup you already have.

  So: embedding = the text equivalent of feature extraction.
      Everything after it is identical to the audio pipeline.

PIPELINE:
  1. Load transcripts (one row per recording: file, group, text).
  2. Clean the text -> plain natural language (strip CHAT/CLAN markup).
  3. Turn each transcript into a vector with a sentence-transformer.  <-- NEW
  4. Speaker-independent split (same discipline as the audio model).
  5. Scale + tune + compare SVM / logreg / random forest.
  6. Report honest accuracy + F1 on held-out speakers.

-----------------------------------------------------------------------------
 INSTALL (note: this pulls in PyTorch, which is a large download):
     pip install sentence-transformers scikit-learn pandas
 The FIRST run also downloads the language model (~90 MB) -- needs internet.
-----------------------------------------------------------------------------
 If you only have raw .cha files (not a text column), extract text first.
 The `pylangacq` library reads CHAT files and can give you participant-only
 speech -- do that, then save a CSV with columns [file, group, text].
=============================================================================
"""

import re
import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# =============================================================================
# CONFIG  -- edit to match your files, then run.
# =============================================================================

# CSV with one row per recording. Needs a text column and a label column.
TRANSCRIPTS_CSV = "transcripts.csv"

FILE_COLUMN  = "file"    # recording/file id (used to derive the speaker)
TEXT_COLUMN  = "text"    # the transcript text itself
GROUP_COLUMN = "group"   # label column; values like "dementia" / "control"
POSITIVE_LABEL = "dementia"   # which group counts as the "1" we predict

# Which pretrained model turns text into numbers.
#   "all-MiniLM-L6-v2"   -> fast, 384-dim, great default
#   "all-mpnet-base-v2"  -> slower, 768-dim, usually a bit more accurate
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cache embeddings to disk so you don't recompute them every run.
EMBEDDING_CACHE = "text_embeddings.npy"

# Derive a SPEAKER id from the file name so the same person never lands in
# both train and test. Adjust the regex if your names differ.
def speaker_id_from_name(name: str) -> str:
    m = re.match(r"([A-Za-z0-9]+)", str(name))
    return m.group(1) if m else str(name)

TEST_FRACTION = 0.2
RANDOM_SEED = 42


# =============================================================================
# 1. LOAD TRANSCRIPTS
# =============================================================================
print("Loading transcripts from:", TRANSCRIPTS_CSV)
df = pd.read_csv(TRANSCRIPTS_CSV)
df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)
df["_speaker"] = df[FILE_COLUMN].apply(speaker_id_from_name)


# =============================================================================
# 2. CLEAN THE TEXT for embeddings.
#
#    IMPORTANT: this is the OPPOSITE choice from Route 1. For hand-crafted
#    linguistic features you KEEP the CHAT markers (pauses, repetitions --
#    they're predictive). For embeddings you want plain natural language,
#    because the model was trained on ordinary text and the markup just
#    confuses it. So here we strip the annotation symbols out.
#
#    Also: make sure `text` is the PARTICIPANT's speech only, not the
#    interviewer's -- otherwise the model can cheat off interviewer prompts.
# =============================================================================
def clean_chat_text(t: str) -> str:
    t = str(t)
    t = re.sub(r"\[.*?\]", " ", t)      # remove [/], [//], [: ...] error/repair codes
    t = re.sub(r"&[-=]?\w+", " ", t)    # remove &-uh fillers and &=laughs actions
    t = re.sub(r"\(\.+\)", " ", t)      # remove (.) (..) pause markers
    t = re.sub(r"[<>]", " ", t)         # remove <...> scope markers
    t = re.sub(r"\bxxx\b", " ", t)      # unintelligible marker
    t = re.sub(r"[^\w\s.,!?']", " ", t) # drop leftover special symbols
    t = re.sub(r"\s+", " ", t).strip()  # collapse whitespace
    return t

df["_clean"] = df[TEXT_COLUMN].apply(clean_chat_text)

# Labels and speaker groups.
y = (df[GROUP_COLUMN] == POSITIVE_LABEL).astype(int).values
groups = df["_speaker"].values


# =============================================================================
# 3. SANITY CHECKS
# =============================================================================
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)
print(f"Transcripts:      {len(df)}")
print(f"Unique speakers:  {pd.Series(groups).nunique()}")
print(f"Label counts:     {pd.Series(y).value_counts().to_dict()}  (1 = {POSITIVE_LABEL})")
majority = pd.Series(y).value_counts(normalize=True).max()
print(f"Majority baseline: {majority:.1%}  (any real model must beat THIS)")
print("\nExample cleaned transcript (check it reads like plain speech):")
print("  ", df["_clean"].iloc[0][:300], "...")


# =============================================================================
# 4. TURN TRANSCRIPTS INTO NUMBERS  <-- the new part
#
#    model.encode() is the text equivalent of running openSMILE. Each
#    transcript becomes one vector. That's the whole "embedding" step.
#    (Long transcripts get truncated to a few hundred words; Cookie-Theft
#    descriptions are short, so this is rarely an issue here.)
# =============================================================================
if os.path.exists(EMBEDDING_CACHE):
    print(f"\nLoading cached embeddings from {EMBEDDING_CACHE}")
    X = np.load(EMBEDDING_CACHE)
else:
    print(f"\nComputing embeddings with '{EMBEDDING_MODEL}' (first run downloads it)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    X = embedder.encode(df["_clean"].tolist(), show_progress_bar=True)
    np.save(EMBEDDING_CACHE, X)
    print(f"Saved embeddings to {EMBEDDING_CACHE}")

print(f"Embedding matrix shape: {X.shape}  (rows = transcripts, cols = numbers per transcript)")


# =============================================================================
# 5. SPEAKER-INDEPENDENT TRAIN / TEST SPLIT   (same as the audio model)
# =============================================================================
splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_SEED)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train = groups[train_idx]

overlap = set(groups[train_idx]) & set(groups[test_idx])
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)} | Speaker overlap: {len(overlap)} (must be 0)")


# =============================================================================
# 6. DEFINE THE 3 RECIPES + hyperparameters.
#
#    Embeddings are high-dimensional (~384) vs your ~300 recordings, so
#    OVERFITTING is the risk here. That's why the grids lean on strong
#    regularization (small C = simpler model). Scaling still helps SVM/logreg.
# =============================================================================
models = {
    "Logistic Regression": {
        "pipe": Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=3000))]),
        "grid": {"clf__C": [0.001, 0.01, 0.1, 1]},
    },
    "SVM (RBF)": {
        "pipe": Pipeline([("scaler", StandardScaler()),
                          ("clf", SVC())]),
        "grid": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.001]},
    },
    "Random Forest": {
        "pipe": Pipeline([("scaler", StandardScaler()),
                          ("clf", RandomForestClassifier(random_state=RANDOM_SEED))]),
        "grid": {"clf__n_estimators": [300], "clf__max_depth": [None, 10]},
    },
}

inner_cv = GroupKFold(n_splits=min(5, pd.Series(groups_train).nunique()))


# =============================================================================
# 7. TRAIN + TUNE EACH MODEL, THEN TEST ON HELD-OUT SPEAKERS
# =============================================================================
results = {}
for name, spec in models.items():
    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    search = GridSearchCV(spec["pipe"], spec["grid"], cv=inner_cv,
                          scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train, groups=groups_train)

    preds = search.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print(f"Best settings: {search.best_params_}")
    print(f"HELD-OUT TEST accuracy: {acc:.3f}")
    print(f"HELD-OUT TEST F1:       {f1:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

    results[name] = {"accuracy": acc, "f1": f1, "best_params": search.best_params_}


# =============================================================================
# 8. SUMMARY -- this is your text model.
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY  (compare vs the majority baseline printed above)")
print("=" * 60)
for name, r in sorted(results.items(), key=lambda kv: kv[1]["f1"], reverse=True):
    print(f"  {name:22s}  acc={r['accuracy']:.3f}  f1={r['f1']:.3f}")

best = max(results, key=lambda k: results[k]["f1"])
print(f"\nBest text model: {best}")
print("Expect this to clearly beat the acoustic model -- text carries the real signal.")
print("Next: combine this model's probabilities with the audio model's (fusion).")