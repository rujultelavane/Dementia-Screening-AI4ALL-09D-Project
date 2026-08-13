"""
=============================================================================
 SAVE THE FUSION MODEL  (deployable artifact + honest accuracy)
=============================================================================

TWO SEPARATE THINGS, kept separate on purpose:

  1. HONEST ACCURACY  -> comes from cross-validation (train_fusion_model.py).
     That ~0.767 measures how the RECIPE performs on unseen speakers.
     We recompute it here so it's printed right next to the saved model.

  2. DEPLOYABLE MODEL -> trained ONCE on ALL 352 recordings and saved.
     It uses every sample, so it has NO held-out data left -> its accuracy
     cannot be measured directly. Report the CV number (step 1) as "accuracy",
     NEVER a number computed on this saved model. They are different objects.

WHAT GETS SAVED (three pieces, because fusion has three models):
    model/audio_model.pkl   scale -> top-20 features -> balanced logreg
    model/text_model.pkl    scale -> balanced logreg
    model/meta_model.pkl    logreg that combines the two probabilities
    model/feature_lists.pkl the exact column names each model expects

 Input:  multimodal_aligned.csv        Needs: pandas, numpy, scikit-learn, joblib
=============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ---- config (identical label + feature setup to the evaluation script) ----
ALIGNED_CSV = "multimodal_aligned.csv"
SPEAKER_COL, DIAGNOSIS_COL = "speaker", "diagnosis"
POSITIVE_DX, NEGATIVE_DX = {"ProbableAD", "PossibleAD"}, {"Control"}
AUDIO_K, N_FOLDS = 20, 5
TEXT_FEATURES = ['n_words','n_unique_words','n_utterances','ttr','mlu','n_filled_pauses',
    'n_repetitions','n_pauses','pronoun_ratio','info_units_mentioned','n_nouns','n_verbs',
    'n_adjectives','pronoun_noun_ratio','pct_content_words','avg_sentence_length',
    'avg_dependency_tree_depth','mattr','mtld','readability_score','n_repeated_bigrams',
    'n_repeated_trigrams','n_vague_words','avg_word_length','clause_density','semantic_coherence']
META_COLS = {'file','group','file_stem','patient_id','patient_id_x','patient_id_y',
             'label','speaker','age','sex','diagnosis','education','dx_label','y'}

def make_audio(dim):
    return Pipeline([("sc", StandardScaler()),
                     ("sel", SelectKBest(f_classif, k=min(AUDIO_K, dim))),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
def make_text():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])

# ---- load + label + features (same as evaluation script) ----
df = pd.read_csv(ALIGNED_CSV)
df[DIAGNOSIS_COL] = df[DIAGNOSIS_COL].astype(str).str.strip()
keep = df[df[DIAGNOSIS_COL].isin(POSITIVE_DX | NEGATIVE_DX)].copy()
keep["y"] = keep[DIAGNOSIS_COL].isin(POSITIVE_DX).astype(int)
sides = keep.groupby(SPEAKER_COL)["y"].nunique()
keep = keep[~keep[SPEAKER_COL].isin(sides[sides > 1].index)].copy()

text_features = [c for c in TEXT_FEATURES if c in keep.columns]
audio_features = [c for c in keep.columns
                  if c not in META_COLS and c not in TEXT_FEATURES
                  and pd.api.types.is_numeric_dtype(keep[c])]
Xa = keep[audio_features].fillna(keep[audio_features].mean()).values
Xt = keep[text_features].fillna(keep[text_features].mean()).values
y  = keep["y"].values
groups = keep[SPEAKER_COL].astype(str).values
print(f"Data: {len(y)} recordings, {audio_features.__len__()} audio + {len(text_features)} text features")

# =============================================================================
# STEP 1 -- HONEST ACCURACY via speaker-independent cross-validation
#           (this is the number you REPORT)
# =============================================================================
outer = GroupKFold(n_splits=N_FOLDS)
meta = LogisticRegression()
stack_pred = np.zeros(len(y), dtype=int)
for tr, te in outer.split(Xa, y, groups):
    g_tr = groups[tr]
    inner = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(g_tr))))
    oof_a = cross_val_predict(make_audio(Xa.shape[1]), Xa[tr], y[tr], groups=g_tr, cv=inner, method="predict_proba")[:, 1]
    oof_t = cross_val_predict(make_text(),             Xt[tr], y[tr], groups=g_tr, cv=inner, method="predict_proba")[:, 1]
    meta.fit(np.column_stack([oof_a, oof_t]), y[tr])
    am = make_audio(Xa.shape[1]).fit(Xa[tr], y[tr]); tm = make_text().fit(Xt[tr], y[tr])
    pa = am.predict_proba(Xa[te])[:, 1]; pt = tm.predict_proba(Xt[te])[:, 1]
    stack_pred[te] = meta.predict(np.column_stack([pa, pt]))

cv_acc = accuracy_score(y, stack_pred)
cv_f1  = f1_score(y, stack_pred, average="macro")
print(f"\n>>> HONEST (cross-validated) fusion accuracy: {cv_acc:.3f}, macro-F1: {cv_f1:.3f}")
print(">>> THIS is the number to report for the saved model.")

# =============================================================================
# STEP 2 -- DEPLOYABLE MODEL: train each piece ONCE on ALL data, then save.
#           To combine them honestly, the meta-model is trained on OUT-OF-FOLD
#           probabilities (so it still never sees base models' cheating outputs),
#           while the base models themselves are refit on everything.
# =============================================================================
oof_a = cross_val_predict(make_audio(Xa.shape[1]), Xa, y, groups=groups, cv=outer, method="predict_proba")[:, 1]
oof_t = cross_val_predict(make_text(),             Xt, y, groups=groups, cv=outer, method="predict_proba")[:, 1]
final_meta = LogisticRegression().fit(np.column_stack([oof_a, oof_t]), y)

final_audio = make_audio(Xa.shape[1]).fit(Xa, y)
final_text  = make_text().fit(Xt, y)

os.makedirs("model", exist_ok=True)
joblib.dump(final_audio, "model/audio_model.pkl")
joblib.dump(final_text,  "model/text_model.pkl")
joblib.dump(final_meta,  "model/meta_model.pkl")
joblib.dump({"audio_features": audio_features, "text_features": text_features,
             "reported_cv_accuracy": float(cv_acc)}, "model/feature_lists.pkl")

print("\nSaved deployable fusion model to model/:")
print("  audio_model.pkl, text_model.pkl, meta_model.pkl, feature_lists.pkl")
print("\n!! Report accuracy as the CV number above (%.3f)." % cv_acc)
print("!! Do NOT compute accuracy on these saved models -- they trained on all data.")