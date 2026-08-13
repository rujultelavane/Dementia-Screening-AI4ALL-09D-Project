"""
=============================================================================
 FUSION MODEL (capstone) -- combine audio + text for dementia screening
=============================================================================

FRAMING (decided deliberately):  ProbableAD + PossibleAD  vs  Control
  We DROP MCI, Vascular, Memory, Other, and blank diagnoses. Those labels are
  ambiguous (MCI sits between healthy and dementia), so keeping them would
  blur the very boundary the model is trying to learn. Expected clean set:
  ~189 AD vs ~163 control.

HOW IT COMBINES THE TWO MODELS (late fusion / stacking):
  audio features -> audio model -> probability
  text features  -> text  model -> probability
  those two probabilities -> meta-model -> final answer
  The meta-model LEARNS to trust the stronger (text) model more, so the weak
  audio model can't drag the result down the way a blind average could.

HONEST-EVALUATION MACHINERY (the whole point):
  * Everything is speaker-independent (GroupKFold on 'speaker') -- the same
    person never appears in both train and test, at ANY stage.
  * The meta-model is trained on OUT-OF-FOLD probabilities, so the base
    models never produced the probabilities from data they trained on.
  * We report a cross-validated estimate (not one lucky split), comparing
    audio-only, text-only, average-fusion, and stacked-fusion side by side.

 Input:  multimodal_aligned.csv  (from align_transcripts_to_audio.py)
 Needs:  pandas, numpy, scikit-learn
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =============================================================================
# CONFIG
# =============================================================================
ALIGNED_CSV   = "multimodal_aligned.csv"
SPEAKER_COL   = "speaker"
DIAGNOSIS_COL = "diagnosis"

POSITIVE_DX = {"ProbableAD", "PossibleAD"}   # -> label 1 (dementia/AD)
NEGATIVE_DX = {"Control"}                     # -> label 0 (healthy)
# everything else (MCI, Vascular, Memory, Other, blank) is dropped.

AUDIO_K = 20          # keep the 20 most discriminative acoustic features
N_FOLDS = 5
RANDOM_SEED = 42

TEXT_FEATURES = ['n_words','n_unique_words','n_utterances','ttr','mlu','n_filled_pauses',
    'n_repetitions','n_pauses','pronoun_ratio','info_units_mentioned','n_nouns','n_verbs',
    'n_adjectives','pronoun_noun_ratio','pct_content_words','avg_sentence_length',
    'avg_dependency_tree_depth','mattr','mtld','readability_score','n_repeated_bigrams',
    'n_repeated_trigrams','n_vague_words','avg_word_length','clause_density','semantic_coherence']

# columns that are NEVER features (metadata / ids / labels).
# 'y' MUST be here: it's the answer we add below, and if it leaked into the
# feature matrix the model would "predict" by peeking at the label.
META_COLS = {'file','group','file_stem','patient_id','patient_id_x','patient_id_y',
             'label','speaker','age','sex','diagnosis','education','dx_label','y'}


# =============================================================================
# 1. LOAD + BUILD THE AD-vs-CONTROL LABEL
# =============================================================================
df = pd.read_csv(ALIGNED_CSV)
df[DIAGNOSIS_COL] = df[DIAGNOSIS_COL].astype(str).str.strip()

keep = df[df[DIAGNOSIS_COL].isin(POSITIVE_DX | NEGATIVE_DX)].copy()
keep["y"] = keep[DIAGNOSIS_COL].isin(POSITIVE_DX).astype(int)

print("=" * 60)
print("LABEL DEFINITION: ProbableAD+PossibleAD vs Control")
print("=" * 60)
print(f"Kept {len(keep)} of {len(df)} recordings")
print(f"  AD (1):      {int(keep['y'].sum())}")
print(f"  Control (0): {int((keep['y']==0).sum())}")
print(f"  Dropped:     {len(df)-len(keep)}  (MCI/Vascular/Memory/Other/blank)")

# --- verify no speaker sits on BOTH sides (would be leakage) ---
sides = keep.groupby(SPEAKER_COL)["y"].nunique()
bad = sides[sides > 1].index.tolist()
if bad:
    print(f"  !! {len(bad)} speaker(s) on both sides -> dropping them: {bad}")
    keep = keep[~keep[SPEAKER_COL].isin(bad)].copy()
else:
    print("  Speaker check: no speaker appears on both sides. Good.")
print(f"Final: {len(keep)} recordings, {keep[SPEAKER_COL].nunique()} speakers")


# =============================================================================
# 2. IDENTIFY THE TWO FEATURE SETS
# =============================================================================
text_features = [c for c in TEXT_FEATURES if c in keep.columns]
audio_features = [c for c in keep.columns
                  if c not in META_COLS and c not in TEXT_FEATURES
                  and pd.api.types.is_numeric_dtype(keep[c])]

print(f"\nAudio features: {len(audio_features)} (expect 88)")
print(f"Text  features: {len(text_features)} (expect 26)")

Xa = keep[audio_features].fillna(keep[audio_features].mean()).values
Xt = keep[text_features].fillna(keep[text_features].mean()).values
y  = keep["y"].values
groups = keep[SPEAKER_COL].astype(str).values


# =============================================================================
# 3. THE TWO BASE MODELS
#    Audio: scale -> keep top-K features -> balanced logistic regression
#           (the ~63% recipe: balanced weights fix the imbalance, SelectKBest
#            drops noisy acoustic features).
#    Text:  scale -> balanced logistic regression (26 features, no selection).
#    Logistic regression is used partly because its probabilities are already
#    well-calibrated, so "0.7" means about the same thing from both models --
#    which is what makes combining them sound.
# =============================================================================
def make_audio():
    return Pipeline([("sc", StandardScaler()),
                     ("sel", SelectKBest(f_classif, k=min(AUDIO_K, Xa.shape[1]))),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])

def make_text():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])


# =============================================================================
# 4. CROSS-VALIDATED FUSION (speaker-independent, leakage-safe stacking)
#
#    Outer GroupKFold gives every recording an honest test prediction from
#    models that never saw its speaker. Inside each outer-train fold we build
#    out-of-fold probabilities to train the meta-model, so the meta-model also
#    never sees inflated inputs.
# =============================================================================
outer = GroupKFold(n_splits=N_FOLDS)
meta = LogisticRegression()

# collect out-of-fold TEST predictions for all four approaches
pred = {k: np.zeros(len(y), dtype=int) for k in ["audio", "text", "avg", "stack"]}

for tr, te in outer.split(Xa, y, groups):
    g_tr = groups[tr]
    inner = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(g_tr))))

    # out-of-fold base probabilities on the training portion -> train meta
    oof_a = cross_val_predict(make_audio(), Xa[tr], y[tr], groups=g_tr,
                              cv=inner, method="predict_proba")[:, 1]
    oof_t = cross_val_predict(make_text(),  Xt[tr], y[tr], groups=g_tr,
                              cv=inner, method="predict_proba")[:, 1]
    meta.fit(np.column_stack([oof_a, oof_t]), y[tr])

    # fit base models on the full training portion, predict the held-out fold
    am, tm = make_audio().fit(Xa[tr], y[tr]), make_text().fit(Xt[tr], y[tr])
    pa = am.predict_proba(Xa[te])[:, 1]
    pt = tm.predict_proba(Xt[te])[:, 1]

    pred["audio"][te] = (pa >= 0.5).astype(int)
    pred["text"][te]  = (pt >= 0.5).astype(int)
    pred["avg"][te]   = ((pa + pt) / 2 >= 0.5).astype(int)
    pred["stack"][te] = meta.predict(np.column_stack([pa, pt]))


# =============================================================================
# 5. RESULTS
# =============================================================================
baseline = max(y.mean(), 1 - y.mean())
print("\n" + "=" * 60)
print(f"CROSS-VALIDATED RESULTS  (majority baseline = {baseline:.3f})")
print("=" * 60)
labels = {"audio": "Audio only", "text": "Text only",
          "avg": "Fusion: average", "stack": "Fusion: stacking"}
for k in ["audio", "text", "avg", "stack"]:
    acc = accuracy_score(y, pred[k])
    f1  = f1_score(y, pred[k], average="macro")
    print(f"  {labels[k]:20s}  acc={acc:.3f}  f1={f1:.3f}")

print("\nStacking confusion matrix (rows=true, cols=pred; 0=control,1=AD):")
print(confusion_matrix(y, pred["stack"]))

# interpretable payoff: how much did stacking lean on each modality?
full_a = cross_val_predict(make_audio(), Xa, y, groups=groups,
                           cv=outer, method="predict_proba")[:, 1]
full_t = cross_val_predict(make_text(),  Xt, y, groups=groups,
                           cv=outer, method="predict_proba")[:, 1]
meta.fit(np.column_stack([full_a, full_t]), y)
wa, wt = meta.coef_[0]
print(f"\nLearned stacking weights -> audio: {wa:.2f}, text: {wt:.2f}")
print("(larger = the meta-model trusts that modality more; expect text > audio)")
print("\nNext: fairness audit using the age/sex/education columns now in this file.")
