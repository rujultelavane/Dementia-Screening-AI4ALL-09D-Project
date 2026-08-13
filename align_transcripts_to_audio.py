"""
=============================================================================
 ALIGN transcripts <-> acoustics using the .cha headers  (unblocks fusion)
=============================================================================

WHY THIS EXISTS:
  acoustic features are keyed by filename ("001-0.wav").
  transcript features are keyed by a TalkBank id ("11312/a-00090320-0").
  They don't share a key -> fusion couldn't run.

  Every .cha file's header contains BOTH:
      @PID:   11312/a-00090320-0   <- matches transcript_features.csv
      @Media: 001-0                <- matches acoustic_features.csv (001-0.wav)
  ...plus the @ID line with age / sex / diagnosis / education.

  So we scan the .cha files once, build the bridge, and attach the real
  filename (and demographics) to existing transcript rows. No feature
  recomputation -- your 24 linguistic features stay exactly as they are.

WHAT YOU GET OUT:
  1. transcript_features_keyed.csv  -> text features + 'file' + demographics
  2. multimodal_aligned.csv         -> ONLY recordings that have BOTH modalities
-----------------------------------------------------------------------------
 EDIT the three paths in CONFIG, then run.  Needs only: pandas
=============================================================================
"""

import os, re, glob
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
CHA_ROOT        = "transcript_files"                 # folder holding the .cha files (searched recursively)
TRANSCRIPT_CSV  = "transcript_files/transcript_features.csv"          # has patient_id + 24 features
ACOUSTIC_CSV    ="acoustic_features.csv"            # has file + 88 features

TRANSCRIPT_ID_COL = "patient_id"                     # column in TRANSCRIPT_CSV that equals @PID
ACOUSTIC_FILE_COL = "file"                            # column in ACOUSTIC_CSV, e.g. "001-0.wav"


# =============================================================================
# 1. PARSE EVERY .cha HEADER -> bridge table
# =============================================================================
def parse_cha(path):
    pid = media = None
    demo = {"age": None, "sex": None, "diagnosis": None, "education": None}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("@PID:"):
                pid = line.split("\t", 1)[1].strip()
            elif line.startswith("@Media:"):
                media = line.split("\t", 1)[1].split(",")[0].strip()
            elif line.startswith("@ID:") and "|PAR|" in line:
                f = line.split("\t", 1)[1].split("|")
                # CHAT @ID: lang|corpus|code|age|sex|group|SES|role|edu|...
                demo = {"age": f[3].rstrip(";"), "sex": f[4],
                        "diagnosis": f[5], "education": f[8]}
    # filename stem is the reliable key; @Media should agree with it
    stem = os.path.splitext(os.path.basename(path))[0]   # "001-0"
    if media and media != stem:
        print(f"  note: @Media '{media}' != filename stem '{stem}' in {path}")
    return {"patient_id": pid, "file_stem": stem, **demo, "source": path}

cha_files = glob.glob(os.path.join(CHA_ROOT, "**", "*.cha"), recursive=True)
print(f"Found {len(cha_files)} .cha files under '{CHA_ROOT}'")
bridge = pd.DataFrame([parse_cha(p) for p in cha_files])
bridge = bridge.dropna(subset=["patient_id"])
print(f"Parsed {len(bridge)} headers with a @PID.")
print("Diagnoses seen:", sorted(bridge['diagnosis'].dropna().unique()))

# normalize a clean binary label from the diagnosis (cross-check on your existing label)
bridge["dx_label"] = bridge["diagnosis"].apply(
    lambda d: "control" if str(d).lower().startswith("control") else "dementia")


# =============================================================================
# 2. ATTACH THE REAL FILENAME (+ demographics) TO YOUR TRANSCRIPT FEATURES
# =============================================================================
tf = pd.read_csv(TRANSCRIPT_CSV)
keyed = tf.merge(
    bridge[["patient_id", "file_stem", "age", "sex", "diagnosis", "education", "dx_label"]],
    left_on=TRANSCRIPT_ID_COL, right_on="patient_id", how="left"
)
matched = keyed["file_stem"].notna().sum()
print(f"\nTranscript rows matched to a .cha filename: {matched} / {len(keyed)}")
if matched < len(keyed):
    print("  (unmatched rows had a patient_id with no corresponding .cha file)")

keyed.to_csv("transcript_features_keyed.csv", index=False)
print("Wrote transcript_features_keyed.csv")


# =============================================================================
# 3. BUILD THE FUSION SET: recordings that have BOTH modalities
# =============================================================================
ac = pd.read_csv(ACOUSTIC_CSV)
ac["file_stem"] = ac[ACOUSTIC_FILE_COL].astype(str).str.replace(r"\.wav$", "", regex=True)

# speaker id = the participant number before the first '-' (e.g. "001-0" -> "001")
def speaker_of(stem): 
    return str(stem).split("-")[0]

merged = ac.merge(keyed, on="file_stem", how="inner", suffixes=("_ac", "_tx"))
merged["speaker"] = merged["file_stem"].apply(speaker_of)

print(f"\n=== FUSION SET ===")
print(f"Recordings with BOTH acoustic + text: {len(merged)}")
print(f"Unique speakers in fusion set:        {merged['speaker'].nunique()}")
print(f"Label balance:\n{merged['group'].value_counts() if 'group' in merged else merged['dx_label'].value_counts()}")

merged.to_csv("multimodal_aligned.csv", index=False)
print("\nWrote multimodal_aligned.csv  <- feed THIS to the fusion script")
print("(it has both feature sets, a 'speaker' column, and demographics for the audit)")