import os
import opensmile
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "revised_audio_files")
GROUPS = ["control", "dementia"]

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

rows = []
for group in GROUPS:
    folder = os.path.join(BASE_DIR, group)
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(folder, fname)
        feats = smile.process_file(path)
        feats.insert(0, "file", fname)
        feats.insert(1, "group", group)
        rows.append(feats)
        print(f"done: {group}/{fname}")

df = pd.concat(rows, ignore_index=True)
out_path = os.path.join(os.path.dirname(__file__), "..", "acoustic_features.csv")
df.to_csv(out_path, index=False)
print(f"\nextracted {len(df)} files, {df.shape[1]} columns")
print(df["group"].value_counts())