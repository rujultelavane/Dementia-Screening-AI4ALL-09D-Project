from pathlib import Path
from advanced_features import extract_advanced_features
import re
import pandas as pd

root = Path(__file__).resolve().parent

def get_patient_id_from_file(filepath):
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"@PID:\s*(\S+)", text)
    if match:
        return match.group(1)
    return filepath.stem  # fallback to filename if no @PID line found

import re

def clean_line(line):
    line = line.replace("*PAR:", "")
    line = re.sub(r"\x15\d+_\d+\x15", "", line)
    line = re.sub(r"[<>]", "", line)
    line = re.sub(r"&-\w+", "", line)
    line = re.sub(r"\[/+\]", "", line)
    line = re.sub(r"\(\.+\)", "", line)
    line = re.sub(r"[.?!,]", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()

#extracting only the patient speaking
def get_patient_speaking(filepath):
    par_lines = []
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if line.startswith("*PAR:"):
            par_lines.append(line)   
    return par_lines

def extract_features(par_lines):
    raw_text = " ".join(par_lines)

    # disfluency counts — must use RAW lines, since markers get removed during cleaning
    n_filled_pauses = len(re.findall(r"&-\w+", raw_text))
    n_repetitions = len(re.findall(r"\[/+\]", raw_text))
    n_pauses = len(re.findall(r"\(\.+\)", raw_text))

    # word-level stats — use CLEANED lines
    cleaned_lines = [clean_line(line) for line in par_lines]
    all_words = []
    for line in cleaned_lines:
        all_words.extend(line.lower().split())

    n_words = len(all_words)
    n_unique_words = len(set(all_words))
    ttr = n_unique_words / n_words if n_words else 0

    n_utterances = len(par_lines)
    mlu = n_words / n_utterances if n_utterances else 0

    pronouns = {"i", "you", "he", "she", "it", "we", "they", "this", "that", "thing"}
    n_pronouns = sum(1 for w in all_words if w in pronouns)
    pronoun_ratio = n_pronouns / n_words if n_words else 0

    info_unit_words = ["boy", "girl", "woman", "mother", "cookie", "jar", "stool",
                        "sink", "dish", "plate", "water", "overflow", "curtain",
                        "window", "cupboard", "counter"]
    joined = " ".join(all_words)
    info_units_mentioned = sum(1 for unit in info_unit_words if unit in joined)

    return {
        "n_words": n_words,
        "n_unique_words": n_unique_words,
        "n_utterances": n_utterances,
        "ttr": round(ttr, 4),
        "mlu": round(mlu, 4),
        "n_filled_pauses": n_filled_pauses,
        "n_repetitions": n_repetitions,
        "n_pauses": n_pauses,
        "pronoun_ratio": round(pronoun_ratio, 4),
        "info_units_mentioned": info_units_mentioned,
    }

def process_one_file(filepath, label):
    patient_id = get_patient_id_from_file(filepath)
    par_lines = get_patient_speaking(filepath)   
    features = extract_features(par_lines)

    cleaned_full_text = " ".join(clean_line(line) for line in par_lines)
    advanced = extract_advanced_features(cleaned_full_text)
    features.update(advanced)


    features["patient_id"] = patient_id
    features["label"] = label
    return features

if __name__ == "__main__":
    root = Path("transcript_files")
    rows = []

    for label in ["Control", "Dementia"]:
        folder = root / label / "cookie"
        for filepath in folder.glob("*.cha"):
            rows.append(process_one_file(filepath, label))

    df = pd.DataFrame(rows)
    df.to_csv("transcript_features.csv", index=False)
    print(df)