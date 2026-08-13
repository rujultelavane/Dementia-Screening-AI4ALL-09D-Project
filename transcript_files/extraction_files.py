# extraction_pipeline.py
import re
from advanced_features import extract_advanced_features

def get_patient_speaking_from_text(text):
    """Same logic as your file-based version, but works on raw text content."""
    par_lines = []
    for line in text.splitlines():
        if line.startswith("*PAR:"):
            par_lines.append(line)
    return par_lines

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

def extract_features(par_lines):
    raw_text = " ".join(par_lines)

    n_filled_pauses = len(re.findall(r"&-\w+", raw_text))
    n_repetitions = len(re.findall(r"\[/+\]", raw_text))
    n_pauses = len(re.findall(r"\(\.+\)", raw_text))

    cleaned_lines = [clean_line(line) for line in par_lines]
    all_words = []
    for line in cleaned_lines:
        all_words.extend(line.lower().split())

    n_words = len(all_words)
    n_utterances = len(par_lines)
    n_unique_words = len(set(all_words))
    ttr = n_unique_words / n_words if n_words else 0
    mean_utterance_length = n_words / n_utterances if n_utterances else 0

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
        "mean_utterance_length": round(mean_utterance_length, 4),
        "n_filled_pauses": n_filled_pauses,
        "n_repetitions": n_repetitions,
        "n_pauses": n_pauses,
        "pronoun_ratio": round(pronoun_ratio, 4),
        "info_units_mentioned": info_units_mentioned,
    }

# The exact 11 features your model was trained on, in order
MODEL_FEATURES = [
    "info_units_mentioned",
    "n_words",
    "n_unique_words",
    "ttr",
    "mean_utterance_length",
    "avg_word_length",
    "avg_dependency_tree_depth",
    "avg_sentence_length",
    "pronoun_ratio",
    "pronoun_noun_ratio",
    "n_nouns",
    "n_verbs",
    "n_adjectives",
    "pct_content_words",
    "n_repetitions",
    "n_filled_pauses",
    "n_vague_words",
    "n_pauses",
    "mtld",
    "mattr",
    "readability_score",
    "n_repeated_bigrams",
    "n_repeated_trigrams",
    "clause_density",
    "semantic_coherence",
]

def process_transcript_text(text):
    """
    Takes the raw content of a .cha file (as a string) and returns
    a dict with exactly the 11 features the model expects, in order.
    """
    par_lines = get_patient_speaking_from_text(text)
    if not par_lines:
        raise ValueError("No *PAR: lines found — is this a valid CHAT-format transcript?")

    features = extract_features(par_lines)

    cleaned_full_text = " ".join(clean_line(line) for line in par_lines)
    advanced = extract_advanced_features(cleaned_full_text)
    features.update(advanced)

    # The fusion model calls mean utterance length "mlu". Keep both names so
    # the older text tab and the newer fusion model can use the same extractor.
    features["mlu"] = features["mean_utterance_length"]

    # Return the complete feature dictionary. The older text tab still selects
    # MODEL_FEATURES explicitly, while fusion selects its own saved feature list.
    return features
