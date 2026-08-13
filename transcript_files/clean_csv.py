import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "transcript_features.csv")

keep_features = [
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
    "label",   
]

# only keep columns that actually exist, in case a name is slightly different
keep_features = [c for c in keep_features if c in df.columns]

df_clean = df[keep_features]
df_clean.to_csv(BASE_DIR / "transcript_features_clean.csv", index=False)
print(df_clean)
