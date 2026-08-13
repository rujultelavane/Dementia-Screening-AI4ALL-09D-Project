"""Compare transcript feature sets and classifiers with 5-fold CV."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "transcript_features_clean.csv")
X_all = df.drop(columns=["label"])
y_all = df["label"]

original = [
    "info_units_mentioned", "avg_word_length",
    "avg_dependency_tree_depth", "avg_sentence_length",
    "pronoun_ratio", "n_repetitions", "n_vague_words", "n_pauses",
    "mtld", "clause_density", "semantic_coherence",
]
word_features = ["n_words", "n_unique_words", "ttr", "mean_utterance_length"]
grammar_features = [
    "pronoun_noun_ratio", "n_nouns", "n_verbs", "n_adjectives",
    "pct_content_words", "avg_dependency_tree_depth", "avg_sentence_length",
]
fluency_features = [
    "n_filled_pauses", "n_repeated_bigrams", "n_repeated_trigrams",
    "n_repetitions", "n_pauses",
]
lexical_features = ["mtld", "mattr", "ttr", "avg_word_length", "readability_score"]

feature_sets = {
    "original_11": original,
    "original_plus_word": original + word_features,
    "original_plus_grammar": list(dict.fromkeys(original + grammar_features)),
    "original_plus_fluency": list(dict.fromkeys(original + fluency_features)),
    "original_plus_lexical": list(dict.fromkeys(original + lexical_features)),
    "all_features": list(X_all.columns),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
results = []

for feature_name, columns in feature_sets.items():
    columns = [c for c in columns if c in X_all.columns]
    X = X_all[columns]

    candidates = {
        "logistic_regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=3000)),
            ]),
            {
                "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "svm_rbf": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC()),
            ]),
            {
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__gamma": ["scale", 0.001, 0.01],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            Pipeline([
                ("classifier", RandomForestClassifier(random_state=42)),
            ]),
            {
                "classifier__n_estimators": [200, 500],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }

    for model_name, (pipeline, grid) in candidates.items():
        search = GridSearchCV(
            pipeline,
            grid,
            cv=cv,
            scoring=scoring,
            refit="accuracy",
            # One worker is more portable on macOS/Python environments that
            # restrict process semaphores; the dataset is small enough that
            # this remains practical.
            n_jobs=1,
        )
        search.fit(X, y_all)
        results.append({
            "feature_set": feature_name,
            "model": model_name,
            "n_features": len(columns),
            "accuracy": search.best_score_,
            "f1_macro": search.cv_results_["mean_test_f1_macro"][search.best_index_],
            "best_parameters": str(search.best_params_),
        })
        print(
            f"{feature_name:25s} {model_name:22s} "
            f"accuracy={search.best_score_:.4f} "
            f"f1={results[-1]['f1_macro']:.4f}"
        )

results_df = pd.DataFrame(results).sort_values(
    ["accuracy", "f1_macro"], ascending=False
)
results_df.to_csv(BASE_DIR / "text_model_comparison.csv", index=False)

print("\nBest configuration:")
print(results_df.iloc[0].to_string())
print(f"\nSaved results to {BASE_DIR / 'text_model_comparison.csv'}")
