"""
Extended feature extraction — Tier 1 & Tier 2 additions.
Requires: pip install spacy lexicalrichness textstat
          python -m spacy download en_core_web_sm
"""
import spacy
from lexicalrichness import LexicalRichness
import textstat
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")
VAGUE_WORDS = {"thing", "things", "stuff", "something", "somewhere", "someone", "whatever"}
def tree_depth(token):
    """Recursively measure how deep a dependency tree goes from this token."""
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(tree_depth(child) for child in children)

def extract_coherence_features(doc):
    """
    doc: a spaCy doc object (already processed text)
    """
    sentences = list(doc.sents)

    # ---- Clause density: subordinate/dependent clauses per sentence ----
    clause_labels = {"advcl", "acl", "ccomp", "xcomp", "relcl", "mark"}
    n_clauses = sum(1 for token in doc if token.dep_ in clause_labels)
    clause_density = n_clauses / len(sentences) if sentences else 0

    # ---- Semantic coherence: similarity between consecutive sentences ----
    sentence_texts = [sent.text for sent in sentences if len(sent.text.strip()) > 0]
    if len(sentence_texts) >= 2:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentence_texts)
        similarities = []
        for i in range(len(sentence_texts) - 1):
            sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i + 1])[0][0]
            similarities.append(sim)
        coherence_score = sum(similarities) / len(similarities)
    else:
        coherence_score = 0

    return {
        "clause_density": round(clause_density, 4),
        "semantic_coherence": round(coherence_score, 4),
    }

def extract_advanced_features(cleaned_text):
    """
    cleaned_text: one string containing ALL of a patient's cleaned speech
                  (join your cleaned_lines together before calling this)
    """
    doc = nlp(cleaned_text)
    words = [t.text.lower() for t in doc if not t.is_punct]
    # ---- POS counts ----
    pos_counts = Counter(token.pos_ for token in doc)
    n_pronouns = pos_counts.get("PRON", 0)
    n_nouns = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    n_verbs = pos_counts.get("VERB", 0)
    n_adjectives = pos_counts.get("ADJ", 0)
    pronoun_noun_ratio = n_pronouns / n_nouns if n_nouns else 0
    n_content_words = n_nouns + n_verbs + n_adjectives + pos_counts.get("ADV", 0)
    pct_content_words = n_content_words / len(words) if words else 0
    # ---- sentence-level stats ----
    sentences = list(doc.sents)
    sent_lengths = [len([t for t in sent if not t.is_punct]) for sent in sentences]
    avg_sentence_length = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
    depths = [tree_depth(sent.root) for sent in sentences if len(sent) > 0]
    avg_tree_depth = sum(depths) / len(depths) if depths else 0
    # ---- lexical richness ----
    try:
        lex = LexicalRichness(cleaned_text)
        mattr = lex.mattr(window_size=min(25, lex.words)) if lex.words >= 2 else 0
        mtld = lex.mtld() if lex.words >= 2 else 0
    except Exception:
        mattr, mtld = 0, 0
    # ---- readability ----
    # textstat may require the optional NLTK cmudict resource. Keep feature
    # extraction usable when that resource is unavailable.
    try:
        readability = textstat.flesch_reading_ease(cleaned_text) if words else 0
    except (LookupError, Exception):
        readability = 0
    # ---- bigram / trigram repetition ----
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    n_repeated_bigrams = sum(1 for c in Counter(bigrams).values() if c > 1)
    n_repeated_trigrams = sum(1 for c in Counter(trigrams).values() if c > 1)
    # ---- vague words & avg word length ----
    n_vague_words = sum(1 for w in words if w in VAGUE_WORDS)
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    coherence_features = extract_coherence_features(doc)

    return {
        "n_nouns": n_nouns,
        "n_verbs": n_verbs,
        "n_adjectives": n_adjectives,
        "pronoun_noun_ratio": round(pronoun_noun_ratio, 4),
        "pct_content_words": round(pct_content_words, 4),
        "avg_sentence_length": round(avg_sentence_length, 4),
        "avg_dependency_tree_depth": round(avg_tree_depth, 4),
        "mattr": round(mattr, 4),
        "mtld": round(mtld, 4),
        "readability_score": round(readability, 2),
        "n_repeated_bigrams": n_repeated_bigrams,
        "n_repeated_trigrams": n_repeated_trigrams,
        "n_vague_words": n_vague_words,
        "avg_word_length": round(avg_word_length, 4),
        **coherence_features,   # unpacks clause_density and semantic_coherence into this dictionary
    }
