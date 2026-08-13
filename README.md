# AI4ALL Group09D — Dementia Screening from Picture-Description Tasks

A research project exploring whether **dementia can be screened from
the "Cookie Theft" picture-description task** — using three independent, separately
trained models:

- **Text model** — hand-crafted linguistic features extracted from a CHAT-format
  (`.cha`) transcript, fed into a **Logistic Regression** pipeline.
- **Audio model** — 88 eGeMAPS acoustic features (pitch, loudness, voice quality,
  timing) extracted from the raw recording with
  [openSMILE](https://audeering.github.io/opensmile-python/), fed into an
  **SVM (RBF kernel)** pipeline.
- **Fusion model** - Leakage-safe (out-of-fold) stacking combines both verbal and
  audio signals. Best accuracy out of the three models.

This is a demo/screening tool for a class project, **not a diagnostic instrument**.

## Dataset

[DementiaBank English Pitt Corpus](https://dementia.talkbank.org/) (noise-reduced
version, DOI: `10.21415/CQCW-1F92`) — Cookie Theft picture-description recordings
and CHAT transcripts from Control and Dementia participants.

- Text: 552 transcripts (243 Control / 309 Dementia)
- Audio: 395 recordings (163 Control / 232 Dementia)

## Models & methodology

### Text model

Eleven features extracted per transcript (`transcript_files/extraction_files.py`,
`transcript_files/advanced_features.py`):

`info_units_mentioned`, `avg_word_length`, `avg_dependency_tree_depth`,
`avg_sentence_length`, `pronoun_ratio`, `n_repetitions`, `n_vague_words`,
`n_pauses`, `mtld` (lexical diversity), `clause_density`, `semantic_coherence`.

Trained with `StandardScaler` + `LogisticRegression` in an sklearn `Pipeline`.
Validated with 5-fold stratified cross-validation: **76.3% ± 3.8% accuracy**.

### Audio model

88 eGeMAPS functional features (pitch, loudness, spectral, and timing statistics)
extracted per recording with openSMILE. Trained with `StandardScaler` + `SVC(rbf)`.
Validated with 5-fold **speaker-grouped** cross-validation (no speaker appears in
both train and test folds): **61.8% ± 2.5% accuracy**.

The text model is meaningfully stronger — consistent with the exploratory
comparisons of Logistic Regression / Random Forest / SVM on the acoustic features
in `model_comparison.png`, `roc_curves.png`, `confusion_matrices.png`, and
`rf_feature_importance.png` (produced by `modelevaluation`), which is what led to
picking the SVM for the deployed audio pipeline.

## Repository layout

```
streamlit_app.py            Deployed demo app (Streamlit) — both models, live predictions
transcript_files/           Text pipeline: transcripts, feature extraction, trained model
  extraction_files.py         Core 11-feature extractor used by the deployed app
  advanced_features.py        spaCy/lexicalrichness/textstat-based feature extraction
  train_transcript_mode.py    Trains the final Logistic Regression pipeline on all data
  evaluate_model.py           Honest 5-fold stratified CV evaluation
  clean_csv.py                Trims the raw feature CSV down to the model's 11 columns
  app.py                      Standalone FastAPI service for the text model (alternative to the Streamlit app)
  model/                      Saved logistic_regression_pipeline.pkl
  Control/ Dementia/          Raw .cha transcripts by group
scripts/
  extract_features.py          Runs openSMILE over revised_audio_files/ -> acoustic_features.csv
  remove_instructor_audio.py   Trims interviewer speech out of the raw recordings, using transcript timestamps
  audio_model_training_wip.py  Grouped train/test split + model selection (SVM/LogReg/RandomForest) for audio
  text_model_training_wip.py   Exploratory alternative text pipeline using sentence-transformer embeddings
model/                        Saved audio_model.pkl (deployed SVM pipeline)
audio_files/                  Original recordings by group
revised_audio_files/          Recordings with instructor audio removed (used for feature extraction)
acoustic_features.csv         88-dim eGeMAPS features, one row per recording
modelevaluation               Exploratory acoustic model comparison (LogReg/RF/SVM) + generates the *.png plots
```

## Known limitations

- Trained on a small research dataset, one task (Cookie Theft) only — accuracy will
  not generalize to other elicitation tasks or populations.
- The audio model's accuracy (61.8%) is well below the text model's — acoustic
  features alone are a weaker signal for this task with this amount of data.
- Transcripts must follow CHAT format with participant speech marked `*PAR:`.
- Screening/demo tool only, not a diagnostic instrument.

## Contributors

Rujul Telavane, Krishiv Jadhwani, Liza Khudorozhkova, Nino (Nintsi) Chkhaidze, Stella Chen, Zeba Vora, Saim Meher
