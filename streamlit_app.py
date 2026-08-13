import sys
import tempfile
from pathlib import Path

import joblib
import opensmile
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
TRANSCRIPT_DIR = ROOT / "transcript_files"
sys.path.insert(0, str(TRANSCRIPT_DIR))

from extraction_files import process_transcript_text, MODEL_FEATURES  # noqa: E402

st.set_page_config(
    page_title="Dementia Screening Research Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# A small visual system keeps the app feeling like one product instead of a
# collection of default Streamlit widgets. The model and data logic below are
# intentionally unchanged.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #17213b;
        --muted: #68738a;
        --line: #e5e7ef;
        --purple: #6256d9;
        --teal: #159e98;
        --coral: #e87968;
        --surface: #ffffff;
        --soft: #f5f7fc;
    }

    .stApp {
        background: radial-gradient(circle at 8% 4%, rgba(98,86,217,.08), transparent 24rem), radial-gradient(circle at 92% 48%, rgba(21,158,152,.07), transparent 26rem), #f8f8fb;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    .block-container { max-width: 1180px; padding: 1.8rem 2.5rem 4rem; }
    h1, h2, h3, h4 { color: var(--ink); font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.035em; }
    h1 { font-size: clamp(2.1rem, 4vw, 3.55rem) !important; line-height: 1.04 !important; }
    h2 { margin-top: 1.5rem; }
    p, li, label, .stCaption { color: var(--muted); }
    .hero { position: relative; overflow: hidden; background: linear-gradient(120deg, #17213b 0%, #29396e 55%, #514cb4 100%); border-radius: 28px; padding: 2.8rem 3rem; margin: 0 0 1.35rem; box-shadow: 0 22px 50px rgba(31, 43, 86, .2); }
    .hero:before { content: ''; position: absolute; width: 370px; height: 370px; background: radial-gradient(circle, rgba(21,158,152,.28), transparent 64%); right: -80px; bottom: -220px; }
    .hero:after { content: ''; position: absolute; width: 270px; height: 270px; border: 1px solid rgba(255,255,255,.18); border-radius: 50%; right: -55px; top: -95px; box-shadow: 0 0 0 30px rgba(255,255,255,.035), 0 0 0 62px rgba(255,255,255,.025); }
    .hero h1, .hero p { position: relative; z-index: 1; color: white !important; margin: 0; }
    .hero p { max-width: 700px; margin-top: .9rem; color: #dce4ff !important; font-size: 1.05rem; line-height: 1.6; }
    .eyebrow { color: #80e4dc !important; text-transform: uppercase; letter-spacing: .15em; font-size: .72rem !important; font-weight: 700; margin-bottom: .85rem !important; }
    .pill { display: inline-block; margin-top: 1.4rem; padding: .45rem .8rem; border: 1px solid rgba(255,255,255,.2); border-radius: 999px; color: #e9edff; font-size: .78rem; }
    .stTabs [data-baseweb="tab-list"] { gap: .35rem; background: rgba(255,255,255,.82); padding: .4rem; border: 1px solid var(--line); border-radius: 14px; box-shadow: 0 8px 24px rgba(36, 45, 76, .055); }
    .stTabs [data-baseweb="tab"] { height: 2.65rem; border-radius: 999px; color: var(--muted); font-weight: 600; padding: 0 1.1rem; }
    .stTabs [aria-selected="true"] { color: white !important; background: linear-gradient(135deg, #183653, #168f8a); box-shadow: 0 5px 14px rgba(24, 70, 87, .22); }
    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] div { color: white !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stSelectbox, .stFileUploader, [data-testid="stExpander"] { background: rgba(255,255,255,.92); border: 1px solid var(--line); border-radius: 14px; padding: .25rem .75rem; box-shadow: 0 7px 20px rgba(32, 47, 83, .045); }
    [data-testid="stFileUploader"] { padding: .65rem .8rem; }
    [data-testid="stMetric"] { background: rgba(255,255,255,.95); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.2rem; box-shadow: 0 8px 22px rgba(32, 47, 83, .06); }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stAlert"] { border-radius: 14px; border-width: 1px; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, var(--teal), var(--purple)); }
    .stButton > button { border-radius: 999px; border: 1px solid #d9deea; color: var(--ink); font-weight: 600; padding: .45rem 1.15rem; }
    .stButton > button:hover { border-color: var(--teal); color: var(--teal); }
    .section-note { background: linear-gradient(100deg, #eef0ff, #f2fbfa); border-left: 4px solid var(--purple); border-radius: 0 12px 12px 0; padding: .9rem 1rem; margin: .8rem 0 1.3rem; color: #445274; font-size: .92rem; box-shadow: 0 4px 14px rgba(42, 55, 99, .035); }
    .result-heading { color: var(--muted); font-size: .74rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin: 1.35rem 0 .55rem; }
    .status-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: 0 0 1.75rem; }
    .status-card { position: relative; overflow: hidden; background: rgba(255,255,255,.86); border: 1px solid var(--line); border-radius: 15px; padding: .95rem 1rem; box-shadow: 0 7px 20px rgba(32, 47, 83, .045); }
    .status-card:before { content: ''; position: absolute; left: 0; top: 0; width: 100%; height: 3px; background: linear-gradient(90deg, var(--teal), var(--purple)); }
    .status-card strong { display: block; color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: .95rem; }
    .status-card span { color: var(--muted); font-size: .77rem; }
    .status-dot { display: inline-block; width: 7px; height: 7px; background: var(--teal); border-radius: 50%; margin-right: .35rem; box-shadow: 0 0 0 4px rgba(22,166,160,.12); }
    @media (max-width: 700px) { .block-container { padding: 1rem 1rem 3rem; } .hero { padding: 2rem 1.4rem; border-radius: 20px; } .status-grid { grid-template-columns: 1fr; } .stTabs [data-baseweb="tab"] { padding: 0 .55rem; font-size: .78rem; } }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_tab_intro(title: str, description: str):
    st.subheader(title)
    st.markdown(f'<div class="section-note">{description}</div>', unsafe_allow_html=True)

TEXT_EXAMPLES = {
    "Sample recording 1 — Control": TRANSCRIPT_DIR / "Control" / "cookie" / "013-3.cha",
    "Sample recording 2 — PossibleAD": TRANSCRIPT_DIR / "Dementia" / "cookie" / "354-0.cha",
    "Sample recording 3 — PossibleAD": TRANSCRIPT_DIR / "Dementia" / "cookie" / "579-0.cha",
}
AUDIO_EXAMPLES = {
    "Sample recording 1 — Control": ROOT / "revised_audio_files" / "Control" / "013-3.wav",
    "Sample recording 2 — PossibleAD": ROOT / "revised_audio_files" / "Dementia" / "354-0.wav",
    "Sample recording 3 — PossibleAD": ROOT / "revised_audio_files" / "Dementia" / "579-0.wav",
}
FUSION_EXAMPLES = {
    "Sample recording 1 — Control": (
        TRANSCRIPT_DIR / "Control" / "cookie" / "013-3.cha",
        ROOT / "revised_audio_files" / "Control" / "013-3.wav",
    ),
    "Sample recording 2 — PossibleAD": (
        TRANSCRIPT_DIR / "Dementia" / "cookie" / "354-0.cha",
        ROOT / "revised_audio_files" / "Dementia" / "354-0.wav",
    ),
    "Sample recording 3 — PossibleAD": (
        TRANSCRIPT_DIR / "Dementia" / "cookie" / "579-0.cha",
        ROOT / "revised_audio_files" / "Dementia" / "579-0.wav",
    ),
}

TEXT_PREDICTION_EXPLANATIONS = {
    "Control": (
        "This transcript's linguistic features look similar to the **Control** group in "
        "our training data: closer to typical sentence length and grammatical complexity, "
        "fewer word-finding pauses and repetitions, richer vocabulary variety (MTLD), and "
        "most of the picture's key details (boy, girl, cookie jar, overflowing sink, etc.) "
        "mentioned."
    ),
    "Alzheimer's": (
        "This transcript's linguistic features look similar to the **Alzheimer's** group in "
        "our training data: shorter or simpler sentences, more repetitions and pauses, "
        "more vague words (\"thing\", \"stuff\") in place of specific nouns, lower "
        "vocabulary variety (MTLD), and/or fewer of the picture's key details mentioned."
    ),
}

AUDIO_PREDICTION_EXPLANATIONS = {
    "control": (
        "This recording's acoustic features look similar to the **Control** group in our "
        "training data: steadier speaking rate and pauses, and pitch/loudness variation in "
        "the typical range for this task."
    ),
    "alzheimer": (
        "This recording's acoustic features look similar to the **Alzheimer's** group in our "
        "training data: patterns such as more/longer pauses, slower or less steady speaking "
        "rate, and reduced pitch or loudness variation compared to the Control group."
    ),
}


@st.cache_resource
def load_text_model():
    return joblib.load(ROOT / "model" / "text_model.pkl")


@st.cache_resource
def load_audio_model():
    return joblib.load(ROOT / "model" / "audio_model.pkl")


@st.cache_resource
def load_fusion_models():
    return (
        joblib.load(ROOT / "model" / "audio_model.pkl"),
        joblib.load(ROOT / "model" / "text_model.pkl"),
        joblib.load(ROOT / "model" / "meta_model.pkl"),
        joblib.load(ROOT / "model" / "feature_lists.pkl"),
    )


@st.cache_resource
def load_smile():
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def render_text_prediction(raw_text: str):
    model = load_text_model()
    try:
        features = process_transcript_text(raw_text)
    except ValueError as e:
        st.error(f"Couldn't process this transcript: {e}")
        return

    feature_lists = joblib.load(ROOT / "model" / "feature_lists.pkl")
    model_features = feature_lists["text_features"]
    missing = [name for name in model_features if name not in features]
    if missing:
        st.error(f"Couldn't build the text-model input; missing features: {', '.join(missing)}")
        return
    X = pd.DataFrame([[features[name] for name in model_features]], columns=model_features)
    prediction = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    alzheimer_probability = float(probs[1])

    st.markdown('<div class="result-heading">Screening result</div>', unsafe_allow_html=True)
    if prediction == 1:
        st.warning("**Predicted class: Alzheimer's disease**")
    else:
        st.success("**Predicted class: Control**")

    col1, col2 = st.columns(2)
    col1.metric("Control probability", f"{1 - alzheimer_probability:.1%}")
    col2.metric("Alzheimer's probability", f"{alzheimer_probability:.1%}")
    st.progress(alzheimer_probability)

    explanation_key = "Alzheimer's" if prediction == 1 else "Control"
    st.markdown(f"**What this means:** {TEXT_PREDICTION_EXPLANATIONS[explanation_key]}" )

    with st.expander("Extracted linguistic features"):
        st.dataframe(
            pd.Series(features, name="value").rename_axis("feature").reset_index(),
            hide_index=True,
        )


def render_audio_prediction(wav_bytes: bytes):
    model = load_audio_model()
    smile = load_smile()
    feature_lists = joblib.load(ROOT / "model" / "feature_lists.pkl")
    feature_names = feature_lists["audio_features"]

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            features = smile.process_file(tmp.name)
    except Exception as e:
        st.error(f"Couldn't process this audio file: {e}")
        return

    st.audio(wav_bytes, format="audio/wav")

    X = features[feature_names]
    prediction = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    alzheimer_probability = float(probs[1])

    st.markdown('<div class="result-heading">Screening result</div>', unsafe_allow_html=True)
    if prediction == 1:
        st.warning("**Predicted class: Alzheimer's disease**")
    else:
        st.success("**Predicted class: Control**")

    col1, col2 = st.columns(2)
    col1.metric("Control probability", f"{1 - alzheimer_probability:.1%}")
    col2.metric("Alzheimer's probability", f"{alzheimer_probability:.1%}")
    st.progress(alzheimer_probability)

    explanation_key = "alzheimer" if prediction == 1 else "control"
    st.markdown(f"**What this means:** {AUDIO_PREDICTION_EXPLANATIONS[explanation_key]}")

    with st.expander("Extracted acoustic features (eGeMAPS, 88 dims)"):
        st.dataframe(X.T.rename(columns={X.index[0]: "value"}))


def render_fusion_prediction(raw_text: str, wav_bytes: bytes):
    audio_model, text_model, meta_model, feature_lists = load_fusion_models()

    try:
        text_features = process_transcript_text(raw_text)
        text_names = feature_lists["text_features"]
        missing_text = [name for name in text_names if name not in text_features]
        if missing_text:
            raise ValueError(f"Transcript is missing features: {', '.join(missing_text)}")
        X_text = pd.DataFrame([[text_features[name] for name in text_names]], columns=text_names)

        smile = load_smile()
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            audio_features = smile.process_file(tmp.name)
        audio_names = feature_lists["audio_features"]
        X_audio = audio_features[audio_names]
    except Exception as e:
        st.error(f"Couldn't process the transcript and audio together: {e}")
        return

    audio_probability = float(audio_model.predict_proba(X_audio)[0, 1])
    text_probability = float(text_model.predict_proba(X_text)[0, 1])
    meta_input = [[audio_probability, text_probability]]
    fusion_prediction = int(meta_model.predict(meta_input)[0])
    fusion_probability = float(meta_model.predict_proba(meta_input)[0, 1])

    st.audio(wav_bytes, format="audio/wav")
    st.markdown('<div class="result-heading">Combined screening result</div>', unsafe_allow_html=True)
    if fusion_prediction == 1:
        st.warning("**Predicted class: Alzheimer's disease**")
    else:
        st.success("**Predicted class: Control**")

    col1, col2 = st.columns(2)
    col1.metric("Control probability", f"{1 - fusion_probability:.1%}")
    col2.metric("Alzheimer's probability", f"{fusion_probability:.1%}")
    st.progress(fusion_probability)

    st.markdown('<div class="result-heading">Signal breakdown</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Audio model: Alzheimer's", f"{audio_probability:.1%}")
    col2.metric("Text model: Alzheimer's", f"{text_probability:.1%}")
    st.caption("The fusion model combines these two probabilities; it does not simply average them.")


def predict_text_tab():
    render_tab_intro(
        "Predict from a picture-description transcript",
        "Research / educational demo — not a medical diagnosis. This screen looks for linguistic patterns associated with Alzheimer’s disease in the training data.",
    )

    example_choice = st.selectbox(
        "Try a real example, or upload your own .cha file below",
        ["— none —"] + list(TEXT_EXAMPLES.keys()),
        key="text_example",
    )

    uploaded = st.file_uploader("Upload a CHAT-format transcript (.cha)", type=["cha"])

    raw_text = None
    if uploaded is not None:
        raw_text = uploaded.read().decode("utf-8", errors="ignore")
    elif example_choice != "— none —":
        raw_text = TEXT_EXAMPLES[example_choice].read_text(encoding="utf-8", errors="ignore")

    if raw_text is None:
        st.info("Choose an example or upload a .cha file to get a prediction.")
        return

    with st.expander("Raw transcript"):
        st.text(raw_text)

    render_text_prediction(raw_text)


def predict_audio_tab():
    render_tab_intro(
        "Predict from a picture-description recording",
        "Research / educational demo — not a medical diagnosis. This screen evaluates acoustic patterns such as pitch, pauses, voice quality, and timing. Audio results are independent of the text screen.",
    )

    example_choice = st.selectbox(
        "Try a real example, or upload your own .wav file below",
        ["— none —"] + list(AUDIO_EXAMPLES.keys()),
        key="audio_example",
    )

    uploaded = st.file_uploader("Upload a recording (.wav)", type=["wav"])

    wav_bytes = None
    if uploaded is not None:
        wav_bytes = uploaded.read()
    elif example_choice != "— none —":
        wav_bytes = AUDIO_EXAMPLES[example_choice].read_bytes()

    if wav_bytes is None:
        st.info("Choose an example or upload a .wav file to get a prediction.")
        return

    render_audio_prediction(wav_bytes)


def predict_fusion_tab():
    render_tab_intro(
        "Predict from audio + transcript",
        "Choose a prepared recording to load its matching audio and transcript automatically. The fusion model combines both signals into one Alzheimer’s-vs-Control screening result.",
    )

    sample_choice = st.selectbox(
        "Choose a prepared matching recording",
        ["— choose a sample —"] + list(FUSION_EXAMPLES),
        key="fusion_sample",
    )

    raw_text = None
    wav_bytes = None
    if sample_choice != "— choose a sample —":
        transcript_path, audio_path = FUSION_EXAMPLES[sample_choice]
        raw_text = transcript_path.read_text(encoding="utf-8", errors="ignore")
        wav_bytes = audio_path.read_bytes()
        st.caption("The matching transcript and recording were loaded automatically.")

    if raw_text is None or wav_bytes is None:
        st.info("Choose a sample recording to run fusion.")
        return

    render_fusion_prediction(raw_text, wav_bytes)


def about_tab():
    render_tab_intro(
        "About this app",
        "A research interface for exploring how language and voice signals behave on the DementiaBank Pitt picture-description task.",
    )
    st.markdown(
        """

        This app screens a picture-description task for **Control vs.
Alzheimer's disease** patterns. The current research pipeline uses text, audio,
and a fusion model:

- **Text model** — linguistic features (word choice, sentence structure,
  repetitions, pauses, and coherence) extracted from a `.cha` transcript.
- **Audio model** — 88 eGeMAPS acoustic features (pitch, loudness, voice quality,
  timing) extracted from a `.wav` recording with
  [openSMILE](https://audeering.github.io/opensmile-python/), fed into an
  acoustic model. A fusion model combines audio and text probabilities using a
  meta-model.

The current research target combines `ProbableAD` and `PossibleAD` into an
Alzheimer's disease class and compares it with `Control`. MCI, Vascular, Memory,
Other, and blank-diagnosis recordings are excluded from fusion training.

**Known limitations**
- Trained on a small research dataset (DementiaBank Pitt corpus, picture-description
  task only) — accuracy will not generalize perfectly to other tasks or populations.
- This is a screening/demo tool, not a diagnostic instrument.
- Transcripts must follow CHAT format with participant speech marked `*PAR:`.

**Run locally**
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
        """
    )

    st.subheader("Updated model performance")
    st.markdown(
        """
| Model | Validation | Accuracy |
|---|---|---|
| Text only | 5-fold speaker-independent CV, 352 aligned recordings | **~76.7%** |
| Audio only | 5-fold speaker-independent CV, 352 aligned recordings | **~62.2%** |
| Audio + text fusion | Leakage-safe 5-fold speaker-independent CV | **~76.7%** |

The task is **Alzheimer’s disease vs Control**: `ProbableAD` and `PossibleAD`
are combined into the Alzheimer’s class. The main finding is that language
carries most of the predictive signal; fusion currently ties the text model
rather than improving on it. These are cross-validation estimates on unseen
speakers, not accuracy measured on the final model retrained on all available data.
        """
    )

    st.subheader("Fusion model evaluation")
    fusion_matrix = ROOT / "fusion_confusion_matrix.png"
    if fusion_matrix.exists():
        _, image_col, _ = st.columns([1, 2, 1])
        with image_col:
            st.image(str(fusion_matrix), caption="Fusion model confusion matrix", width=680)

    st.subheader("Exploratory acoustic model comparison plots")
    st.caption(
        "From `modelevaluation` / `acoustic_features.csv` — an earlier exploration "
        "comparing Logistic Regression, Random Forest, and SVM on the acoustic "
        "features before the SVM pipeline above was selected and saved."
    )
    for img, caption in [
        ("model_comparison.png", "Acoustic model comparison: accuracy / precision / recall / F1 / ROC-AUC"),
        ("roc_curves.png", "Acoustic model comparison: ROC curves"),
        ("confusion_matrices.png", "Acoustic model comparison: confusion matrices"),
        ("rf_feature_importance.png", "Acoustic model: Random Forest feature importance"),
    ]:
        path = ROOT / img
        if path.exists():
            _, image_col, _ = st.columns([1, 2, 1])
            with image_col:
                st.image(str(path), caption=caption, width=680)


st.markdown(
    """
    <section class="hero">
        <p class="eyebrow">Research interface · DementiaBank Pitt task</p>
        <h1>🧠 Dementia Screening<br>Research Demo</h1>
        <p>Explore text, audio, and multimodal predictions from a picture-description task. Upload your own sample or start with one of the prepared examples.</p>
        <span class="pill">⚕ Not a diagnostic instrument</span>
    </section>
    <div class="status-grid">
        <div class="status-card"><strong><span class="status-dot"></span>Text analysis</strong><span>Linguistic features · CHAT transcript</span></div>
        <div class="status-card"><strong><span class="status-dot"></span>Voice analysis</strong><span>Acoustic features · eGeMAPS</span></div>
        <div class="status-card"><strong><span class="status-dot"></span>Fusion model</strong><span>Combined multimodal screening</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_text, tab_audio, tab_fusion, tab_about = st.tabs(
    ["✦  Text Model", "◉  Audio Model", "◎  Fusion Model", "About / Model Performance"]
)
with tab_text:
    predict_text_tab()
with tab_audio:
    predict_audio_tab()
with tab_fusion:
    predict_fusion_tab()
with tab_about:
    about_tab()
