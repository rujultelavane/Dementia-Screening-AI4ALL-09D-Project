# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib
import pandas as pd
from pathlib import Path
from extraction_files import process_transcript_text, MODEL_FEATURES

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model" / "logistic_regression_pipeline.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".cha"):
        raise HTTPException(status_code=400, detail="Please upload a .cha transcript file.")

    content = (await file.read()).decode("utf-8", errors="ignore")

    try:
        features = process_transcript_text(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    X = pd.DataFrame([features], columns=MODEL_FEATURES)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    classes = model.classes_.tolist()  # e.g. ['Control', 'Dementia']

    confidence = dict(zip(classes, [round(float(p), 4) for p in probabilities]))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "features_used": features,
    }
