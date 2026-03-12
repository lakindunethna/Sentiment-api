from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import os

app = FastAPI()

# Load the trained pipeline
model_path = "model\sentiment_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Pydantic models
class TextInput(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class BatchInput(BaseModel):
    texts: List[str]

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Predict single text
@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(input_data: TextInput):
    if model is None:
        return {"text": input_data.text, "sentiment": "error", "confidence": 0.0}
    
    try:
        text = input_data.text
        sentiment = model.predict([text])[0]
        
        # Get confidence (probability of predicted class)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            class_index = list(model.classes_).index(sentiment)
            confidence = float(probs[class_index])
        else:
            confidence = 1.0  # fallback if model does not support probabilities

        return {"text": text, "sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        return {"text": input_data.text, "sentiment": "error", "confidence": 0.0}

# Batch predictions
@app.post("/predict_batch", response_model=List[PredictResponse])
def predict_batch(input_data: BatchInput):
    if model is None:
        return [{"text": t, "sentiment": "error", "confidence": 0.0} for t in input_data.texts]
    
    try:
        predictions = model.predict(input_data.texts)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data.texts)
        else:
            probs = [[1.0] for _ in predictions]

        responses = []
        for i, text in enumerate(input_data.texts):
            sentiment = predictions[i]
            class_index = list(model.classes_).index(sentiment)
            confidence = float(probs[i][class_index])
            responses.append({"text": text, "sentiment": sentiment, "confidence": confidence})
        return responses
    except Exception as e:
        return [{"text": t, "sentiment": "error", "confidence": 0.0} for t in input_data.texts]