# app/main.py
from fastapi import FastAPI
from app.model import load_model, predict
from app.schemas import PredictRequest, PredictResponse, HealthResponse

app = FastAPI(title="Sentiment Analysis API")

# Load the model at startup
load_model()

# Health check
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()

# Single prediction
@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        sentiment, confidence = predict(request.text)
        return PredictResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception:
        return PredictResponse(
            text=request.text,
            sentiment="error",
            confidence=0.0
        )

# Optional: batch predictions
@app.post("/predict_batch", response_model=list[PredictResponse])
def predict_batch_endpoint(requests: list[PredictRequest]):
    responses = []
    for req in requests:
        try:
            sentiment, confidence = predict(req.text)
        except Exception:
            sentiment, confidence = "error", 0.0
        responses.append(PredictResponse(
            text=req.text,
            sentiment=sentiment,
            confidence=confidence
        ))
    return responses