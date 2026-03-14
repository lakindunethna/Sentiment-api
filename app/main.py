# app/main.py
from fastapi import FastAPI
from app.model import load_model, predict
from app.schemas import PredictRequest, PredictResponse, HealthResponse

app = FastAPI(title="Sentiment Analysis API")

# Load the model at startup
load_model()

@app.get("/", response_model=dict)
def root():
    return {"message": "Sentiment Analysis API is running!"}

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """
    Single review prediction.
    Returns the sentiment and confidence score.
    """
    try:
        sentiment, confidence = predict(request.review)
        return PredictResponse(
            review=request.review,
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception:
        return PredictResponse(
            review=request.review,
            sentiment="error",
            confidence=0.0
        )

@app.post("/predict_batch", response_model=list[PredictResponse])
def predict_batch_endpoint(requests: list[PredictRequest]):
    """
    Batch prediction for multiple reviews.
    Returns a list of sentiment predictions and confidence scores.
    """
    responses = []
    for req in requests:
        try:
            sentiment, confidence = predict(req.review)
        except Exception:
            sentiment, confidence = "error", 0.0
        responses.append(PredictResponse(
            review=req.review,
            sentiment=sentiment,
            confidence=confidence
        ))
    return responses