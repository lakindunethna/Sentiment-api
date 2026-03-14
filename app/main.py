from fastapi import FastAPI, HTTPException, Body
from app.model import load_model, predict
from app.schemas import PredictRequest, PredictResponse, HealthResponse
from typing import List

app = FastAPI(title="Sentiment Analysis API")

# Load model at startup
load_model()

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        sentiment, confidence = predict(request.text)
        return PredictResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/batch", response_model=List[PredictResponse])
def predict_batch_endpoint(requests: List[PredictRequest] = Body(...)):
    """
    Accepts JSON array: [{"text": "..."}, {"text": "..."}]
    """
    if not requests:
        raise HTTPException(status_code=400, detail="No texts provided")

    responses = []
    for req in requests:
        sentiment, confidence = predict(req.text)
        responses.append(PredictResponse(
            text=req.text,
            sentiment=sentiment,
            confidence=confidence
        ))
    return responses