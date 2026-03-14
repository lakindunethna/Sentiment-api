# app/schemas.py
from pydantic import BaseModel

class PredictRequest(BaseModel):
    review: str

class PredictResponse(BaseModel):
    review: str
    sentiment: str
    confidence: float

class HealthResponse(BaseModel):
    status: str