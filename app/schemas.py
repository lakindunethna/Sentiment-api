from pydantic import BaseModel
from typing import List

# Request model for single prediction
class PredictRequest(BaseModel):
    text: str

# Response model for single prediction
class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

# Health check response
class HealthResponse(BaseModel):
    status: str = "ok"

# Batch request model
class BatchPredictRequest(BaseModel):
    texts: List[str]