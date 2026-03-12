from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float