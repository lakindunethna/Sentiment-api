# app/model.py
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline

model: Pipeline = None

def load_model():
    """Load the trained model from the model/ folder."""
    global model
    model_path = Path(__file__).parent.parent / "model" / "sentiment_model.pkl"
    model = joblib.load(model_path)

def predict(text: str):
    """Predict sentiment and confidence for a given text."""
    pred = model.predict([text])[0]
    proba = model.predict_proba([text]).max()
    return pred, float(proba)