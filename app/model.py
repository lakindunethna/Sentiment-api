import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline

model: Pipeline = None
CONFIDENCE_THRESHOLD = 0.6  # below this, return "neutral"

def load_model():
    """Load the trained model from model/ folder."""
    global model
    model_path = Path(__file__).parent.parent / "model" / "model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")

def predict(text: str):
    """Predict sentiment and confidence safely."""
    if model is None:
        raise RuntimeError("Model not loaded")

    # Handle empty or whitespace-only text
    if not isinstance(text, str) or not text.strip():
        return "neutral", 0.0

    try:
        pred = model.predict([text])[0]
        proba = float(model.predict_proba([text]).max())

        # Convert low-confidence predictions to "neutral"
        if proba < CONFIDENCE_THRESHOLD:
            return "neutral", proba

        return pred, proba
    except Exception as e:
        print(f"Prediction error for text '{text}': {e}")
        return "neutral", 0.0