import joblib

model = None


def load_model():
    global model
    model = joblib.load("C:/Users/Lakindu Nethna/Desktop/sentiment-api/model/sentiment_model.pkl")


def predict(text: str):

    pred = model.predict([text])[0]

    proba = model.predict_proba([text]).max()

    return pred, float(proba)