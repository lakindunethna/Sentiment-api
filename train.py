import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("data/IMDB_Dataset.csv") 

# Features and labels
X = df["review"]
y = df["sentiment"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
    ("classifier", LogisticRegression(max_iter=500))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, pos_label="positive"))
print("Recall:", recall_score(y_test, pred, pos_label="positive"))
print("F1 Score:", f1_score(y_test, pred, pos_label="positive"))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Save trained model
model_path = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(model, model_path)
print(f"Model trained and saved as {model_path}")

# Quick test
print(model.predict(["This movie was absolutely amazing"]))