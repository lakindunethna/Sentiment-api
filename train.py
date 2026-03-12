# train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("data/IMDB_Dataset.csv")  # make sure this CSV is included in your repo

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
    ("classifier", LogisticRegression())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, pos_label="positive"))
print("Recall:", recall_score(y_test, pred, pos_label="positive"))
print("F1 Score:", f1_score(y_test, pred, pos_label="positive"))

# Save trained model
joblib.dump(model, "app/model.pkl")
print("Model trained and saved as app/model.pkl")

# Test
print(model.predict(["This movie was absolutely amazing"]))