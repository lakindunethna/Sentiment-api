This project wraps a trained Logistic Regression pipeline in a FastAPI service. The pipeline uses TfidfVectorizer for text preprocessing and LogisticRegression for classification. The API exposes a POST /predict endpoint for sentiment prediction and a GET /health endpoint to verify that the service is running.

# Setup Instructions

Python version: Python 3.10+ recommended

1. Clone the repository:

git clone https://github.com/lakindunethna/Sentiment-api
cd sentiment-api

2. Create and activate virtual environment:

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Start the server with

uvicorn app.main:app --reload

5. Example

import requests

url = "http://127.0.0.1:8000/predict"
data = {"text": "I love this product!"}
response = requests.post(url, json=data)
print(response.json())