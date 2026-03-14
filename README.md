This project wraps a trained Logistic Regression pipeline in a FastAPI service. The pipeline uses TfidfVectorizer for text preprocessing and LogisticRegression for classification. The API exposes a POST /predict endpoint for sentiment prediction and a GET /health endpoint to verify that the service is running.

# Setup Instructions

### Folder Structure

Python version: Python 3.10+ recommended

### Clone the repository:

git clone https://github.com/lakindunethna/Sentiment-api
cd sentiment-api

### Create and activate virtual environment:

python -m venv venv
venv\Scripts\activate

### Install dependencies

pip install -r requirements.txt

### Training the Model

python train.py

### Start the server with

uvicorn app.main:app --reload

### API Endpoints

Root

GET /

Returns: {"message": "Sentiment Analysis API is running!"}

Health Check

GET /health

Returns: {"status": "ok"}

### Testing Predictions

#### Using swagger 

Open Swagger UI: http://127.0.0.1:8000/docs

##### Single prediction 

Method: POST

Endpoint: /predict

Example request body: {
  "text": "I absolutely loved this movie! The acting was great."
}

##### Batch prediction

Method: POST

Endpoint: /predict/batch

Example request body:{
  "texts": [
    "This was the best movie I have seen all year!",
    "Terrible plot and poor acting, I hated it.",
    "It was okay, nothing special but not bad either."
  ]
}

#### Using powershell

##### Single prediction

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict `
-Method POST `
-ContentType "application/json" `
-Body '{"text":"I absolutely loved this movie! The acting was great."}'

##### Batch prediction

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict/batch `
-Method POST `
-ContentType "application/json" `
-Body '{"texts":["This was the best movie I have seen all year!","Terrible plot and poor acting, I hated it.","It was okay, nothing special but not bad either."]}'





Use these to test all endpoints interactively.

