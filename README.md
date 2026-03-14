This project wraps a trained Logistic Regression pipeline in a FastAPI service. The pipeline uses TfidfVectorizer for text preprocessing and LogisticRegression for classification. The API exposes a POST /predict endpoint for sentiment prediction and a GET /health endpoint to verify that the service is running.

# Setup Instructions

### Folder Structure

Python version: Python 3.10+ recommended

### 1. Clone the repository :

git clone https://github.com/lakindunethna/Sentiment-api
cd sentiment-api

### 2. Create and activate virtual environment : isolates dependencies so nothing conflicts with other Python projects

python -m venv venv
venv\Scripts\activate

### 3. Install dependencies : ensures the environment has all packages required to run and train the API

pip install -r requirements.txt

### 4. Training the Model

python train.py

### 5. Start the server with : Starts the server so endpoints /predict and /predict/batch are available. --reload automatically updates the server when code changes.

uvicorn app.main:app --reload

### 6. API Endpoints

#### Health Check

GET /health

Returns:{"status": "ok"}

### Testing Predictions

#### Using swagger 

Open Swagger UI: http://127.0.0.1:8000/docs

##### Single prediction 

Method: POST

Endpoint: /predict

Example request body: 
{
  "text": "I absolutely loved this movie! The acting was great."
}

##### Batch prediction

Method: POST

Endpoint: /predict/batch

Example request body:
[
  {"text":"This was the best movie I have seen all year!"},
  {"text":"Terrible plot and poor acting, I hated it."},
  {"text":"It was okay, nothing special but not bad either."}
]

#### Using powershell

##### Single prediction

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method POST -ContentType "application/json" -Body '{"text":"This movie was amazing!"}'

##### Batch prediction

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict/batch -Method POST -ContentType "application/json" -Body '[{"text":"This was the best movie I have seen all year!"},{"text":"Terrible plot and poor acting, I hated it."},{"text":"It was okay, nothing special but not bad either."}]'




Use these to test all endpoints interactively.

