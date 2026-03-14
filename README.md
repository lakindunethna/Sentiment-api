This project wraps a trained Logistic Regression pipeline in a FastAPI service. The pipeline uses TfidfVectorizer for text preprocessing and LogisticRegression for classification. The API exposes a POST /predict endpoint for sentiment prediction and a GET /health endpoint to verify that the service is running.

# Setup Instructions

### Folder Structure

Python version: Python 3.10+ recommended

sentiment-api/
│
├── app/
│ ├── main.py # FastAPI app
│ ├── model.py # Load and predict using trained model
│ └── schemas.py # Pydantic request/response models
│
├── model/ # Trained model
│ ├── model.pkl
│ 
│
├── train.py # Train the model and save to model/
├── data/ # Dataset folder
│ └── IMDB_Dataset.csv
├── requirements.txt # Python dependencies
└── README.md # Project documentation

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

Single Prediction

POST /predict

Request body:

### Testing

Open Swagger UI: http://127.0.0.1:8000/docs

Open ReDoc: http://127.0.0.1:8000/redoc

Use these to test all endpoints interactively.