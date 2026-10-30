from pathlib import Path

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.datasets import load_breast_cancer


# Location of the trained machine learning model
MODEL_PATH = Path("artifacts/best_model.joblib")


# Create FastAPI application
app = FastAPI(
    title="Student MLOps Classification API",
    description="Educational demonstration only; not for medical decisions."
)


# Define input format
class PredictionRequest(BaseModel):

    features: list[float] = Field(
        min_length=30,
        max_length=30
    )


# Health check API
@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists()
    }


# Prediction API
@app.post("/predict")
def predict(request: PredictionRequest):

    # Check whether trained model exists
    if not MODEL_PATH.exists():

        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run python -m src.train"
        )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Check expected number of features
    expected = len(load_breast_cancer().feature_names)

    if len(request.features) != expected:

        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} features"
        )

    # Generate prediction
    prediction = int(
        model.predict(
            np.array(request.features).reshape(1, -1)
        )[0]
    )

    # Convert prediction into label
    label = "malignant" if prediction == 0 else "benign"

    return {
        "prediction_code": prediction,
        "predicted_label": label,
        "notice": "Educational dataset output only; not a medical diagnosis."
    }