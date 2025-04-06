from datetime import datetime
import os
from fastapi import APIRouter, HTTPException, Query
import joblib

from services.DataLoaderService import DataLoaderService
from services.ModelEvaluatorService import ModelEvaluatorService
from services.PreprocessorService import PreprocessorService
from services.TrainingDataProcessorService import TrainingDataProcessorService


testRouter = APIRouter()

@testRouter.get("/test")
def test_endpoint(
    text: str = Query(..., description="Text to analyze for potential self-harm content"),
    model_name: str = Query(default="best_rf_model.pkl", description="Name of the model to use")
):
    """
     Enpdoint to test the model reponse to suicidal text passed in input. 
    """
    try:
        model_path = os.path.join("snapshots", model_name)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found. Please train the model first.")

        model = joblib.load(model_path)
        vectorizer_path = os.path.join("snapshots", f"tfidf_vectorizer_{os.path.splitext(model_name)[0]}.pkl")
        
        if not os.path.exists(vectorizer_path):
            raise HTTPException(status_code=404, detail="Vectorizer not found. Please train the model first.")
            
        vectorizer = joblib.load(vectorizer_path)

        text_vectorized = vectorizer.transform([text])
        
        prediction = model.predict(text_vectorized)

        return {
            "message": "Text analyzed successfully",
            "prediction": int(prediction[0]),
            "prediction_label": "Potentially suicidal" if prediction[0] == 1 else "Non-suicidal"
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))