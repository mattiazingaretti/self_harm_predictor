from fastapi import APIRouter, FastAPI, File, UploadFile
from models.TrainRequest import TrainRequest
from models.TrainingResponse import TrainingResponse
from services.DataLoaderService import DataLoaderService
from services.PreprocessorService import PreprocessorService
from services.TrainingDataProcessorService import TrainingDataProcessorService
from services.TrainingPipelineService import TrainingPipelineService
from datetime import datetime


"""
TODO [NTH] - It would be nice to have a training endpoint that accepts a specific dataset and train the model on it, instead of using the default datasets.
Something like this:
    @app.post("/train", response_model=TrainingResponse)
    async def train_model(file: UploadFile = File(...)):
        ...
"""

trainRouter = APIRouter()

@trainRouter.post("/train")
async def train_model(request: TrainRequest):
    """
    Endpoint to train the model with optional cross-validation.
    """
    try:
        model, model_path = TrainingPipelineService().train_random_forest(
            perform_CV=request.perform_cv, 
            verbose=3
        )

        return TrainingResponse(
            message="Model trained successfully",
            model_path=model_path
        )
    except Exception as e:
        return {"error": str(e)}