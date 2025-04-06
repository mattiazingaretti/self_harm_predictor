from fastapi import APIRouter, FastAPI, File, UploadFile
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
async def train_model():
    """
    Endpoint to train the model. This endpoint is used to trigger the training process and return the model path and best parameters.
    """
    try:
        data_loader_service = DataLoaderService()
        df_container = data_loader_service.load_all_datasets()

        preprocessor_service = PreprocessorService(df_container)
        df = preprocessor_service.preprocess_all()

        training_data_processor_service = TrainingDataProcessorService(df)
        X_train, X_test, y_train, y_test = training_data_processor_service.split_dataset()

        training_pipeline_service = TrainingPipelineService()
        model, best_params = training_pipeline_service.train_random_forest(X_train, y_train)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"snapshots/best_rf_model_{timestamp}.pkl"

        return TrainingResponse(
            message="Model trained successfully",
            model_path=model_path,
            best_params=best_params
        )
    except Exception as e:
        return {"error": str(e)}