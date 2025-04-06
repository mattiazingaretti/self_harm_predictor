


from fastapi import APIRouter, Body
from models.EvaluateRequest import EvaluateRequest
from services.DataLoaderService import DataLoaderService
from services.ModelEvaluatorService import ModelEvaluatorService
from services.PreprocessorService import PreprocessorService
from services.TrainingDataProcessorService import TrainingDataProcessorService


router = APIRouter()

@router.post("/evaluate")
def evaluate_model(request: EvaluateRequest = Body(...)):
    """
    Endpoint to evaluate the model passed in input. This endpoint is used to trigger the evaluation process and return the evaluation report.
    """
    try:
        data_loader_service = DataLoaderService()
        df_container = data_loader_service.load_all_datasets()

        preprocessor_service = PreprocessorService(df_container)
        df = preprocessor_service.preprocess_all()

        training_data_processor_service = TrainingDataProcessorService(df)
        X_train, X_test, y_train, y_test = training_data_processor_service.split_dataset()

        model_evaluator_service = ModelEvaluatorService(X_test, y_test, request.model_name)
        accuracy, report = model_evaluator_service.evaluate_model()

        return {
            "message": "Model evaluated successfully",
            "accuracy": accuracy,
            "report": report
        }
    except Exception as e:
        return {"error": str(e)}