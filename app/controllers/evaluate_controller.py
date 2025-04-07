


from datetime import datetime
from fastapi import APIRouter, Body
from models.EvaluateRequest import EvaluateRequest
from services.DataLoaderService import DataLoaderService
from services.ModelEvaluatorService import ModelEvaluatorService
from services.PreprocessorService import PreprocessorService
from services.TrainingDataProcessorService import TrainingDataProcessorService

import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

evaluateRouter = APIRouter()

@evaluateRouter.post("/evaluate")
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
        
        X_train, X_test, y_train, y_test = training_data_processor_service.split_dataset(request.model_name)

        model_evaluator_service = ModelEvaluatorService(X_test, y_test, request.model_name)
        accuracy, report, cm, correlation_matrix = model_evaluator_service.evaluate_model()

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        cm_buffer = BytesIO()
        plt.savefig(cm_buffer, format='png', bbox_inches='tight')
        cm_base64 = base64.b64encode(cm_buffer.getvalue()).decode('utf-8')
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix.iloc[:50, :50], annot=False, cmap='coolwarm')  # Show subset
        corr_buffer = BytesIO()
        plt.savefig(corr_buffer, format='png', bbox_inches='tight')
        corr_base64 = base64.b64encode(corr_buffer.getvalue()).decode('utf-8')
        plt.close()

        return {
            "message": "Model evaluated successfully",
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm_base64,
            "correlation_matrix": corr_base64
        }

        
        
    except Exception as e:
        return {"error": str(e)}