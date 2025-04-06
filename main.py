
from services.DataLoaderService import DataLoaderService
from services.TrainingPipelineService import TrainingPipelineService
from services.ModelEvaluatorService import ModelEvaluatorService
    

if __name__ == "__main__":
    
    model, X_test, y_test = TrainingPipelineService(DataLoaderService()).train_random_forest()
    print("Model training completed and saved to 'snapshots/best_rf_model.pkl'.")
    print("Model evaluation:")
    ModelEvaluatorService(X_test, y_test).evaluate_model()