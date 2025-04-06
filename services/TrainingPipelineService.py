from datetime import datetime
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from services.TrainingDataProcessorService import TrainingDataProcessorService
from services.DataLoaderService import DataLoaderService
from services.PreprocessorService import PreprocessorService
from contextlib import redirect_stdout

class TrainingPipelineService:

    def __init__(self, data_loader_service: DataLoaderService = DataLoaderService() ):
        self.df = data_loader_service.load_all_datasets()


    
    def train_random_forest(self, perform_CV=True, verbose = 3):
        preprocessor_service = PreprocessorService(self.df)
        self.df = preprocessor_service.preprocess_all()
        
        training_data_processor_service = TrainingDataProcessorService(self.df)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"snapshots/best_rf_model_{timestamp}.pkl"
        X_train, X_test, y_train, y_test = training_data_processor_service.split_dataset(f"best_rf_model_{timestamp}.pkl")


        model = RandomForestClassifier(
             class_weight='balanced', 
             random_state=42,
             n_estimators=100,
             max_depth=20,
             min_samples_split=5,
             verbose=verbose
        )

        if perform_CV:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [15, 20, None],
                'min_samples_split': [2, 5]
            }

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=2,
                scoring='f1',
                n_jobs=-1,
                verbose = verbose
            )
            with open('logs/training_rf_cv.log', 'w') as log_file:
                with redirect_stdout(log_file):
                    grid_search.fit(X_train, y_train)

            print("Best parameters found: ", grid_search.best_params_)        
            joblib.dump(grid_search.best_estimator_, f"CV_{model_path}")
        else:
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
            return model, None    
        
        os.makedirs('snapshots', exist_ok=True)
        
        
        
        return model, model_path        
