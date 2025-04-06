import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from services.TrainingDataProcessorService import TrainingDataProcessorService
from services.DataLoaderService import DataLoaderService
from services.PreprocessorService import PreprocessorService


class TrainingPipelineService:

    def __init__(self, data_loader_service: DataLoaderService = DataLoaderService() ):
        self.df = data_loader_service.load_all_datasets()


    
    def train_random_forest(self, verbose = 3):
        preprocessor_service = PreprocessorService(self.df)
        self.df = preprocessor_service.preprocess_all()
        
        training_data_processor_service = TrainingDataProcessorService(self.df)
        X_train, X_test, y_train, y_test = training_data_processor_service.split_dataset()


        model = RandomForestClassifier( class_weight='balanced',  random_state=42)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 20, None],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(
               estimator=model,
               param_grid=param_grid,
               cv=5,
               scoring='f1',
               n_jobs=-1,
               verbose = verbose
        )
        grid_search.fit(X_train, y_train)
        
        os.makedirs('snapshots', exist_ok=True)
        joblib.dump(grid_search.best_estimator_, 'snapshots/best_rf_model.pkl')

        return model, X_test, y_test
        
