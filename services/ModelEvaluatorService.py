import joblib
from sklearn.metrics import accuracy_score, classification_report


class ModelEvaluatorService:

    def __init__(self, X_test, y_test, model_name = "best_rf_model.pkl"):
        
        self.X_test = X_test
        self.y_test = y_test

        try:
            self.model = joblib.load(f"snapshots\\{model_name}")
        except FileNotFoundError:
            raise FileNotFoundError("Model file not found. Please train the model first.")

    def evaluate_model(self, verbose=True):

        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        if verbose:
            print("Accuracy:", accuracy_score(self.y_test, y_pred))
            print("Classification Report:\n", classification_report(self.y_test, y_pred))

        return accuracy, report