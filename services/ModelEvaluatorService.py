import os
import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ModelEvaluatorService:

    def __init__(self, X_test, y_test, model_name = "best_rf_model.pkl"):
        
        self.X_test = X_test
        self.y_test = y_test

        try:
            self.model = joblib.load(f"snapshots\\{model_name}")
            self.tokenizer = joblib.load(f"snapshots\\tfidf_vectorizer_{os.path.splitext(model_name)[0]}.pkl")
        except FileNotFoundError:
            raise FileNotFoundError("Model file not found. Please train the model first.")

    def evaluate_model(self, verbose=True):

        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        

        if hasattr(self.X_test, "toarray"):  
            X_dense = self.X_test.toarray()
        else:
            X_dense = self.X_test
        
        if X_dense.shape[1] > 1000:  # If too many features
            svd = TruncatedSVD(n_components=100)
            X_reduced = svd.fit_transform(X_dense)
            corr_matrix = pd.DataFrame(X_reduced).corr()
            print("Truncated  Correlation Matrix")
        else:
            corr_matrix = pd.DataFrame(X_dense).corr()
            
        if verbose:
            print("Accuracy:", accuracy_score(self.y_test, y_pred))
            print("Classification Report:\n", classification_report(self.y_test, y_pred))
            print("\nConfusion Matrix:\n", cm)
            print("\nCorrelation Matrix:")
            print(corr_matrix.to_string())
        
        return accuracy, report, cm, corr_matrix
        
