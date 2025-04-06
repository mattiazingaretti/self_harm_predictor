from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    model_name: str = "best_rf_model.pkl"