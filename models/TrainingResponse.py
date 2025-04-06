from pydantic import BaseModel


class TrainingResponse(BaseModel):
    message: str
    model_path: str
    best_params: dict
