from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    model_name: str 