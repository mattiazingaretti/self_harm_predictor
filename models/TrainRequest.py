from pydantic import BaseModel

class TrainRequest(BaseModel):
    perform_cv: bool = True