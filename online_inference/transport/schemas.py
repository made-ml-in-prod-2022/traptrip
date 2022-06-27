from typing import List, Union
from pydantic import BaseModel


class HeardDeseaseRequest(BaseModel):
    features: List[str]
    data: List[List[Union[int, float, None]]]


class HeardDeseaseResponse(BaseModel):
    result: List[float]
