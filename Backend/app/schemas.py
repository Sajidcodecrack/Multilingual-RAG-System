from pydantic import BaseModel
from typing import List, Tuple

class Query(BaseModel):
    question: str

class ChatQuery(BaseModel):
    question: str
    history: List[Tuple[str, str]] = []

class Answer(BaseModel):
    response: str