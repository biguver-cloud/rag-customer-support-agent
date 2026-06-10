from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class CitationItem(BaseModel):
    category: str
    source: str
    page: int | None = None
    quote: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    category: str
    best_score: float | None = None
    accuracy: int
    completeness: int
    agent_loops: int
    agent_tokens: int
    citations: list[CitationItem]


class LogFile(BaseModel):
    filename: str
    size: int
    modified: str
