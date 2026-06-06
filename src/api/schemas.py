from pydantic import BaseModel, Field
from typing import Optional, List


# ── /predict ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to classify")

class PredictResponse(BaseModel):
    label: int          = Field(..., description="0=negative, 1=positive")
    sentiment: str      = Field(..., description="'positive' or 'negative'")
    confidence: float   = Field(..., description="Model probability for predicted class")


# ── /search ───────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str              = Field(..., min_length=1, description="Search query text")
    n: int                  = Field(5, ge=1, le=50, description="Number of results")
    sentiment_filter: Optional[int] = Field(
        None, description="Filter results: 0=negative only, 1=positive only, null=all"
    )

class SearchResult(BaseModel):
    document: str
    index: int
    score: float
    label: Optional[int] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
