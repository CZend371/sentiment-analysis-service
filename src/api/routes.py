from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List

from src.services.sentiment_service import SentimentService
from src.dataclasses import SentimentInput

router = APIRouter()


# --- Pydantic Request/Response Models ---

class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    text: str
    label: str
    score: float
    scores: Dict[str, float]
    model_name: str


class BatchSentimentRequest(BaseModel):
    texts: List[str]


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]


# --- Endpoints ---

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of a single text. Returns positive, neutral, or negative."""
    try:
        service = SentimentService()
        input_data = SentimentInput(text=request.text)
        result = service.analyze(input_data)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment/batch", response_model=BatchSentimentResponse)
async def analyze_sentiment_batch(request: BatchSentimentRequest):
    """Analyze sentiment of multiple texts."""
    try:
        service = SentimentService()
        inputs = [SentimentInput(text=t) for t in request.texts]
        results = service.analyze_batch(inputs)
        return BatchSentimentResponse(
            results=[SentimentResponse(**r) for r in results]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def model_info():
    """Get information about the loaded sentiment model."""
    try:
        service = SentimentService()
        return service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
