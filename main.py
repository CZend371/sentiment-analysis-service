from fastapi import FastAPI
from src.api.routes import router as api_router
from src.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="BERT-based Sentiment Analysis API (positive/neutral/negative)",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
