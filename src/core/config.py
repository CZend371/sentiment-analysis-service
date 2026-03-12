from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    PROJECT_NAME: str = "Sentiment Analysis Service"
    DEBUG: bool = True
    MODEL_NAME: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    MODEL_CACHE_DIR: Optional[str] = "models/bert_cache"
    MAX_INPUT_LENGTH: int = 512


settings = Settings()
