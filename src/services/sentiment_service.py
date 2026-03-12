from typing import Dict, Any, List
from src.model.bert_model import BertSentimentModel
from src.dataclasses import SentimentInput


class SentimentService:
    """Service that orchestrates BERT-based sentiment analysis."""

    def __init__(self):
        self.model = BertSentimentModel()

    def analyze(self, input_data: SentimentInput) -> Dict[str, Any]:
        """Analyze sentiment of a single text input."""
        result = self.model.predict(input_data.text)
        return {
            "text": input_data.text,
            "label": result["label"],
            "score": result["score"],
            "scores": result["scores"],
            "model_name": self.model.model_name,
        }

    def analyze_batch(self, inputs: List[SentimentInput]) -> List[Dict[str, Any]]:
        """Analyze sentiment of multiple text inputs."""
        return [self.analyze(inp) for inp in inputs]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model.model_name,
            "labels": ["positive", "neutral", "negative"],
        }
