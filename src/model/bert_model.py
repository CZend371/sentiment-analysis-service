from typing import Dict, Any, List
from transformers import pipeline
from src.core.config import settings


class BertSentimentModel:
    """Wrapper around HuggingFace transformers sentiment pipeline.

    Uses distilbert-base-multilingual-cased-sentiments-student for
    3-class sentiment classification: positive, neutral, negative.
    """

    def __init__(self):
        self.model_name = settings.MODEL_NAME
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the transformers sentiment-analysis pipeline."""
        if self._pipeline is None:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,
            )
        return self._pipeline

    def predict(self, text: str) -> Dict[str, Any]:
        """Run sentiment prediction on a single text string.

        Returns dict with top label, its score, and all class scores.
        """
        pipe = self._load_pipeline()
        results = pipe(text, truncation=True, max_length=settings.MAX_INPUT_LENGTH)

        # results is a list of dicts: [{"label": "positive", "score": 0.95}, ...]
        # With top_k=None we get all classes
        scores_list = results[0] if isinstance(results[0], list) else results
        scores = {item["label"]: round(item["score"], 4) for item in scores_list}
        top = max(scores_list, key=lambda x: x["score"])

        return {
            "label": top["label"],
            "score": round(top["score"], 4),
            "scores": scores,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run sentiment prediction on multiple texts."""
        return [self.predict(text) for text in texts]
