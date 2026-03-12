from dataclasses import dataclass


@dataclass
class SentimentInput:
    """Domain object representing text to analyze for sentiment"""
    text: str
