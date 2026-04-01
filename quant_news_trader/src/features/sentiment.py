from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

_analyzer = SentimentIntensityAnalyzer()


def compute_sentiment(text: Optional[str]) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    scores = _analyzer.polarity_scores(text)
    return scores["compound"]


def add_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return news_df.assign(sentiment=pd.Series(dtype=float))

    df = news_df.copy()
    combined = (
        df["title"].fillna("")
        + ". "
        + df["description"].fillna("")
        + ". "
        + df["content"].fillna("")
    )
    df["sentiment"] = combined.apply(compute_sentiment)
    return df
