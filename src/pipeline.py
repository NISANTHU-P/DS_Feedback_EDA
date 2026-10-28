"""Reusable preprocessing and EDA helpers for customer feedback."""
from pathlib import Path
import re
import pandas as pd

COLUMN_ALIASES = {
    "feedback_id": ["feedback_id", "id", "review_id", "response_id"],
    "date": ["date", "created_at", "timestamp", "feedback_date"],
    "channel": ["channel", "source", "platform", "feedback_channel"],
    "customer_id": ["customer_id", "user_id", "client_id"],
    "rating": ["rating", "score", "stars", "satisfaction_score", "csat"],
    "feedback_text": ["feedback_text", "text", "review", "comment", "message", "feedback"],
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    rename = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = target
                break
    return df.rename(columns=rename)

def clean_feedback(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    if "feedback_text" not in df.columns:
        raise ValueError("CSV must contain a feedback text column (e.g., feedback_text, review, comment, or message).")
    if "feedback_id" not in df.columns:
        df["feedback_id"] = [f"ROW{i+1:05d}" for i in range(len(df))]
    if "channel" not in df.columns:
        df["channel"] = "Unknown"
    if "date" not in df.columns:
        df["date"] = pd.NaT
    if "rating" not in df.columns:
        df["rating"] = pd.NA
    if "customer_id" not in df.columns:
        df["customer_id"] = pd.NA

    df["feedback_text"] = df["feedback_text"].fillna("").astype(str).str.strip()
    df["channel"] = df["channel"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    df["feedback_id"] = df["feedback_id"].fillna("").astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["feedback_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["feedback_text", "channel"], keep="first")
    df = df.drop_duplicates(subset=["feedback_id"], keep="first")
    df["text_length"] = df["feedback_text"].str.len()
    df["word_count"] = df["feedback_text"].str.split().str.len()
    return df.reset_index(drop=True)

def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        df["sentiment_score"] = df["feedback_text"].map(lambda x: analyzer.polarity_scores(x)["compound"])
        df["sentiment"] = df["sentiment_score"].map(
            lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral")
        )
    except ImportError:
        df["sentiment_score"] = pd.NA
        df["sentiment"] = "Not calculated (install vaderSentiment)"
    return df
