"""Reusable preprocessing and EDA helpers for customer feedback.

Supports both the original synthetic schema and the Yelp dataset schema.
All original source columns are preserved; nothing is silently discarded.
"""
from pathlib import Path
import re
import pandas as pd

# ---------------------------------------------------------------------------
# Column-name normalisation aliases
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "feedback_id":    ["feedback_id", "sl_no", "id", "review_id", "response_id"],
    "date":           ["date", "created_at", "timestamp", "feedback_date"],
    "channel":        ["channel", "source", "platform", "feedback_channel"],
    "customer_id":    ["customer_id", "user_id", "client_id"],
    "rating":         ["rating", "stars_review", "score", "stars",
                       "satisfaction_score", "csat"],
    "feedback_text":  ["feedback_text", "text", "review", "comment",
                       "message", "feedback"],
    # Yelp-specific extras kept under stable names
    "business_name":  ["business_name", "name"],
    "categories":     ["categories"],
    "city":           ["city"],
    "state":          ["state"],
    "business_rating":["business_rating", "stars_business"],
    "review_count":   ["review_count"],
    "sentiment_source": ["sentiment_source", "rating_review"],
}

# Extended English stop-word list for word-frequency analysis
STOPWORDS = set(
    "the and for was with this that have had are but not you your our "
    "they their from very after when were still has its its been just "
    "will would could should about also more than into out all one two "
    "get got did does didn even though here there then than so if at "
    "an in on to of is it we he she be by do as up no my me him her "
    "us who what how why which where i a s t re ve ll m d".split()
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and de-special-char all column names; apply aliases."""
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
        for c in df.columns
    ]
    rename = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and alias != target:
                rename[alias] = target
                break
    return df.rename(columns=rename)


def clean_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalise a feedback DataFrame.

    Returns the cleaned DataFrame and prints a removal summary.
    All original columns are preserved; new derived columns are appended.
    """
    original_count = len(df)
    df = normalize_columns(df)

    # ── Mandatory column check ────────────────────────────────────────────
    if "feedback_text" not in df.columns:
        raise ValueError(
            "CSV must contain a feedback text column "
            "(e.g. feedback_text, text, review, comment, or message)."
        )

    # ── Inject missing optional columns ──────────────────────────────────
    if "feedback_id" not in df.columns:
        df["feedback_id"] = [f"ROW{i+1:05d}" for i in range(len(df))]
    if "channel" not in df.columns:
        df["channel"] = "Yelp"          # default for Yelp data
    if "date" not in df.columns:
        df["date"] = pd.NaT
    if "rating" not in df.columns:
        df["rating"] = pd.NA
    if "customer_id" not in df.columns:
        df["customer_id"] = pd.NA

    # ── Type coercions ────────────────────────────────────────────────────
    df["feedback_text"] = df["feedback_text"].fillna("").astype(str).str.strip()
    df["channel"]       = (df["channel"].fillna("Unknown").astype(str)
                           .str.strip().replace("", "Unknown"))
    df["feedback_id"]   = df["feedback_id"].fillna("").astype(str).str.strip()
    df["date"]          = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["rating"]        = pd.to_numeric(df["rating"], errors="coerce")

    # Clamp ratings to 1–5 (flag out-of-range as NaN)
    if df["rating"].notna().any():
        invalid_rating = df["rating"].notna() & ~df["rating"].between(1, 5)
        if invalid_rating.sum():
            print(f"  [clean] {invalid_rating.sum()} records had ratings outside 1–5 → set to NaN")
            df.loc[invalid_rating, "rating"] = pd.NA

    # Preserve original sentiment label if present
    if "sentiment_source" in df.columns:
        df["sentiment_source"] = df["sentiment_source"].fillna("Unknown").astype(str).str.strip()

    # Preserve numeric business metadata
    for col in ("business_rating", "review_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Preserve string metadata
    for col in ("business_name", "categories", "city", "state"):
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # ── Remove empty-text records ─────────────────────────────────────────
    empty_mask = df["feedback_text"].str.len() == 0
    empty_removed = empty_mask.sum()
    df = df[~empty_mask]

    # ── Remove exact duplicates (same text + channel) ─────────────────────
    before_dup = len(df)
    df = df.drop_duplicates(subset=["feedback_text", "channel"], keep="first")
    dup_text_removed = before_dup - len(df)

    # ── Remove duplicate feedback_ids ────────────────────────────────────
    before_id = len(df)
    df = df.drop_duplicates(subset=["feedback_id"], keep="first")
    dup_id_removed = before_id - len(df)

    # ── Derived columns ───────────────────────────────────────────────────
    df["text_length"] = df["feedback_text"].str.len()
    df["word_count"]  = df["feedback_text"].str.split().str.len()

    df = df.reset_index(drop=True)

    # ── Removal summary ───────────────────────────────────────────────────
    total_removed = original_count - len(df)
    print(f"  [clean] Raw records          : {original_count:,}")
    print(f"  [clean] Empty text removed   : {empty_removed:,}")
    print(f"  [clean] Dup text+ch removed  : {dup_text_removed:,}")
    print(f"  [clean] Dup feedback_id rem  : {dup_id_removed:,}")
    print(f"  [clean] Cleaned records      : {len(df):,}  "
          f"(removed {total_removed:,} total)")

    return df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append VADER sentiment_score and sentiment columns.

    The original sentiment_source column (from rating_review) is NOT
    overwritten. VADER results are stored separately as 'sentiment' and
    'sentiment_score'.
    """
    df = df.copy()
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        df["sentiment_score"] = df["feedback_text"].map(
            lambda x: analyzer.polarity_scores(str(x))["compound"]
        )
        df["sentiment"] = df["sentiment_score"].map(
            lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral")
        )
    except ImportError:
        df["sentiment_score"] = pd.NA
        df["sentiment"] = "Not calculated (install vaderSentiment)"
    return df


def get_top_words(series: pd.Series, n: int = 20) -> pd.DataFrame:
    """Return a DataFrame of the top-n most frequent meaningful words."""
    from collections import Counter
    tokens = []
    for txt in series.astype(str):
        tokens.extend(
            w for w in re.findall(r"[a-zA-Z]{3,}", txt.lower())
            if w not in STOPWORDS
        )
    return pd.DataFrame(Counter(tokens).most_common(n), columns=["word", "count"])


def get_top_bigrams(series: pd.Series, n: int = 15) -> pd.DataFrame:
    """Return a DataFrame of top-n most frequent 2-word phrases (bigrams)."""
    from collections import Counter
    bigrams = []
    for txt in series.astype(str):
        words = [
            w for w in re.findall(r"[a-zA-Z]{3,}", txt.lower())
            if w not in STOPWORDS
        ]
        bigrams.extend(zip(words[:-1], words[1:]))
    counts = Counter(f"{w1} {w2}" for w1, w2 in bigrams)
    return pd.DataFrame(counts.most_common(n), columns=["bigram", "count"])


def sentiment_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare original sentiment_source labels against VADER sentiment.
    Returns a crosstab (original rows × VADER columns).
    Note: original labels are reference labels and may themselves contain errors.
    """
    if "sentiment_source" not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(
        df["sentiment_source"], df["sentiment"],
        rownames=["Original label"], colnames=["VADER label"]
    )
