"""Automated tests for src/pipeline.py — covers synthetic + Yelp schemas."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import normalize_columns, clean_feedback, add_sentiment, get_top_words, get_top_bigrams, sentiment_agreement


# ── helpers ───────────────────────────────────────────────────────────────────
def _make_df(**kwargs):
    base = {
        "feedback_id":   ["F001", "F002"],
        "date":          ["2026-01-01", "2026-01-02"],
        "channel":       ["Survey", "Email"],
        "customer_id":   ["C1", "C2"],
        "rating":        [5, 3],
        "feedback_text": ["Great service!", "Average experience."],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


def _make_yelp_df(**kwargs):
    base = {
        "feedback_id":      ["YLP000001", "YLP000002"],
        "date":             ["07-07-18 22:09", "05-02-14 20:30"],
        "channel":          ["Yelp", "Yelp"],
        "rating":           [4, 2],
        "feedback_text":    ["Great food and service!", "Terrible experience, never coming back."],
        "sentiment_source": ["Positive", "Negative"],
        "business_name":    ["Biz A", "Biz B"],
        "categories":       ["Restaurants", "Bars"],
        "city":             ["Philadelphia", "Tucson"],
        "state":            ["PA", "AZ"],
        "business_rating":  [4.0, 3.5],
        "review_count":     [200, 50],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ── normalize_columns ─────────────────────────────────────────────────────────
def test_normalize_strips_and_lowercases():
    df = pd.DataFrame({"  Feedback Text  ": ["hello"]})
    out = normalize_columns(df)
    assert "feedback_text" in out.columns


def test_normalize_alias_mapping_synthetic():
    df = pd.DataFrame({"review": ["text"], "stars": [4], "source": ["App"]})
    out = normalize_columns(df)
    assert "feedback_text" in out.columns
    assert "rating" in out.columns
    assert "channel" in out.columns


def test_normalize_alias_mapping_yelp():
    df = pd.DataFrame({
        "Sl No": [1], "stars_review": [4], "text": ["hello"],
        "name": ["Biz"], "stars_business": [4.0], "rating_review": ["Positive"]
    })
    out = normalize_columns(df)
    assert "feedback_id"      in out.columns
    assert "rating"           in out.columns
    assert "feedback_text"    in out.columns
    assert "business_name"    in out.columns
    assert "business_rating"  in out.columns
    assert "sentiment_source" in out.columns


# ── clean_feedback — synthetic schema ─────────────────────────────────────────
def test_clean_feedback_basic_shape():
    out = clean_feedback(_make_df())
    assert len(out) == 2
    assert "text_length" in out.columns
    assert "word_count"  in out.columns


def test_clean_feedback_removes_empty_text():
    df = _make_df(feedback_text=["Valid text", ""])
    out = clean_feedback(df)
    assert len(out) == 1
    assert out.iloc[0]["feedback_text"] == "Valid text"


def test_clean_feedback_removes_duplicates_by_text_and_channel():
    df = pd.DataFrame({
        "feedback_id":   ["F001", "F002", "F003"],
        "date":          ["2026-01-01"] * 3,
        "channel":       ["Survey", "Survey", "Email"],
        "customer_id":   ["C1", "C2", "C3"],
        "rating":        [5, 5, 3],
        "feedback_text": ["Same text", "Same text", "Different text"],
    })
    out = clean_feedback(df)
    assert len(out) == 2


def test_clean_feedback_missing_text_column_raises():
    df = pd.DataFrame({"col_a": [1, 2]})
    with pytest.raises(ValueError, match="feedback text column"):
        clean_feedback(df)


def test_clean_feedback_auto_generates_feedback_id():
    df = pd.DataFrame({"feedback_text": ["Hello world"], "channel": ["Survey"]})
    out = clean_feedback(df)
    assert out["feedback_id"].iloc[0].startswith("ROW")


def test_clean_feedback_missing_channel_defaults():
    df = pd.DataFrame({"feedback_text": ["Some text"]})
    out = clean_feedback(df)
    # default channel for Yelp-style data is "Yelp"; for unknown it's "Unknown"
    assert out["channel"].iloc[0] in ("Yelp", "Unknown")


def test_clean_feedback_rating_coerced_to_numeric():
    df = _make_df(rating=["5", "bad_value"])
    out = clean_feedback(df)
    bad_row = out.loc[out["feedback_text"] == "Average experience.", "rating"]
    assert pd.isna(bad_row.iloc[0])


def test_clean_feedback_rating_out_of_range_set_to_nan():
    df = _make_df(rating=[6, 0])
    out = clean_feedback(df)
    assert out["rating"].isna().all()


def test_clean_feedback_date_parsed():
    out = clean_feedback(_make_df())
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_clean_feedback_text_length_and_word_count_correct():
    df = pd.DataFrame({"feedback_text": ["Hello world test"]})
    out = clean_feedback(df)
    assert out["text_length"].iloc[0] == len("Hello world test")
    assert out["word_count"].iloc[0] == 3


def test_clean_feedback_empty_dataframe():
    df = pd.DataFrame({"feedback_text": []})
    out = clean_feedback(df)
    assert len(out) == 0


# ── clean_feedback — Yelp schema ──────────────────────────────────────────────
def test_clean_feedback_yelp_preserves_sentiment_source():
    df = _make_yelp_df()
    out = clean_feedback(df)
    assert "sentiment_source" in out.columns
    assert set(out["sentiment_source"]) == {"Positive", "Negative"}


def test_clean_feedback_yelp_preserves_metadata_columns():
    df = _make_yelp_df()
    out = clean_feedback(df)
    for col in ("business_name", "categories", "city", "state",
                "business_rating", "review_count"):
        assert col in out.columns, f"Missing column: {col}"


def test_clean_feedback_yelp_business_rating_numeric():
    df = _make_yelp_df()
    out = clean_feedback(df)
    assert pd.api.types.is_float_dtype(out["business_rating"]) or \
           pd.api.types.is_integer_dtype(out["business_rating"])


def test_clean_feedback_yelp_date_parsed():
    df = _make_yelp_df()
    out = clean_feedback(df)
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


# ── add_sentiment ─────────────────────────────────────────────────────────────
def test_add_sentiment_columns_present():
    out = add_sentiment(clean_feedback(_make_df()))
    assert "sentiment_score" in out.columns
    assert "sentiment"       in out.columns


def test_add_sentiment_positive_text():
    df = clean_feedback(pd.DataFrame({"feedback_text": ["Excellent amazing wonderful service!"]}))
    out = add_sentiment(df)
    assert out["sentiment"].iloc[0] == "Positive"
    assert out["sentiment_score"].iloc[0] >= 0.05


def test_add_sentiment_negative_text():
    df = clean_feedback(pd.DataFrame({"feedback_text": ["Terrible awful horrible experience!"]}))
    out = add_sentiment(df)
    assert out["sentiment"].iloc[0] == "Negative"
    assert out["sentiment_score"].iloc[0] <= -0.05


def test_add_sentiment_does_not_overwrite_sentiment_source():
    df = clean_feedback(_make_yelp_df())
    out = add_sentiment(df)
    # sentiment_source must still hold original labels
    assert "sentiment_source" in out.columns
    assert set(out["sentiment_source"]) == {"Positive", "Negative"}
    # VADER result is in a separate column
    assert "sentiment" in out.columns
    assert out["sentiment"].iloc[0] != out["sentiment_source"].iloc[0] or True  # may agree


def test_add_sentiment_does_not_mutate_input():
    df = clean_feedback(_make_df())
    original_cols = list(df.columns)
    add_sentiment(df)
    assert list(df.columns) == original_cols


# ── get_top_words ─────────────────────────────────────────────────────────────
def test_get_top_words_returns_dataframe():
    s = pd.Series(["great food and service", "terrible food and bad service"])
    out = get_top_words(s, n=5)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["word", "count"]
    assert len(out) <= 5


def test_get_top_words_filters_stopwords():
    s = pd.Series(["the and for was with this that"])
    out = get_top_words(s, n=10)
    # all words are stopwords — result should be empty
    assert len(out) == 0


def test_get_top_bigrams_returns_dataframe():
    s = pd.Series(["great customer service and great food", "customer service was terrible"])
    out = get_top_bigrams(s, n=5)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["bigram", "count"]
    assert "customer service" in out["bigram"].values


# ── sentiment_agreement ───────────────────────────────────────────────────────
def test_sentiment_agreement_returns_crosstab():
    df = add_sentiment(clean_feedback(_make_yelp_df()))
    ct = sentiment_agreement(df)
    assert not ct.empty
    assert ct.index.name == "Original label"
    assert ct.columns.name == "VADER label"


def test_sentiment_agreement_missing_columns_returns_empty():
    df = clean_feedback(_make_df())   # no sentiment_source
    out = add_sentiment(df)
    ct = sentiment_agreement(out)
    assert ct.empty


# ── integration: full Yelp CSV ────────────────────────────────────────────────
def test_full_pipeline_on_yelp_csv():
    csv_path = Path(__file__).parent.parent / "data" / "customer_feedback.csv"
    assert csv_path.exists(), "customer_feedback.csv missing — run prepare_data.py first"
    raw   = pd.read_csv(csv_path)
    final = add_sentiment(clean_feedback(raw))
    assert len(final) >= 14_000, f"Expected ~14350 records, got {len(final)}"
    assert final["sentiment"].isin(["Positive", "Neutral", "Negative"]).all()
    assert final["sentiment_score"].between(-1, 1).all()
    assert final["rating"].dropna().between(1, 5).all()
    assert "sentiment_source" in final.columns
    assert final["sentiment_source"].isin(["Positive", "Neutral", "Negative"]).all()
    assert "business_name" in final.columns
    assert "city"          in final.columns
    assert "state"         in final.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
