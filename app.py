from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from src.pipeline import clean_feedback, add_sentiment

ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "customer_feedback.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

st.set_page_config(page_title="ChurnSense | Customer Feedback EDA", page_icon="📊", layout="wide")
st.title("ChurnSense — Customer Feedback EDA")
st.caption("An end-to-end pipeline for uncovering structural patterns in multi-channel customer feedback.")

uploaded = st.sidebar.file_uploader("Upload feedback CSV", type=["csv"])
if uploaded:
    raw = pd.read_csv(uploaded)
else:
    raw = pd.read_csv(DEFAULT_DATA)
    st.sidebar.info("Using the included synthetic sample dataset.")

try:
    cleaned = add_sentiment(clean_feedback(raw))
except Exception as exc:
    st.error(str(exc))
    st.stop()

OUT.mkdir(exist_ok=True)
cleaned.to_csv(OUT / "cleaned_feedback.csv", index=False)

total = len(cleaned)
channels = cleaned["channel"].nunique()
avg_rating = cleaned["rating"].mean()
negative_share = (cleaned["sentiment"] == "Negative").mean() * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Feedback records", f"{total:,}")
c2.metric("Channels", f"{channels:,}")
c3.metric("Average rating", "N/A" if pd.isna(avg_rating) else f"{avg_rating:.2f}/5")
c4.metric("Negative sentiment", f"{negative_share:.1f}%" if cleaned["sentiment_score"].notna().any() else "N/A")

st.subheader("Dataset preview")
st.dataframe(cleaned, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Feedback volume by channel")
    volume = cleaned["channel"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots()
    volume.plot(kind="bar", ax=ax)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Number of feedback records")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Sentiment distribution")
    sentiment_counts = cleaned["sentiment"].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of feedback records")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Rating by channel")
rating_data = cleaned.dropna(subset=["rating"])
if not rating_data.empty:
    st.bar_chart(rating_data.groupby("channel")["rating"].mean())
else:
    st.info("No usable rating column was found in the uploaded file.")

st.subheader("Common words")
from collections import Counter
import re
stopwords = set("the and for was with this that have had are but not you your our they their from very after when were still has".split())
tokens = []
for value in cleaned["feedback_text"].astype(str):
    tokens.extend(w for w in re.findall(r"[a-zA-Z]{3,}", value.lower()) if w not in stopwords)
common = pd.DataFrame(Counter(tokens).most_common(15), columns=["word", "count"])
if not common.empty:
    st.dataframe(common, hide_index=True, use_container_width=True)
else:
    st.info("No words available to summarize.")

st.download_button(
    "Download cleaned dataset",
    data=cleaned.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_feedback.csv",
    mime="text/csv",
)
st.caption("Use real, authorized data for final analysis. Sentiment is an approximate signal, not a definitive label.")
