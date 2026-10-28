"""
ChurnSense — Customer Feedback EDA Dashboard
Real-data Yelp review analytics pipeline.
Run: py -3.12 -m streamlit run app.py
"""
from pathlib import Path
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.pipeline import clean_feedback, add_sentiment, get_top_words, sentiment_agreement

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "customer_feedback.csv"
OUT          = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SENT_COLORS = {"Positive": "#4caf50", "Neutral": "#ff9800", "Negative": "#f44336"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense | Yelp Feedback EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("ChurnSense")
st.sidebar.caption("Customer Feedback Analytics")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("Upload a feedback CSV", type=["csv"])
if uploaded:
    raw = pd.read_csv(uploaded)
    st.sidebar.success(f"Loaded uploaded file: {uploaded.name}")
else:
    raw = pd.read_csv(DEFAULT_DATA)
    st.sidebar.info("Using Yelp dataset (14,351 reviews).")

# ── Load & process ────────────────────────────────────────────────────────────
with st.spinner("Cleaning data and running VADER sentiment analysis…"):
    try:
        cleaned = add_sentiment(clean_feedback(raw))
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        st.stop()

# Save cleaned CSV
cleaned.to_csv(OUT / "cleaned_feedback.csv", index=False)

# ── Derived values ────────────────────────────────────────────────────────────
total_raw     = len(raw)
total_clean   = len(cleaned)
avg_rating    = cleaned["rating"].mean()
med_rating    = cleaned["rating"].median()
neg_pct       = (cleaned["sentiment"] == "Negative").mean() * 100
n_businesses  = cleaned["business_name"].nunique() if "business_name" in cleaned.columns else 0
n_cities      = cleaned["city"].nunique()           if "city"          in cleaned.columns else 0
n_states      = cleaned["state"].nunique()          if "state"         in cleaned.columns else 0
has_source    = "sentiment_source" in cleaned.columns

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")

vader_opts = ["All"] + sorted(cleaned["sentiment"].unique().tolist())
sel_vader  = st.sidebar.selectbox("VADER Sentiment", vader_opts)

rating_opts = ["All"] + sorted(cleaned["rating"].dropna().astype(int).unique().tolist())
sel_rating  = st.sidebar.selectbox("Star Rating", rating_opts)

if "city" in cleaned.columns:
    city_opts = ["All"] + sorted(cleaned["city"].unique().tolist())
    sel_city  = st.sidebar.selectbox("City", city_opts)
else:
    sel_city = "All"

if "state" in cleaned.columns:
    state_opts = ["All"] + sorted(cleaned["state"].unique().tolist())
    sel_state  = st.sidebar.selectbox("State", state_opts)
else:
    sel_state = "All"

# Apply filters
view = cleaned.copy()
if sel_vader  != "All": view = view[view["sentiment"]  == sel_vader]
if sel_rating != "All": view = view[view["rating"]     == int(sel_rating)]
if sel_city   != "All": view = view[view["city"]       == sel_city]
if sel_state  != "All": view = view[view["state"]      == sel_state]

# ── Main title ────────────────────────────────────────────────────────────────
st.title("📊 ChurnSense — Yelp Customer Feedback EDA")
st.caption(
    "Real-world Yelp review analytics: sentiment analysis, rating patterns, "
    "geographic distribution, and business insights. "
    "Negative sentiment indicates possible dissatisfaction — not verified churn."
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## A. Overview")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Raw records",       f"{total_raw:,}")
c2.metric("Cleaned records",   f"{total_clean:,}")
c3.metric("Avg star rating",   f"{avg_rating:.2f} / 5")
c4.metric("Negative (VADER)",  f"{neg_pct:.1f}%")
c5.metric("Unique businesses", f"{n_businesses:,}")
c6.metric("Cities / States",   f"{n_cities} / {n_states}")

removed = total_raw - total_clean
if removed:
    st.info(f"Data cleaning removed **{removed:,}** duplicate record(s) from the raw dataset.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## B. Dataset Explorer")

search = st.text_input("Search review text (case-insensitive)", "")
display = view.copy()
if search.strip():
    display = display[display["feedback_text"].str.contains(search.strip(), case=False, na=False)]

st.caption(f"Showing **{len(display):,}** records after filters{' and search' if search else ''}.")

show_cols = ["feedback_id", "date", "rating", "sentiment", "sentiment_score",
             "feedback_text", "business_name", "city", "state", "categories",
             "business_rating", "review_count"]
show_cols = [c for c in show_cols if c in display.columns]
st.dataframe(display[show_cols].head(500), use_container_width=True, height=320)

if len(display) > 500:
    st.caption("Preview limited to 500 rows. Download the full cleaned dataset below.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — VISUAL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## C. Visual Analytics")

# ── Row 1: Rating distribution + Original sentiment ──────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Review Rating Distribution")
    rc = cleaned["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(rc.index.astype(str), rc.values, color="mediumpurple", edgecolor="white")
    ax.set_xlabel("Stars (1–5)")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Rating Distribution")
    for bar in bars:
        ax.annotate(f"{int(bar.get_height()):,}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

with col2:
    if has_source:
        st.subheader("Original Sentiment (rating_review)")
        sc = cleaned["sentiment_source"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(sc.index, sc.values,
                      color=[SENT_COLORS.get(s, "grey") for s in sc.index],
                      edgecolor="white")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Original Sentiment Distribution")
        for bar in bars:
            ax.annotate(f"{int(bar.get_height()):,}",
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Original sentiment labels not available in this dataset.")

# ── Row 2: VADER sentiment + Comparison ──────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("VADER Sentiment Distribution")
    vc = cleaned["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(vc.index, vc.values,
                  color=[SENT_COLORS.get(s, "grey") for s in vc.index],
                  edgecolor="white")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("VADER Sentiment Distribution")
    for bar in bars:
        ax.annotate(f"{int(bar.get_height()):,}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

with col4:
    if has_source:
        st.subheader("Original vs VADER Sentiment")
        agree = sentiment_agreement(cleaned)
        if not agree.empty:
            import numpy as np
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(agree.values, cmap="Blues")
            ax.set_xticks(range(len(agree.columns)))
            ax.set_xticklabels(agree.columns)
            ax.set_yticks(range(len(agree.index)))
            ax.set_yticklabels(agree.index)
            ax.set_xlabel("VADER label")
            ax.set_ylabel("Original label")
            ax.set_title("Sentiment Comparison (counts)\n"
                         "Note: original labels are reference only")
            for i in range(len(agree.index)):
                for j in range(len(agree.columns)):
                    ax.text(j, i, f"{agree.values[i,j]:,}",
                            ha="center", va="center", fontsize=10,
                            color="white" if agree.values[i,j] > agree.values.max()*0.6 else "black")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
    else:
        st.info("Comparison requires original sentiment labels.")

# ── Row 3: Sentiment by star rating + Avg rating by sentiment ────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("VADER Sentiment by Star Rating")
    sent_star = cleaned.groupby(["rating", "sentiment"]).size().unstack(fill_value=0)
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in sent_star.columns:
            sent_star[col] = 0
    sent_star = sent_star[["Positive", "Neutral", "Negative"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    sent_star.plot(kind="bar", stacked=True, ax=ax,
                   color=["#4caf50", "#ff9800", "#f44336"], edgecolor="white")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Sentiment by Star Rating")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

with col6:
    st.subheader("Avg Star Rating by VADER Sentiment")
    rbs = cleaned.groupby("sentiment")["rating"].mean().reindex(["Positive", "Neutral", "Negative"])
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(rbs.index, rbs.values,
                  color=[SENT_COLORS.get(s, "grey") for s in rbs.index],
                  edgecolor="white")
    ax.set_xlabel("VADER Sentiment")
    ax.set_ylabel("Average Stars")
    ax.set_ylim(0, 5.5)
    ax.set_title("Average Rating by Sentiment")
    for bar in bars:
        ax.annotate(f"{bar.get_height():.2f}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ── Row 4: Top cities + Top states ───────────────────────────────────────────
if "city" in cleaned.columns and "state" in cleaned.columns:
    col7, col8 = st.columns(2)
    with col7:
        st.subheader("Top 15 Cities by Review Volume")
        top_cities = cleaned["city"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        top_cities[::-1].plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
        ax.set_xlabel("Number of Reviews")
        ax.set_title("Top 15 Cities")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col8:
        st.subheader("Review Volume by State")
        top_states = cleaned["state"].value_counts().head(16)
        fig, ax = plt.subplots(figsize=(7, 5))
        top_states[::-1].plot(kind="barh", ax=ax, color="teal", edgecolor="white")
        ax.set_xlabel("Number of Reviews")
        ax.set_title("Reviews by State")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

# ── Row 5: Top businesses + Word frequency ───────────────────────────────────
col9, col10 = st.columns(2)

with col9:
    if "business_name" in cleaned.columns:
        st.subheader("Top 15 Businesses by Review Volume")
        top_biz = cleaned["business_name"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        top_biz[::-1].plot(kind="barh", ax=ax, color="coral", edgecolor="white")
        ax.set_xlabel("Number of Reviews")
        ax.set_title("Top 15 Businesses")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

with col10:
    st.subheader("Top 20 Most Frequent Words")
    top_words = get_top_words(cleaned["feedback_text"], 20)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top_words["word"][::-1], top_words["count"][::-1], color="teal")
    ax.set_xlabel("Frequency")
    ax.set_title("Top 20 Words (All Reviews)")
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ── Negative review word frequency ───────────────────────────────────────────
st.subheader("Top 20 Words in Negative Reviews")
neg_reviews = cleaned.loc[cleaned["sentiment"] == "Negative", "feedback_text"]
if len(neg_reviews):
    top_neg = get_top_words(neg_reviews, 20)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_neg["word"][::-1], top_neg["count"][::-1], color="#f44336")
    ax.set_xlabel("Frequency")
    ax.set_title("Top 20 Words — Negative Reviews")
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)
else:
    st.info("No negative reviews in current filter selection.")

# ── Business rating comparison ────────────────────────────────────────────────
if "business_rating" in cleaned.columns and "business_name" in cleaned.columns:
    st.subheader("Business Rating vs Review Rating (Top 20 Businesses)")
    top20_biz = cleaned["business_name"].value_counts().head(20).index
    biz_df = (cleaned[cleaned["business_name"].isin(top20_biz)]
              .groupby("business_name")[["rating", "business_rating"]]
              .mean().round(2).sort_values("rating"))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(biz_df))
    ax.barh([i - 0.2 for i in x], biz_df["rating"],        height=0.4, label="Avg Review Rating",   color="steelblue")
    ax.barh([i + 0.2 for i in x], biz_df["business_rating"], height=0.4, label="Business Star Rating", color="coral")
    ax.set_yticks(list(x))
    ax.set_yticklabels(biz_df.index, fontsize=8)
    ax.set_xlabel("Stars")
    ax.set_xlim(0, 5.5)
    ax.set_title("Review Rating vs Business Rating — Top 20 Businesses")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D — REVIEW INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## D. Review Insights")

def _show_sample(label: str, color: str, mask: pd.Series, n: int = 3):
    st.markdown(f"**{label}**")
    sample = cleaned[mask].sample(min(n, mask.sum()), random_state=42)
    for _, row in sample.iterrows():
        biz  = row.get("business_name", "")
        city = row.get("city", "")
        with st.container():
            st.markdown(
                f'<div style="border-left:4px solid {color};padding:8px 12px;'
                f'margin-bottom:8px;background:#f9f9f9;border-radius:4px;">'
                f'<b>{biz}</b> — {city} &nbsp;|&nbsp; '
                f'{"⭐" * int(row["rating"])} ({int(row["rating"])}/5) &nbsp;|&nbsp; '
                f'VADER: <b>{row["sentiment"]}</b> ({row["sentiment_score"]:.3f})<br>'
                f'<span style="font-size:0.9em">{str(row["feedback_text"])[:500]}'
                f'{"…" if len(str(row["feedback_text"])) > 500 else ""}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

col_a, col_b, col_c = st.columns(3)
with col_a:
    _show_sample("Positive Reviews", "#4caf50", cleaned["sentiment"] == "Positive")
with col_b:
    _show_sample("Neutral Reviews",  "#ff9800", cleaned["sentiment"] == "Neutral")
with col_c:
    _show_sample("Negative Reviews", "#f44336", cleaned["sentiment"] == "Negative")

# ── Summary statistics table ──────────────────────────────────────────────────
st.markdown("### Summary Statistics")
stats_df = pd.DataFrame({
    "Metric": [
        "Total raw records", "Total cleaned records", "Records removed",
        "Avg review rating", "Median review rating",
        "VADER Positive", "VADER Neutral", "VADER Negative",
        "Unique businesses", "Unique cities", "Unique states",
    ],
    "Value": [
        f"{total_raw:,}", f"{total_clean:,}", f"{total_raw - total_clean:,}",
        f"{avg_rating:.3f} / 5", f"{med_rating:.1f} / 5",
        f"{(cleaned['sentiment']=='Positive').sum():,} ({(cleaned['sentiment']=='Positive').mean()*100:.1f}%)",
        f"{(cleaned['sentiment']=='Neutral').sum():,}  ({(cleaned['sentiment']=='Neutral').mean()*100:.1f}%)",
        f"{(cleaned['sentiment']=='Negative').sum():,} ({(cleaned['sentiment']=='Negative').mean()*100:.1f}%)",
        f"{n_businesses:,}", f"{n_cities:,}", f"{n_states:,}",
    ]
})
st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E — DOWNLOADS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## E. Downloads")

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "⬇ Download Cleaned Dataset (CSV)",
        data=cleaned.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_feedback.csv",
        mime="text/csv",
    )
with dl2:
    summary_path = OUT / "analysis_summary.txt"
    if summary_path.exists():
        st.download_button(
            "⬇ Download Analysis Summary (TXT)",
            data=summary_path.read_bytes(),
            file_name="analysis_summary.txt",
            mime="text/plain",
        )

st.markdown("---")
st.caption(
    "Data source: Yelp review dataset (14,351 records). "
    "VADER sentiment is an approximate signal computed from review text — not a definitive label. "
    "Original sentiment labels (rating_review) are reference labels and may contain errors. "
    "This dataset does not contain verified customer churn outcomes or reliable customer identifiers. "
    "Negative sentiment indicates possible dissatisfaction only."
)
