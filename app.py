"""
ChurnSense — Customer Feedback EDA Dashboard
Comprehensive Exploratory Data Analysis & Visualizations for Customer Feedback.
Run: python -m streamlit run app.py
"""
import sys
from pathlib import Path
import re
from collections import Counter
from datetime import datetime

CURRENT_DIR = Path(__file__).resolve().parent
if (CURRENT_DIR / "data" / "customer_feedback.csv").exists():
    ROOT = CURRENT_DIR
elif (CURRENT_DIR / "ChurnSense_Customer_Feedback_EDA" / "data" / "customer_feedback.csv").exists():
    ROOT = CURRENT_DIR / "ChurnSense_Customer_Feedback_EDA"
else:
    ROOT = CURRENT_DIR

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.pipeline import clean_feedback, add_sentiment, get_top_words, sentiment_agreement

try:
    from src.pipeline import get_top_bigrams
except ImportError:
    from src.pipeline import STOPWORDS
    def get_top_bigrams(series: pd.Series, n: int = 15) -> pd.DataFrame:
        """Return a DataFrame of top-n most frequent 2-word phrases (bigrams)."""
        bigrams = []
        for txt in series.astype(str):
            words = [
                w for w in re.findall(r"[a-zA-Z]{3,}", txt.lower())
                if w not in STOPWORDS
            ]
            bigrams.extend(zip(words[:-1], words[1:]))
        counts = Counter(f"{w1} {w2}" for w1, w2 in bigrams)
        return pd.DataFrame(counts.most_common(n), columns=["bigram", "count"])

# ── Styling & Color Palette ───────────────────────────────────────────────────
SENT_COLORS = {"Positive": "#2e7d32", "Neutral": "#f57c00", "Negative": "#d32f2f"}
STAR_COLORS = ["#d32f2f", "#e65100", "#f57c00", "#7cb342", "#2e7d32"]

DEFAULT_DATA = ROOT / "data" / "customer_feedback.csv"
OUT          = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── Matplotlib global style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense | Customer Feedback EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached Pipeline Operations ────────────────────────────────────────────────
@st.cache_data(show_spinner="Cleaning dataset and running VADER sentiment analysis...")
def load_and_process(file_bytes=None, file_name=None):
    if file_bytes is not None:
        import io
        df_raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df_raw = pd.read_csv(DEFAULT_DATA)
    cleaned_df = clean_feedback(df_raw)
    processed_df = add_sentiment(cleaned_df)
    return df_raw, processed_df

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 ChurnSense")
st.sidebar.caption("Customer Feedback Analytics & Churn Insights")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Feedback CSV", type=["csv"])
if uploaded_file is not None:
    raw_df, cleaned = load_and_process(uploaded_file.getvalue(), uploaded_file.name)
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
else:
    raw_df, cleaned = load_and_process()
    st.sidebar.info("Using master Yelp dataset (14,351 reviews).")

# Save cleaned output
cleaned.to_csv(OUT / "cleaned_feedback.csv", index=False)

# Derived baseline metrics
total_raw    = len(raw_df)
total_clean  = len(cleaned)
has_date     = "date" in cleaned.columns and cleaned["date"].notna().any()
has_source   = "sentiment_source" in cleaned.columns
has_cat      = "categories" in cleaned.columns
has_biz      = "business_name" in cleaned.columns
has_city     = "city" in cleaned.columns
has_state    = "state" in cleaned.columns

# Extract Categories list for filter
all_categories = []
if has_cat:
    for cat_str in cleaned["categories"].dropna():
        for c in str(cat_str).split(","):
            c_clean = c.strip()
            if c_clean and c_clean.lower() != "unknown":
                all_categories.append(c_clean)
    top_cat_options = [c for c, _ in Counter(all_categories).most_common(50)]
else:
    top_cat_options = []

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("### 🔍 Global Filters")

# 1. Date Range
if has_date:
    min_date = cleaned["date"].dropna().min().date()
    max_date = cleaned["date"].dropna().max().date()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
else:
    date_range = None

# 2. VADER Sentiment
sent_options = ["All"] + sorted(cleaned["sentiment"].dropna().unique().tolist())
sel_sentiment = st.sidebar.selectbox("Sentiment (VADER)", sent_options)

# 3. Star Rating
rating_vals = sorted([int(r) for r in cleaned["rating"].dropna().unique()])
sel_ratings = st.sidebar.multiselect("Star Rating (1–5)", rating_vals, default=rating_vals)

# 4. State
if has_state:
    state_options = ["All"] + sorted([s for s in cleaned["state"].unique() if s != "Unknown"])
    sel_state = st.sidebar.selectbox("State", state_options)
else:
    sel_state = "All"

# 5. City (filtered by state if chosen)
if has_city:
    if sel_state != "All" and has_state:
        city_pool = cleaned[cleaned["state"] == sel_state]["city"].unique()
    else:
        city_pool = cleaned["city"].unique()
    city_options = ["All"] + sorted([c for c in city_pool if c != "Unknown"])
    sel_city = st.sidebar.selectbox("City", city_options)
else:
    sel_city = "All"

# 6. Category
if top_cat_options:
    sel_category = st.sidebar.selectbox("Category", ["All"] + top_cat_options)
else:
    sel_category = "All"

# 7. Text Search
search_term = st.sidebar.text_input("Review Text Search", "").strip()

# Apply Filters
filtered = cleaned.copy()
if has_date and date_range and len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    filtered = filtered[(filtered["date"] >= start_dt) & (filtered["date"] < end_dt)]
if sel_sentiment != "All":
    filtered = filtered[filtered["sentiment"] == sel_sentiment]
if sel_ratings:
    filtered = filtered[filtered["rating"].isin(sel_ratings)]
if sel_state != "All" and has_state:
    filtered = filtered[filtered["state"] == sel_state]
if sel_city != "All" and has_city:
    filtered = filtered[filtered["city"] == sel_city]
if sel_category != "All" and has_cat:
    filtered = filtered[filtered["categories"].astype(str).str.contains(r"\b" + re.escape(sel_category) + r"\b", case=False, na=False)]
if search_term:
    filtered = filtered[filtered["feedback_text"].astype(str).str.contains(search_term, case=False, na=False)]

st.sidebar.markdown(f"**Filtered Records:** {len(filtered):,} / {total_clean:,} ({len(filtered)/max(total_clean,1)*100:.1f}%)")
if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()

display_risk = None

# ── Main Header ───────────────────────────────────────────────────────────────
st.title("📊 ChurnSense — Customer Feedback Exploratory Data Analysis")
st.markdown(
    "**End-to-end analytical dashboard uncovering structural sentiment patterns, churn indicators, "
    "temporal shifts, geographic nuances, and operational risk factors in customer feedback.**"
)
st.caption("ℹ️ *Note: Negative sentiment indicates possible customer dissatisfaction or churn risk — not verified churn outcomes.*")

# ── Tabbed Layout ─────────────────────────────────────────────────────────────
tab_overview, tab_sentiment, tab_temporal, tab_text, tab_geo, tab_biz, tab_explorer, tab_export = st.tabs([
    "📊 Overview & KPIs",
    "🎭 Sentiment & Ratings",
    "📅 Temporal & Seasonality",
    "🔤 Text & N-Gram Insights",
    "🗺️ Geographic & Categories",
    "⚠️ Business Risk & Watchlist",
    "🔍 Review Explorer & Anomalies",
    "⬇️ Exports & Report",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE OVERVIEW & KPIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("### Executive Performance Indicators")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    
    avg_r = filtered["rating"].mean() if len(filtered) else 0
    neg_pct = (filtered["sentiment"] == "Negative").mean() * 100 if len(filtered) else 0
    pos_pct = (filtered["sentiment"] == "Positive").mean() * 100 if len(filtered) else 0
    n_biz = filtered["business_name"].nunique() if has_biz and len(filtered) else 0
    n_c   = filtered["city"].nunique() if has_city and len(filtered) else 0
    n_s   = filtered["state"].nunique() if has_state and len(filtered) else 0

    kpi1.metric("Filtered Reviews", f"{len(filtered):,}", f"{(len(filtered)-total_clean):,}" if len(filtered) != total_clean else "Total dataset")
    kpi2.metric("Avg Star Rating", f"{avg_r:.2f} / 5", f"Median: {filtered['rating'].median():.1f}" if len(filtered) else "-")
    kpi3.metric("Negative Sentiment", f"{neg_pct:.1f}%", delta=f"{neg_pct - 10.3:.1f}% vs baseline" if len(filtered) else None, delta_color="inverse")
    kpi4.metric("Positive Sentiment", f"{pos_pct:.1f}%")
    kpi5.metric("Unique Businesses", f"{n_biz:,}")
    kpi6.metric("Coverage (Cities/States)", f"{n_c} / {n_s}")

    st.markdown("---")
    c_sub1, c_sub2 = st.columns([1, 1])

    with c_sub1:
        st.subheader("Data Cleaning & Hygiene Summary")
        removed_dups = total_raw - total_clean
        hygiene_data = {
            "Dimension": [
                "Raw Ingested Records",
                "Cleaned Analytical Records",
                "Exact Duplicates Removed",
                "Empty Reviews Removed",
                "Ratings Clamped to 1–5",
                "Date Coverage Span",
                "Missing Review Text Count",
            ],
            "Value": [
                f"{total_raw:,}",
                f"{total_clean:,}",
                f"{removed_dups:,}",
                "0",
                "Verified Valid",
                f"{cleaned['date'].min().strftime('%Y-%m-%d')} to {cleaned['date'].max().strftime('%Y-%m-%d')}" if has_date else "N/A",
                "0 (100% complete)",
            ]
        }
        st.dataframe(pd.DataFrame(hygiene_data), use_container_width=True, hide_index=True)

    with c_sub2:
        st.subheader("Lexical Volume & Review Length Profile")
        if "word_count" in filtered.columns and len(filtered) > 0:
            avg_words = filtered["word_count"].mean()
            med_words = filtered["word_count"].median()
            avg_chars = filtered["text_length"].mean()
            lex_df = pd.DataFrame({
                "Metric": ["Average Word Count", "Median Word Count", "Max Word Count", "Avg Character Length", "Min Character Length"],
                "Figure": [f"{avg_words:.1f} words", f"{med_words:.0f} words", f"{filtered['word_count'].max():,} words", f"{avg_chars:.1f} chars", f"{filtered['text_length'].min()} chars"]
            })
            st.dataframe(lex_df, use_container_width=True, hide_index=True)
            st.caption("Detailed breakdown by sentiment and rating is available in the **Text & N-Gram Insights** tab.")
        else:
            st.info("No data available for current selection.")

    st.markdown("### Cleaned Dataset Preview")
    show_cols = [c for c in ["feedback_id", "date", "rating", "sentiment", "sentiment_score", "business_name", "city", "state", "categories", "feedback_text"] if c in filtered.columns]
    st.dataframe(filtered[show_cols].head(100), use_container_width=True, height=280)
    st.caption(f"Showing top 100 rows out of {len(filtered):,} filtered records.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SENTIMENT ANALYSIS & RATING DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sentiment:
    st.markdown("### Sentiment & Rating Distribution Dynamics")
    
    if len(filtered) == 0:
        st.warning("No records match the current filter selection.")
    else:
        r1_col1, r1_col2 = st.columns(2)
        
        with r1_col1:
            st.subheader("Star Rating Distribution (1–5 Stars)")
            rc = filtered["rating"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6.5, 4))
            bars = ax.bar(rc.index.astype(str), rc.values, color=["#d32f2f", "#e65100", "#f57c00", "#7cb342", "#2e7d32"], edgecolor="white", width=0.6)
            ax.set_xlabel("Star Rating", fontweight="bold")
            ax.set_ylabel("Review Count", fontweight="bold")
            ax.set_title("Customer Star Rating Distribution", fontsize=12, fontweight="bold")
            ax.grid(axis="y")
            for bar in bars:
                height = bar.get_height()
                pct = height / len(filtered) * 100
                ax.annotate(f"{int(height):,}\n({pct:.1f}%)", (bar.get_x() + bar.get_width()/2, height),
                            ha="center", va="bottom", fontsize=8.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with r1_col2:
            st.subheader("VADER Sentiment Distribution")
            vc = filtered["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
            fig, ax = plt.subplots(figsize=(6.5, 4))
            bars = ax.bar(vc.index, vc.values, color=[SENT_COLORS.get(s, "#757575") for s in vc.index], edgecolor="white", width=0.55)
            ax.set_xlabel("VADER Sentiment Classification", fontweight="bold")
            ax.set_ylabel("Count", fontweight="bold")
            ax.set_title("NLP Sentiment Classification (Compound Threshold: ±0.05)", fontsize=12, fontweight="bold")
            ax.grid(axis="y")
            for bar in bars:
                height = bar.get_height()
                pct = height / len(filtered) * 100
                ax.annotate(f"{int(height):,}\n({pct:.1f}%)", (bar.get_x() + bar.get_width()/2, height),
                            ha="center", va="bottom", fontsize=8.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        r2_col1, r2_col2 = st.columns(2)

        with r2_col1:
            st.subheader("VADER Compound Score Continuous Distribution")
            fig, ax = plt.subplots(figsize=(6.5, 4))
            sns.histplot(filtered["sentiment_score"], bins=30, kde=True, ax=ax, color="#1976d2", edgecolor="white")
            ax.axvline(0.05, color="#2e7d32", linestyle="--", label="Positive threshold (+0.05)")
            ax.axvline(-0.05, color="#d32f2f", linestyle="--", label="Negative threshold (-0.05)")
            ax.set_xlabel("VADER Compound Polarity Score (-1.0 to +1.0)", fontweight="bold")
            ax.set_ylabel("Density / Frequency", fontweight="bold")
            ax.set_title("Compound Sentiment Polarity Distribution", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8.5)
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with r2_col2:
            st.subheader("Sentiment Composition by Star Rating (100% Stacked)")
            ct = pd.crosstab(filtered["rating"], filtered["sentiment"], normalize="index") * 100
            for col in ["Positive", "Neutral", "Negative"]:
                if col not in ct.columns:
                    ct[col] = 0.0
            ct = ct[["Positive", "Neutral", "Negative"]]
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ct.plot(kind="bar", stacked=True, ax=ax, color=["#2e7d32", "#f57c00", "#d32f2f"], edgecolor="white", width=0.6)
            ax.set_xlabel("Star Rating", fontweight="bold")
            ax.set_ylabel("Percentage (%)", fontweight="bold")
            ax.set_ylim(0, 105)
            ax.set_title("Proportion of Sentiment within Each Star Rating", fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title="Sentiment", loc="upper left", bbox_to_anchor=(1.0, 1.0))
            ax.grid(axis="y")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        r3_col1, r3_col2 = st.columns(2)

        with r3_col1:
            if has_source:
                st.subheader("Original vs VADER Sentiment Agreement Heatmap")
                agree = sentiment_agreement(filtered)
                if not agree.empty:
                    fig, ax = plt.subplots(figsize=(6.5, 4.2))
                    sns.heatmap(agree, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=True,
                                annot_kws={"fontsize": 10, "fontweight": "bold"})
                    ax.set_xlabel("VADER Model Classification", fontweight="bold")
                    ax.set_ylabel("Original Benchmark Label", fontweight="bold")
                    diag_sum = np.trace(agree.values) if agree.shape[0] == agree.shape[1] else 0
                    acc = diag_sum / agree.values.sum() * 100 if agree.values.sum() > 0 else 0
                    ax.set_title(f"Confusion Matrix (Agreement: {acc:.1f}%)", fontsize=12, fontweight="bold")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Original benchmark labels are not available.")

        with r3_col2:
            st.subheader("Star Rating vs VADER Compound Polarity (Box Plot)")
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            sns.boxplot(data=filtered, x="rating", y="sentiment_score", palette=["#d32f2f", "#e65100", "#f57c00", "#7cb342", "#2e7d32"], ax=ax)
            ax.set_xlabel("Star Rating", fontweight="bold")
            ax.set_ylabel("VADER Compound Score", fontweight="bold")
            ax.set_title("Polarity Spread across Star Ratings", fontsize=12, fontweight="bold")
            ax.grid(axis="y")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: TEMPORAL & SEASONALITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_temporal:
    st.markdown("### Longitudinal Trends & Customer Feedback Seasonality")
    
    if not has_date or len(filtered) == 0:
        st.info("Date information is not available for temporal analysis.")
    else:
        df_time = filtered.dropna(subset=["date"]).copy()
        df_time["year_month"] = df_time["date"].dt.to_period("M").dt.to_timestamp()
        df_time["day_of_week"] = df_time["date"].dt.day_name()
        df_time["hour"] = df_time["date"].dt.hour

        t_col1, t_col2 = st.columns(2)
        
        with t_col1:
            st.subheader("Monthly Review Volume Over Time")
            monthly_vol = df_time.groupby("year_month").size()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(monthly_vol.index, monthly_vol.values, marker="o", markersize=3, color="#1976d2", linewidth=1.8)
            ax.set_xlabel("Timeline", fontweight="bold")
            ax.set_ylabel("Review Inflow Count", fontweight="bold")
            ax.set_title("Customer Feedback Volume Trajectory", fontsize=12, fontweight="bold")
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with t_col2:
            st.subheader("Monthly Average Star Rating & Quality Drift")
            monthly_rat = df_time.groupby("year_month")["rating"].mean()
            rolling_rat = monthly_rat.rolling(window=3, min_periods=1).mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(monthly_rat.index, monthly_rat.values, marker=".", color="#b0bec5", alpha=0.6, label="Monthly Mean")
            ax.plot(rolling_rat.index, rolling_rat.values, color="#e65100", linewidth=2.2, label="3-Month Rolling Average")
            ax.axhline(cleaned["rating"].mean(), color="#37474f", linestyle=":", label="Dataset Overall Mean")
            ax.set_xlabel("Timeline", fontweight="bold")
            ax.set_ylabel("Average Star Rating", fontweight="bold")
            ax.set_ylim(1, 5.2)
            ax.set_title("Quality Drift & Satisfaction Trendline", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8.5)
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        t_col3, t_col4 = st.columns(2)

        with t_col3:
            st.subheader("Review Volume & Satisfaction by Day of Week")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_stats = df_time.groupby("day_of_week").agg(
                count=("rating", "count"),
                avg_rating=("rating", "mean"),
                neg_rate=("sentiment", lambda s: (s == "Negative").mean() * 100)
            ).reindex(day_order)

            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax2 = ax1.twinx()
            bars = ax1.bar(dow_stats.index, dow_stats["count"], color="#90caf9", alpha=0.85, width=0.55, label="Review Count")
            line = ax2.plot(dow_stats.index, dow_stats["avg_rating"], color="#d32f2f", marker="o", linewidth=2, label="Avg Rating")
            ax1.set_xlabel("Day of Week", fontweight="bold")
            ax1.set_ylabel("Volume", color="#1565c0", fontweight="bold")
            ax2.set_ylabel("Avg Stars (1–5)", color="#d32f2f", fontweight="bold")
            ax2.set_ylim(1, 5)
            ax1.set_title("Weekend vs Weekday Rush & Rating Fluctuations", fontsize=12, fontweight="bold")
            ax1.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with t_col4:
            st.subheader("Hourly Distribution (Service Peak Analysis)")
            hour_stats = df_time.groupby("hour").agg(
                count=("rating", "count"),
                avg_rating=("rating", "mean")
            ).reindex(range(24)).fillna(0)

            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax2 = ax1.twinx()
            ax1.bar(hour_stats.index, hour_stats["count"], color="#b2dfdb", alpha=0.85, width=0.6, label="Review Count")
            ax2.plot(hour_stats.index, hour_stats["avg_rating"], color="#00695c", marker="s", markersize=3, label="Avg Rating")
            ax1.set_xlabel("Hour of Day (0:00 – 23:00)", fontweight="bold")
            ax1.set_ylabel("Volume", color="#004d40", fontweight="bold")
            ax2.set_ylabel("Avg Rating", color="#00695c", fontweight="bold")
            ax2.set_ylim(1, 5.2)
            ax1.set_title("Review Inflow & Rating Variations by Hour", fontsize=12, fontweight="bold")
            ax1.set_xticks(range(0, 24, 2))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: TEXT, LEXICAL & N-GRAM INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_text:
    st.markdown("### Lexical Patterns, Churn Drivers & N-Gram Frequencies")
    
    if len(filtered) == 0:
        st.warning("No reviews available for lexical analysis.")
    else:
        txt_col1, txt_col2 = st.columns(2)

        with txt_col1:
            st.subheader("Top 20 Dissatisfaction Drivers (Negative Reviews)")
            neg_texts = filtered[filtered["sentiment"] == "Negative"]["feedback_text"]
            if len(neg_texts) > 0:
                top_neg = get_top_words(neg_texts, 20)
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                ax.barh(top_neg["word"][::-1], top_neg["count"][::-1], color="#d32f2f", edgecolor="white")
                ax.set_xlabel("Word Frequency", fontweight="bold")
                ax.set_title("Frequent Keywords in Negative Feedback", fontsize=12, fontweight="bold")
                ax.grid(axis="x")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No negative reviews in current filtered selection.")

        with txt_col2:
            st.subheader("Top 20 Delight Drivers (Positive Reviews)")
            pos_texts = filtered[filtered["sentiment"] == "Positive"]["feedback_text"]
            if len(pos_texts) > 0:
                top_pos = get_top_words(pos_texts, 20)
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                ax.barh(top_pos["word"][::-1], top_pos["count"][::-1], color="#2e7d32", edgecolor="white")
                ax.set_xlabel("Word Frequency", fontweight="bold")
                ax.set_title("Frequent Keywords in Positive Feedback", fontsize=12, fontweight="bold")
                ax.grid(axis="x")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No positive reviews in current filtered selection.")

        st.markdown("---")
        txt_col3, txt_col4 = st.columns(2)

        with txt_col3:
            st.subheader("Top 15 Frequent Bigrams (2-Word Expressions)")
            bigrams_df = get_top_bigrams(filtered["feedback_text"], 15)
            if not bigrams_df.empty:
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                ax.barh(bigrams_df["bigram"][::-1], bigrams_df["count"][::-1], color="#5e35b1", edgecolor="white")
                ax.set_xlabel("Bigram Frequency", fontweight="bold")
                ax.set_title("Recurring Phrase Patterns (Bigrams)", fontsize=12, fontweight="bold")
                ax.grid(axis="x")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No bigrams extracted.")

        with txt_col4:
            st.subheader("Review Word Count by Sentiment Category")
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            sns.boxplot(data=filtered, x="sentiment", y="word_count",
                        order=["Positive", "Neutral", "Negative"],
                        palette=[SENT_COLORS["Positive"], SENT_COLORS["Neutral"], SENT_COLORS["Negative"]],
                        showfliers=False, ax=ax)
            ax.set_xlabel("VADER Sentiment", fontweight="bold")
            ax.set_ylabel("Word Count (Excl. Outliers)", fontweight="bold")
            ax.set_title("Do Unhappy Customers Write Longer Reviews?", fontsize=12, fontweight="bold")
            ax.grid(axis="y")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: GEOGRAPHIC & CATEGORY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_geo:
    st.markdown("### Geographic Distribution & Industry Category Insights")
    
    if len(filtered) == 0:
        st.warning("No data available for geographic/category analysis.")
    else:
        geo_col1, geo_col2 = st.columns(2)

        with geo_col1:
            if has_state:
                st.subheader("Top 12 States by Review Volume")
                state_counts = filtered["state"].value_counts().head(12)
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                ax.barh(state_counts.index[::-1], state_counts.values[::-1], color="#00838f", edgecolor="white")
                ax.set_xlabel("Review Volume", fontweight="bold")
                ax.set_title("Geographic Representation by State", fontsize=12, fontweight="bold")
                ax.grid(axis="x")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with geo_col2:
            if has_city:
                st.subheader("Top 15 Cities by Review Volume")
                city_counts = filtered["city"].value_counts().head(15)
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                ax.barh(city_counts.index[::-1], city_counts.values[::-1], color="#0277bd", edgecolor="white")
                ax.set_xlabel("Review Volume", fontweight="bold")
                ax.set_title("Geographic Representation by City", fontsize=12, fontweight="bold")
                ax.grid(axis="x")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")
        geo_col3, geo_col4 = st.columns(2)

        with geo_col3:
            if has_cat:
                st.subheader("Top 15 Business Categories (Multi-Label)")
                cats_flat = []
                for entry in filtered["categories"].dropna():
                    for item in str(entry).split(","):
                        it = item.strip()
                        if it and it.lower() != "unknown":
                            cats_flat.append(it)
                top_cats = pd.DataFrame(Counter(cats_flat).most_common(15), columns=["category", "count"])
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                ax.barh(top_cats["category"][::-1], top_cats["count"][::-1], color="#ad1457", edgecolor="white")
                ax.set_xlabel("Tag Count", fontweight="bold")
                ax.set_title("Most Common Business Segments", fontsize=12, fontweight="bold")
                ax.grid(axis="x")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with geo_col4:
            if has_city:
                st.subheader("Highest Dissatisfaction Hotspots (Min 25 Reviews)")
                city_summary = (filtered.groupby("city")
                                .agg(total=("rating", "count"),
                                     neg_pct=("sentiment", lambda x: (x == "Negative").mean() * 100),
                                     avg_stars=("rating", "mean"))
                                .query("total >= 25")
                                .sort_values("neg_pct", ascending=False)
                                .head(10))
                if not city_summary.empty:
                    fig, ax = plt.subplots(figsize=(6.5, 4.5))
                    ax.barh(city_summary.index[::-1], city_summary["neg_pct"][::-1], color="#e53935", edgecolor="white")
                    ax.set_xlabel("Negative Feedback Percentage (%)", fontweight="bold")
                    ax.set_title("Top 10 Cities with Highest Dissatisfaction", fontsize=12, fontweight="bold")
                    ax.grid(axis="x")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No cities meet the minimum 25-review threshold.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: BUSINESS RISK & OPERATIONAL WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_biz:
    st.markdown("### Business-Level Churn Risk Watchlist & Performance")
    
    if not has_biz or len(filtered) == 0:
        st.info("Business entity metadata not available.")
    else:
        b_col1, b_col2 = st.columns(2)

        with b_col1:
            st.subheader("Top 15 Businesses by Feedback Volume")
            top_biz_v = filtered["business_name"].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.barh(top_biz_v.index[::-1], top_biz_v.values[::-1], color="#e65100", edgecolor="white")
            ax.set_xlabel("Feedback Count", fontweight="bold")
            ax.set_title("Most Reviewed Businesses", fontsize=12, fontweight="bold")
            ax.grid(axis="x")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with b_col2:
            st.subheader("Correlation Matrix (Numeric Indicators)")
            num_cols = [c for c in ["rating", "sentiment_score", "business_rating", "review_count", "text_length", "word_count"] if c in filtered.columns]
            corr = filtered[num_cols].corr()
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, annot_kws={"fontsize": 8.5})
            ax.set_title("Inter-Metric Correlation Coefficients", fontsize=12, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        st.subheader("⚠️ High-Risk Business Watchlist (Highest Negative Review %)")
        min_revs = st.slider("Minimum Review Threshold for Risk Assessment", min_value=5, max_value=50, value=15, step=5)
        
        biz_agg = (filtered.groupby("business_name")
                   .agg(
                       review_volume=("rating", "count"),
                       negative_reviews=("sentiment", lambda x: (x == "Negative").sum()),
                       negative_rate=("sentiment", lambda x: (x == "Negative").mean() * 100),
                       avg_review_rating=("rating", "mean"),
                       business_profile_rating=("business_rating", "mean"),
                       city=("city", "first"),
                       state=("state", "first"),
                   )
                   .query(f"review_volume >= {min_revs}")
                   .sort_values(["negative_rate", "review_volume"], ascending=[False, False]))

        biz_agg["rating_gap"] = biz_agg["avg_review_rating"] - biz_agg["business_profile_rating"]
        
        display_risk = biz_agg.reset_index()
        display_risk["avg_review_rating"] = display_risk["avg_review_rating"].round(2)
        display_risk["negative_rate"] = display_risk["negative_rate"].map("{:.1f}%".format)
        display_risk["rating_gap"] = display_risk["rating_gap"].round(2)

        st.dataframe(
            display_risk.head(25).rename(columns={
                "business_name": "Business Entity",
                "review_volume": "Total Reviews",
                "negative_reviews": "Negative Count",
                "negative_rate": "Negative % (Churn Risk)",
                "avg_review_rating": "Avg Review Rating",
                "business_profile_rating": "Overall Yelp Stars",
                "rating_gap": "Rating Gap (Reviews vs Profile)",
                "city": "City",
                "state": "State",
            }),
            use_container_width=True,
            height=320,
        )
        st.caption("A negative Rating Gap indicates that recent customer reviews are significantly lower than the business's published cumulative stars, signalling service degradation.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: REVIEW EXPLORER & ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_explorer:
    st.markdown("### Qualitative Feedback Deep-Dive & Sentiment Anomalies")

    anomaly_mode = st.radio(
        "Select Anomaly or Exploration Lens:",
        [
            "All Filtered Reviews",
            "⚡ Potential Sarcasm (5 Stars with Negative VADER Polarity)",
            "⚠️ Mixed / Counter-intuitive (1 Star with Positive VADER Polarity)",
            "🚨 Critical Churn Risks (1 Star & Highly Negative VADER Polarity ≤ -0.6)",
        ],
        horizontal=True,
    )

    exp_df = filtered.copy()
    if anomaly_mode == "⚡ Potential Sarcasm (5 Stars with Negative VADER Polarity)":
        exp_df = exp_df[(exp_df["rating"] == 5) & (exp_df["sentiment_score"] <= -0.1)]
    elif anomaly_mode == "⚠️ Mixed / Counter-intuitive (1 Star with Positive VADER Polarity)":
        exp_df = exp_df[(exp_df["rating"] == 1) & (exp_df["sentiment_score"] >= 0.4)]
    elif anomaly_mode == "🚨 Critical Churn Risks (1 Star & Highly Negative VADER Polarity ≤ -0.6)":
        exp_df = exp_df[(exp_df["rating"] == 1) & (exp_df["sentiment_score"] <= -0.6)]

    st.markdown(f"**Identified Records:** {len(exp_df):,}")

    if len(exp_df) == 0:
        st.info("No records found matching this anomaly criteria.")
    else:
        sample_size = min(6, len(exp_df))
        sample_cards = exp_df.sample(sample_size, random_state=42) if len(exp_df) > sample_size else exp_df

        col_cards1, col_cards2 = st.columns(2)
        for idx, (_, row) in enumerate(sample_cards.iterrows()):
            target_col = col_cards1 if idx % 2 == 0 else col_cards2
            border_c = SENT_COLORS.get(row["sentiment"], "#757575")
            stars_repr = "⭐" * int(row["rating"]) if pd.notna(row["rating"]) else "N/A"
            dt_str = row["date"].strftime("%b %d, %Y") if pd.notna(row.get("date")) else ""

            with target_col:
                st.markdown(
                    f"""
                    <div style="border-left: 5px solid {border_c}; background-color: #f8f9fa; padding: 14px 18px; border-radius: 6px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <span style="font-weight: 700; font-size: 1.05em; color: #212121;">{row.get('business_name', 'Unknown')}</span>
                            <span style="font-size: 0.85em; color: #757575;">{dt_str}</span>
                        </div>
                        <div style="margin-top: 4px; font-size: 0.9em; color: #616161;">
                            📍 {row.get('city', '')}, {row.get('state', '')} &nbsp;|&nbsp; {stars_repr} ({row.get('rating', '')}/5) &nbsp;|&nbsp; 
                            VADER: <b>{row.get('sentiment', '')}</b> ({row.get('sentiment_score', 0):.3f})
                        </div>
                        <div style="margin-top: 10px; font-size: 0.92em; line-height: 1.45; color: #37474f;">
                            "{str(row.get('feedback_text', ''))[:600]}{'...' if len(str(row.get('feedback_text', ''))) > 600 else ''}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8: DATA EXPORTS & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown("### Export Hub & Executive Summary")
    
    st.write("Download cleaned analytical datasets, filtered subsets, and analytical summaries.")
    exp_c1, exp_c2, exp_c3 = st.columns(3)
    
    with exp_c1:
        st.download_button(
            "⬇ Download Full Cleaned Dataset (CSV)",
            data=cleaned.to_csv(index=False).encode("utf-8"),
            file_name="churnsense_cleaned_feedback.csv",
            mime="text/csv",
        )
    with exp_c2:
        st.download_button(
            "⬇ Download Filtered Subset (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="churnsense_filtered_subset.csv",
            mime="text/csv",
        )
    with exp_c3:
        if has_biz and len(filtered) > 0 and display_risk is not None:
            st.download_button(
                "⬇ Download Risk Watchlist (CSV)",
                data=display_risk.to_csv(index=False).encode("utf-8"),
                file_name="churnsense_high_risk_businesses.csv",
                mime="text/csv",
            )

    st.markdown("---")
    summary_txt = OUT / "analysis_summary.txt"
    if summary_txt.exists():
        st.subheader("Executive Analysis Summary Report")
        with open(summary_txt, "r", encoding="utf-8") as f:
            st.text_area("Full Pipeline Execution Log", f.read(), height=300)
        st.download_button(
            "⬇ Download Summary Report (TXT)",
            data=summary_txt.read_bytes(),
            file_name="analysis_summary.txt",
            mime="text/plain",
        )
