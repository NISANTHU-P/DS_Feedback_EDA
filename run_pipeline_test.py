"""
run_pipeline_test.py
Full pipeline execution: load -> clean -> sentiment -> EDA -> charts -> report.
Uses the Yelp dataset (data/customer_feedback.csv).
"""
import sys, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pipeline import clean_feedback, add_sentiment, get_top_words, sentiment_agreement

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SENT_COLORS = {"Positive": "#4caf50", "Neutral": "#ff9800", "Negative": "#f44336"}

# ============================================================
# 1. LOAD
# ============================================================
raw = pd.read_csv(ROOT / "data" / "customer_feedback.csv")
print("=" * 60)
print("RAW DATA")
print(f"  Shape          : {raw.shape}")
print(f"  Columns        : {list(raw.columns)}")
print(f"  Missing values :\n{raw.isnull().sum().to_string()}")
print(f"  Duplicates     : {raw.duplicated().sum()}")

# ============================================================
# 2. CLEAN
# ============================================================
print("\n" + "=" * 60)
print("CLEANING")
cleaned = clean_feedback(raw)
print(f"  Columns after  : {list(cleaned.columns)}")
print(f"\n  Numeric stats:\n{cleaned[['rating','business_rating','review_count','text_length','word_count']].describe().round(2).to_string()}")

# ============================================================
# 3. SENTIMENT
# ============================================================
print("\n" + "=" * 60)
print("SENTIMENT ANALYSIS")
final = add_sentiment(cleaned)
print(f"  VADER sentiment distribution:")
print(final["sentiment"].value_counts().to_string())
print(f"\n  Original label distribution (rating_review):")
print(final["sentiment_source"].value_counts().to_string())
print(f"\n  Avg VADER score : {final['sentiment_score'].mean():.4f}")

# ============================================================
# 4. KEY METRICS
# ============================================================
print("\n" + "=" * 60)
print("KEY METRICS")
print(f"  Total cleaned records  : {len(final):,}")
print(f"  Unique businesses      : {final['business_name'].nunique():,}")
print(f"  Unique cities          : {final['city'].nunique():,}")
print(f"  Unique states          : {final['state'].nunique():,}")
avg_r = final["rating"].mean()
med_r = final["rating"].median()
print(f"  Avg review rating      : {avg_r:.3f}/5")
print(f"  Median review rating   : {med_r:.1f}/5")
neg_pct = (final["sentiment"] == "Negative").mean() * 100
print(f"  VADER negative %       : {neg_pct:.1f}%")

print("\n  Rating distribution:")
print(final["rating"].value_counts().sort_index().to_string())

print("\n  Avg rating by VADER sentiment:")
print(final.groupby("sentiment")["rating"].mean().round(3).to_string())

print("\n  Top 10 cities by review count:")
print(final["city"].value_counts().head(10).to_string())

print("\n  Top 10 states by review count:")
print(final["state"].value_counts().head(10).to_string())

print("\n  Top 10 businesses by review count:")
print(final["business_name"].value_counts().head(10).to_string())

print("\n  Sentiment comparison (original vs VADER):")
print(sentiment_agreement(final).to_string())

# ============================================================
# 5. TOP WORDS
# ============================================================
print("\n" + "=" * 60)
print("TOP WORDS")
top_all = get_top_words(final["feedback_text"], 20)
print("  All reviews (top 20):")
print(top_all.to_string(index=False))

top_neg = get_top_words(final.loc[final["sentiment"] == "Negative", "feedback_text"], 20)
print("\n  Negative reviews (top 20):")
print(top_neg.to_string(index=False))

# ============================================================
# 6. CHARTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING CHARTS")

# 6a. Rating distribution
fig, ax = plt.subplots(figsize=(7, 4))
rc = final["rating"].value_counts().sort_index()
rc.plot(kind="bar", ax=ax, color="mediumpurple", edgecolor="white")
ax.set_title("Review Rating Distribution (Yelp)")
ax.set_xlabel("Stars (1-5)")
ax.set_ylabel("Number of Reviews")
ax.tick_params(axis="x", rotation=0)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=9)
fig.tight_layout()
p = OUT / "rating_distribution.png"
fig.savefig(p, dpi=120); plt.close(fig); print(f"  Saved: {p}")

# 6b. Original sentiment distribution
fig, ax = plt.subplots(figsize=(6, 4))
sc = final["sentiment_source"].value_counts()
sc.plot(kind="bar", ax=ax,
        color=[SENT_COLORS.get(s, "grey") for s in sc.index], edgecolor="white")
ax.set_title("Original Sentiment Distribution (rating_review)")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
for p2 in ax.patches:
    ax.annotate(f"{int(p2.get_height()):,}", (p2.get_x() + p2.get_width()/2, p2.get_height()),
                ha="center", va="bottom", fontsize=9)
fig.tight_layout()
p = OUT / "source_sentiment_distribution.png"
fig.savefig(p, dpi=120); plt.close(fig); print(f"  Saved: {p}")

# 6c. VADER sentiment distribution
fig, ax = plt.subplots(figsize=(6, 4))
vc = final["sentiment"].value_counts()
vc.plot(kind="bar", ax=ax,
        color=[SENT_COLORS.get(s, "grey") for s in vc.index], edgecolor="white")
ax.set_title("VADER Sentiment Distribution")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
for p2 in ax.patches:
    ax.annotate(f"{int(p2.get_height()):,}", (p2.get_x() + p2.get_width()/2, p2.get_height()),
                ha="center", va="bottom", fontsize=9)
fig.tight_layout()
p = OUT / "vader_sentiment_distribution.png"
fig.savefig(p, dpi=120); plt.close(fig); print(f"  Saved: {p}")

# 6d. Sentiment comparison heatmap
agree = sentiment_agreement(final)
if not agree.empty:
    fig, ax = plt.subplots(figsize=(6, 4))
    import numpy as np
    im = ax.imshow(agree.values, cmap="Blues")
    ax.set_xticks(range(len(agree.columns))); ax.set_xticklabels(agree.columns)
    ax.set_yticks(range(len(agree.index)));   ax.set_yticklabels(agree.index)
    ax.set_xlabel("VADER label")
    ax.set_ylabel("Original label")
    ax.set_title("Original vs VADER Sentiment Comparison")
    for i in range(len(agree.index)):
        for j in range(len(agree.columns)):
            ax.text(j, i, f"{agree.values[i,j]:,}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p = OUT / "sentiment_comparison.png"
    fig.savefig(p, dpi=120); plt.close(fig); print(f"  Saved: {p}")

# 6e. Rating by VADER sentiment
fig, ax = plt.subplots(figsize=(6, 4))
rbs = final.groupby("sentiment")["rating"].mean().reindex(["Positive","Neutral","Negative"])
rbs.plot(kind="bar", ax=ax,
         color=[SENT_COLORS.get(s, "grey") for s in rbs.index], edgecolor="white")
ax.set_title("Average Review Rating by VADER Sentiment")
ax.set_xlabel("VADER Sentiment")
ax.set_ylabel("Average Stars")
ax.set_ylim(0, 5)
ax.tick_params(axis="x", rotation=0)
for p2 in ax.patches:
    ax.annotate(f"{p2.get_height():.2f}", (p2.get_x() + p2.get_width()/2, p2.get_height()),
                ha="center", va="bottom", fontsize=9)
fig.tight_layout()
p = OUT / "rating_by_sentiment.png"
fig.savefig(p, dpi=120); plt.close(fig); print(f"  Saved: {p}")

# 6f. Top 15 cities
fig, ax = plt.subplots(figsize=(8, 5))
top_cities = final["city"].value_counts().head(15)
top_cities[::-1].plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Top 15 Cities by Review Volume")
ax.set_xlabel("Number of Reviews")
fig.tight_layout()
p = OUT / "top_cities.png"
fig.savefig(p, dpi=120); plt.close(fig); print(f"  Saved: {p}")

# 6g. Top 15 businesses
fig, ax = plt.subplots(figsize=(9, 5))
top_biz = final["business_name"].value_counts().head(15)
top_biz[::-1].plot(kind="barh", ax=ax, color="coral", edgecolor="white")
ax.set_title("Top 15 Businesses by Review Volume")
ax.set_xlabel("Number of Reviews")
fig.tight_layout()
p = OUT / "top_businesses.png"
fig.savefig(p, dpi=120); plt.close(fig); print(f"  Saved: {p}")

# 6h. Top 20 words (all reviews)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_all["word"][::-1], top_all["count"][::-1], color="teal")
ax.set_title("Top 20 Most Frequent Words (All Reviews)")
ax.set_xlabel("Frequency")
fig.tight_layout()
p = OUT / "top_words.png"
fig.savefig(p, dpi=120); plt.close(fig); print(f"  Saved: {p}")

# 6i. Top 20 words in negative reviews
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_neg["word"][::-1], top_neg["count"][::-1], color="#f44336")
ax.set_title("Top 20 Words in Negative Reviews")
ax.set_xlabel("Frequency")
fig.tight_layout()
p = OUT / "top_words_negative.png"
fig.savefig(p, dpi=120); plt.close(fig); print(f"  Saved: {p}")

# 6j. Sentiment by star rating (stacked bar)
sent_star = final.groupby(["rating", "sentiment"]).size().unstack(fill_value=0)
for col in ["Positive", "Neutral", "Negative"]:
    if col not in sent_star.columns:
        sent_star[col] = 0
sent_star = sent_star[["Positive", "Neutral", "Negative"]]
fig, ax = plt.subplots(figsize=(8, 4))
sent_star.plot(kind="bar", stacked=True, ax=ax,
               color=["#4caf50", "#ff9800", "#f44336"], edgecolor="white")
ax.set_title("VADER Sentiment by Star Rating")
ax.set_xlabel("Star Rating")
ax.set_ylabel("Number of Reviews")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
p = OUT / "sentiment_by_rating.png"
fig.savefig(p, dpi=120); plt.close(fig); print(f"  Saved: {p}")

# ============================================================
# 7. SAVE CLEANED CSV + ANALYSIS SUMMARY
# ============================================================
csv_out = OUT / "cleaned_feedback.csv"
final.to_csv(csv_out, index=False)
print(f"  Saved: {csv_out}")

summary_lines = [
    "ChurnSense -- Yelp Dataset Analysis Summary",
    "=" * 50,
    f"Source file        : data/raw/yelp_sentiment_master_dataset.csv",
    f"Raw records        : {len(raw):,}",
    f"Cleaned records    : {len(final):,}",
    f"Columns            : {list(final.columns)}",
    "",
    "Missing values (cleaned):",
    final.isnull().sum().to_string(),
    "",
    f"Avg review rating  : {avg_r:.3f}/5",
    f"Median rating      : {med_r:.1f}/5",
    f"VADER negative %   : {neg_pct:.1f}%",
    "",
    "Rating distribution:",
    final["rating"].value_counts().sort_index().to_string(),
    "",
    "Original sentiment distribution:",
    final["sentiment_source"].value_counts().to_string(),
    "",
    "VADER sentiment distribution:",
    final["sentiment"].value_counts().to_string(),
    "",
    "Original vs VADER comparison:",
    sentiment_agreement(final).to_string(),
    "",
    f"Unique businesses  : {final['business_name'].nunique():,}",
    f"Unique cities      : {final['city'].nunique():,}",
    f"Unique states      : {final['state'].nunique():,}",
    "",
    "Top 10 cities:",
    final["city"].value_counts().head(10).to_string(),
    "",
    "Top 10 businesses:",
    final["business_name"].value_counts().head(10).to_string(),
    "",
    "Top 20 words (all reviews):",
    top_all.to_string(index=False),
    "",
    "Top 20 words (negative reviews):",
    top_neg.to_string(index=False),
]
summary_path = OUT / "analysis_summary.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
print(f"  Saved: {summary_path}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)
