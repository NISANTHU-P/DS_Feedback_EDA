"""
prepare_data.py
Maps yelp_sentiment_master_dataset.csv → data/customer_feedback.csv
using the ChurnSense schema.  Run once before launching the app.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
RAW  = ROOT / "data" / "raw" / "yelp_sentiment_master_dataset.csv"
OUT  = ROOT / "data" / "customer_feedback.csv"

print("=" * 60)
print("DATA PREPARATION -- Yelp -> ChurnSense schema")
print("=" * 60)

df = pd.read_csv(RAW)
print(f"  Source rows    : {len(df):,}")
print(f"  Source columns : {list(df.columns)}")

# ── Column mapping ────────────────────────────────────────────────────────────
mapped = pd.DataFrame()
mapped["feedback_id"]      = "YLP" + df["Sl No"].astype(str).str.zfill(6)
mapped["date"]             = df["date"]
mapped["channel"]          = "Yelp"
mapped["rating"]           = df["stars_review"]
mapped["feedback_text"]    = df["text"]
mapped["sentiment_source"] = df["rating_review"]   # original label — preserved
mapped["business_name"]    = df["name"]
mapped["categories"]       = df["categories"]
mapped["city"]             = df["city"]
mapped["state"]            = df["state"]
mapped["business_rating"]  = df["stars_business"]
mapped["review_count"]     = df["review_count"]
# customer_id intentionally omitted — not available in this dataset

mapped.to_csv(OUT, index=False)
print(f"\n  Written to     : {OUT}")
print(f"  Output rows    : {len(mapped):,}")
print(f"  Output columns : {list(mapped.columns)}")
print("\nDone.")
