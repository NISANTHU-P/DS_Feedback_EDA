import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
csv = ROOT / "data" / "yelp_sentiment_master_dataset.csv"
print("File exists:", csv.exists(), "| Size:", csv.stat().st_size, "bytes")

df = pd.read_csv(csv, nrows=5)
print("\nShape (5 rows):", df.shape)
print("Columns:", list(df.columns))
print("\nDtypes:")
print(df.dtypes)
print("\nSample rows:")
print(df.to_string())

# Full shape
full = pd.read_csv(csv)
print("\n\nFULL SHAPE:", full.shape)
print("\nMissing values:")
print(full.isnull().sum())
print("\nDuplicates:", full.duplicated().sum())
print("\nrating_review unique values:", sorted(full["rating_review"].dropna().unique()) if "rating_review" in full.columns else "N/A")
print("\nstars_review value counts:")
if "stars_review" in full.columns:
    print(full["stars_review"].value_counts().sort_index())
print("\nSample text:")
col = "text" if "text" in full.columns else full.columns[3]
print(full[col].iloc[0][:300])
