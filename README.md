# ChurnSense -- Customer Feedback EDA

**An end-to-end EDA pipeline for uncovering structural patterns in real-world
multi-channel customer feedback data.**

---

## Problem statement

Businesses receive customer feedback across many channels. Understanding
sentiment patterns, recurring complaints, and geographic or business-level
differences helps prioritise service improvements. This project builds a
reproducible EDA pipeline and interactive dashboard on top of a real Yelp
review dataset.

**Important limitation:** This dataset does not contain verified customer-churn
outcomes or reliable customer identifiers. Negative sentiment is treated as a
possible dissatisfaction signal only -- not proof that a customer left.

---

## Dataset

| Property | Value |
|---|---|
| Source | Yelp review dataset (`yelp_sentiment_master_dataset.csv`) |
| Raw records | 14,351 |
| Cleaned records | 14,350 (1 exact duplicate removed) |
| Original columns | 11 |
| Mapped columns | 12 (ChurnSense schema) |

### Source columns

| Source column | Mapped to | Notes |
|---|---|---|
| `Sl No` | `feedback_id` | Prefixed with `YLP` |
| `stars_review` | `rating` | 1-5 integer |
| `text` | `feedback_text` | Full review text |
| `date` | `date` | Parsed as datetime |
| `name` | `business_name` | Business name |
| `categories` | `categories` | Business categories |
| `city` | `city` | City |
| `state` | `state` | US state abbreviation |
| `stars_business` | `business_rating` | Overall business rating |
| `review_count` | `review_count` | Total reviews for business |
| `rating_review` | `sentiment_source` | Original sentiment label (preserved) |
| *(derived)* | `channel` | Set to `Yelp` |

---

## Architecture

```
data/raw/yelp_sentiment_master_dataset.csv   <- original source (unchanged)
        |
        v
prepare_data.py                              <- maps Yelp columns to schema
        |
        v
data/customer_feedback.csv                   <- mapped input for the app
        |
        v
src/pipeline.py                              <- clean_feedback() + add_sentiment()
        |
        v
app.py (Streamlit)                           <- interactive dashboard
outputs/                                     <- charts, cleaned CSV, summary
```

### Key modules

- `src/pipeline.py` -- `normalize_columns`, `clean_feedback`, `add_sentiment`,
  `get_top_words`, `sentiment_agreement`
- `app.py` -- Streamlit dashboard (5 sections)
- `prepare_data.py` -- one-time data preparation script
- `run_pipeline_test.py` -- standalone execution + chart generation
- `tests/test_pipeline.py` -- 28 automated tests

---

## Technologies

- Python 3.12
- pandas 2.x, numpy 1.x
- matplotlib 3.x
- vaderSentiment 3.3.2
- Streamlit 1.x

---

## Installation

```bat
py -3.12 -m pip install -r requirements.txt
```

---

## Execution

### Step 1 -- Prepare data (run once)
```bat
py -3.12 prepare_data.py
```

### Step 2 -- Run full pipeline + save charts
```bat
py -3.12 run_pipeline_test.py
```

### Step 3 -- Launch Streamlit dashboard
```bat
py -3.12 -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Run automated tests
```bat
py -3.12 -m pytest tests/test_pipeline.py -v
```

---

## Dashboard sections

| Section | Content |
|---|---|
| A. Overview | 6 KPI metrics: raw/cleaned records, avg rating, negative %, businesses, cities/states |
| B. Dataset Explorer | Filterable table (sentiment, rating, city, state) + text search |
| C. Visual Analytics | 10 charts: rating dist, sentiment dists, comparison heatmap, sentiment by rating, top cities, top states, top businesses, word frequency, negative word frequency, business rating comparison |
| D. Review Insights | 3 sample reviews per sentiment class with business details |
| E. Downloads | Cleaned CSV + analysis summary TXT |

---

## Generated outputs (`outputs/`)

| File | Description |
|---|---|
| `cleaned_feedback.csv` | Cleaned dataset with VADER scores |
| `rating_distribution.png` | Star rating distribution |
| `source_sentiment_distribution.png` | Original sentiment label distribution |
| `vader_sentiment_distribution.png` | VADER sentiment distribution |
| `sentiment_comparison.png` | Original vs VADER heatmap |
| `rating_by_sentiment.png` | Avg rating by VADER sentiment |
| `top_cities.png` | Top 15 cities by review volume |
| `top_businesses.png` | Top 15 businesses by review volume |
| `top_words.png` | Top 20 words (all reviews) |
| `top_words_negative.png` | Top 20 words (negative reviews) |
| `sentiment_by_rating.png` | Stacked sentiment by star rating |
| `analysis_summary.txt` | Full text summary of all metrics |

---

## Test results

28 tests, 28 passed (0 failed) -- Python 3.12, pytest 8.3.4

Tests cover: column normalisation, alias mapping (synthetic + Yelp schemas),
empty-text removal, duplicate removal, rating coercion, out-of-range rating
handling, date parsing, text metrics, sentiment_source preservation, metadata
column preservation, VADER sentiment, get_top_words, sentiment_agreement,
and full integration on the 14,351-record Yelp CSV.

---

## Key analysis results

| Metric | Value |
|---|---|
| Cleaned records | 14,350 |
| Avg review rating | 3.811 / 5 |
| Median review rating | 4.0 / 5 |
| VADER Positive | 12,696 (88.5%) |
| VADER Neutral | 181 (1.3%) |
| VADER Negative | 1,473 (10.3%) |
| Original Positive | 9,857 (68.7%) |
| Original Neutral | 1,795 (12.5%) |
| Original Negative | 2,698 (18.8%) |
| Unique businesses | 2,526 |
| Unique cities | 366 |
| Unique states | 16 |
| Top city | Philadelphia (2,562 reviews) |
| Top business | Luke (187 reviews) |

---

## Limitations

- **Historical data:** Reviews span multiple years; trends may not reflect
  current business performance.
- **Source label uncertainty:** The `rating_review` labels are reference labels
  derived from star ratings and may themselves contain errors.
- **No verified churn:** The dataset contains no churn outcome column and no
  reliable customer identifier. Negative sentiment is a dissatisfaction signal
  only.
- **VADER limitations:** VADER is a lexicon-based model optimised for social
  media text. It may misclassify sarcasm, domain-specific language, or very
  long reviews.
- **Synthetic backup:** The original 20-record synthetic dataset is preserved
  at `data/backup/customer_feedback_synthetic.csv`.
