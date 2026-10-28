# ChurnSense — Multi-Channel Customer Feedback EDA Pipeline

**Project title:** An End-to-End EDA Pipeline for Uncovering Structural Patterns in Multi-Channel Customer Feedback Data

This starter project explores customer feedback from multiple channels (reviews, surveys, support chats, and social media) to uncover recurring themes, sentiment patterns, channel differences, and potential service issues.

> Note: This project title describes a customer-feedback EDA project, which is different from the banking customer churn workflow. The ZIP is organized around the exact title provided.

## Features
- Load a CSV dataset
- Clean column names, missing values, and duplicates
- Standardize feedback channels and text fields
- Exploratory analysis: feedback volume by channel, ratings, sentiment distribution, and frequent terms
- Basic sentiment scoring using VADER when available
- Export cleaned data and charts
- Streamlit dashboard

## Quick start
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

## Input data
Place your CSV at `data/customer_feedback.csv`. Expected columns (flexible; the app tries to recognize common variants):
- `feedback_id`
- `date`
- `channel` (e.g., Survey, Email, App Review, Support Chat, Social Media)
- `customer_id` (optional)
- `rating` (optional numeric score)
- `feedback_text`

A small synthetic sample is included so the app runs immediately. Replace it with your own authorized dataset before drawing conclusions.

## Outputs
- `outputs/cleaned_feedback.csv`
- `outputs/feedback_volume_by_channel.png`
- `outputs/rating_by_channel.png` (when ratings are present)

## Limitations
The included sample is synthetic and is only for demonstrating the pipeline. Sentiment scores are approximate and should not be treated as ground truth. Avoid placing personally identifiable or confidential customer information in public repositories.
