# Spam Classifier

A Streamlit app that classifies text messages as spam or ham (not spam).

## Features
- **Single message check**: paste a message and get an instant spam/not-spam prediction with confidence score.
- **Batch CSV analysis**: upload a CSV of messages and get a full report:
  - Spam vs. ham breakdown with a pie chart
  - Top keywords found in messages flagged as spam
  - If your CSV also has a label column (e.g. spam/ham, 1/0), you'll also get accuracy, precision, recall, F1 score, and a confusion matrix
  - Full results table with predictions + spam probability, downloadable as CSV

The app auto-detects common column names for messages (`message`, `text`, `v2`, `sms`, `content`, `body`...) and labels (`label`, `v1`, `category`, `class`...). If it can't detect them, you can pick the right columns manually from dropdowns.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The model is trained automatically on startup using `spam.csv` (the classic SMS Spam Collection dataset bundled in this folder).

## Files
- `app.py` — the Streamlit app
- `spam.csv` — training dataset used to build the classifier
- `requirements.txt` — Python dependencies
