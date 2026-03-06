import csv
import os
import sys
import torch
import pandas as pd  # pip install pandas
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# ---------------------------------------------------------------------------
# Model Configuration  (CrudeBERT – Captain-1337/CrudeBERT)
# Label order follows HuggingFace reference:
#   index 0 → positive, index 1 → negative, index 2 → neutral
# ---------------------------------------------------------------------------
CONFIG_PATH = "/Users/yuepan/Desktop/campbell-B/model/crudebert/crude_bert_config.json"
MODEL_PATH  = "/Users/yuepan/Desktop/campbell-B/model/crudebert/crude_bert_model.bin"
CLASS_NAMES = ["positive", "negative", "neutral"]

# Load model once at import time
config = AutoConfig.from_pretrained(CONFIG_PATH)
model = AutoModelForSequenceClassification.from_config(config)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
state_dict.pop("bert.embeddings.position_ids", None)
model.load_state_dict(state_dict, strict=False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# ---------------------------------------------------------------------------
# Core prediction helpers
# ---------------------------------------------------------------------------
def predict_sentiment(text: str):
    """
    Predict sentiment for a single text string.
    Returns (label, confidence).
    """
    text = "" if pd.isna(text) else str(text).strip()
    if not text:
        return "neutral", 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    pred_label_id = int(torch.argmax(probs).item())
    predicted_label = CLASS_NAMES[pred_label_id]
    confidence = float(probs[pred_label_id].item())
    return predicted_label, confidence


def predict_to_df(texts, model, tokenizer):
    """
    Batch-predict on a list of texts.
    Returns a DataFrame with columns [Headline, sentiment, confidence].
    """
    data = []
    model.eval()

    for text in texts:
        label, conf = predict_sentiment(text)
        data.append([text, label, conf])

    return pd.DataFrame(data, columns=["Headline", "sentiment", "confidence"])


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------
def resolve_text_column(df: pd.DataFrame) -> str:
    """Return the first column name that looks like a headline / title / text."""
    candidates = ["Headline", "headline", "title", "Title", "text", "content"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "CSV no columns: "
        + ", ".join(candidates)
    )


def process_csv_file(input_csv: str, output_csv: str = None):
    """
    Read *input_csv*, run CrudeBERT sentiment analysis on the title/headline
    column, and write the result to *output_csv* with added ``sentiment``
    and ``confidence`` columns.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    text_col = resolve_text_column(df)
    print(f"Processing {len(df)} rows from {input_csv}  (text column: '{text_col}')")

    sentiments = []
    confidences = []

    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(df)}")
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        label, conf = predict_sentiment(text)
        sentiments.append(label)
        confidences.append(conf)

    df["sentiment"] = sentiments
    df["confidence"] = confidences

    # Generate output path if not provided
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv_results")
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(output_dir, f"{base}_crudebert_{timestamp}.csv")

    df.to_csv(output_csv, index=False, encoding="utf-8")

    # Summary
    print(f"\nSentiment Distribution:")
    for s in CLASS_NAMES:
        count = (df["sentiment"] == s).sum()
        pct = 100 * count / len(df) if len(df) else 0
        print(f"  {s}: {count} ({pct:.1f}%)")
    print(f"Average Confidence: {df['confidence'].mean():.4f}")
    print(f"\nOutput saved to: {output_csv}")
    return output_csv


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python Crudebert_sentiment_analysis_csv.py <input_csv> [output_csv]")
        print("\nExample:")
        print("  python Crudebert_sentiment_analysis_csv.py test_data/oil_news_2024_COMBINED.csv")
        print("  python Crudebert_sentiment_analysis_csv.py test_data/oil_news_2024_COMBINED.csv results.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        output_path = process_csv_file(input_csv, output_csv)
        print(f"\n✓ Done!  →  {output_path}")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

