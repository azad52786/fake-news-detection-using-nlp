# experiments/train_baseline.py
"""
Train a baseline TF-IDF + LogisticRegression model for Fake News detection.
Saves tfidf.pkl, model.pkl, metadata.json to backend/model_artifacts/
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import sys

# Ensure backend preprocessing module is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "backend"))
from preprocessing import preprocess_for_vectorizer

def evaluate_model(model, X, y, pos_label='REAL'):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, pos_label=pos_label)
    rec = recall_score(y, y_pred, pos_label=pos_label)
    f1 = f1_score(y, y_pred, pos_label=pos_label)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "report": classification_report(y, y_pred), "confusion_matrix": confusion_matrix(y, y_pred).tolist()}

def main(args):
    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset:", data_path)
    df = pd.read_csv(data_path)
    # Normalize columns - adjust if necessary
    if 'text' in df.columns and 'label' in df.columns:
        df = df.rename(columns={'text': 'content'})
    # Drop blanks and NAs
    df = df.dropna(subset=['content', 'label'])

    # Build combined text; handle cases where title column is missing (e.g., merged text/label-only files)
    if 'title' in df.columns:
        title_series = df['title'].fillna('')
    else:
        title_series = pd.Series([''] * len(df), index=df.index)
    df['combined'] = title_series.astype(str) + ' ' + df['content'].astype(str)
    df['label'] = df['label'].astype(str).str.upper().apply(lambda x: 'REAL' if 'REAL' in x else 'FAKE')

    X = df['combined'].values
    y = df['label'].values

    # Splits
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)
    val_relative = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_relative, stratify=y_train_val, random_state=args.random_state)

    print("Sizes -> train:", len(X_train), "val:", len(X_val), "test:", len(X_test))

    # Preprocess
    print("Preprocessing texts ...")
    X_train_p = [preprocess_for_vectorizer(t) for t in X_train]
    X_val_p = [preprocess_for_vectorizer(t) for t in X_val]
    X_test_p = [preprocess_for_vectorizer(t) for t in X_test]

    # Vectorize
    print("Fitting TF-IDF ...")
    tfidf = TfidfVectorizer(max_features=args.max_features, ngram_range=tuple(args.ngram_range))
    X_train_vec = tfidf.fit_transform(X_train_p)
    X_val_vec = tfidf.transform(X_val_p)
    X_test_vec = tfidf.transform(X_test_p)

    # Train model
    print("Training model ...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=args.random_state)
    model.fit(X_train_vec, y_train)

    # Evaluate
    print("Evaluating on validation set ...")
    val_metrics = evaluate_model(model, X_val_vec, y_val)
    print("Validation metrics:", val_metrics)

    print("Evaluating on test set ...")
    test_metrics = evaluate_model(model, X_test_vec, y_test)
    print("Test metrics:", test_metrics["report"])
    print("Test confusion matrix:", test_metrics["confusion_matrix"])

    # Save artifacts
    tfidf_path = out_dir / "tfidf.pkl"
    model_path = out_dir / "model.pkl"
    meta_path = out_dir / "metadata.json"

    joblib.dump(tfidf, tfidf_path)
    joblib.dump(model, model_path)

    metadata = {
        "model_version": args.model_version,
        "trained_on": str(data_path),
        "val_metrics": val_metrics,
        "test_metrics": {k: float(v) if isinstance(v, (np.floating, np.float64)) else v for k,v in test_metrics.items() if k in ("accuracy","precision","recall","f1")},
        "max_features": args.max_features,
        "ngram_range": args.ngram_range
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Saved artifacts to {out_dir}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "backend" / "model_artifacts"), help="Output directory for artifacts")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--ngram-range", nargs=2, type=int, default=(1,2), help="Two ints: min_n max_n")
    parser.add_argument("--model-version", type=str, default="baseline_v0.1")
    args = parser.parse_args()
    # ensure ngram_range is tuple of ints
    args.ngram_range = (int(args.ngram_range[0]), int(args.ngram_range[1]))
    main(args)
