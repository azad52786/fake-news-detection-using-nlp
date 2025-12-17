# experiments/data/prepare_dataset.py

import pandas as pd
from pathlib import Path

# Paths  
DATA_DIR = Path(__file__).parent
FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"
OUTPUT_PATH = DATA_DIR / "data.csv"

def main():
    print("Loading datasets...")

    if not FAKE_PATH.exists():
        raise FileNotFoundError("Fake.csv not found in experiments/data/")
    if not TRUE_PATH.exists():
        raise FileNotFoundError("True.csv not found in experiments/data/")

    # Load CSVs
    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    print("Fake samples:", len(fake_df))
    print("True samples:", len(true_df))

    # Add labels
    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"

    # Standardize column names
    fake_df = fake_df.rename(columns={"text": "content"})
    true_df = true_df.rename(columns={"text": "content"})

    # Keep only required columns
    fake_df = fake_df[["title", "content", "label"]]
    true_df = true_df[["title", "content", "label"]]

    # Merge datasets
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)

    # Drop missing values
    combined_df.dropna(subset=["content", "label"], inplace=True)

    # Shuffle dataset
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("Total samples after merge:", len(combined_df))
    print("Label distribution:")
    print(combined_df["label"].value_counts())

    # Save final dataset
    combined_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n✅ Dataset prepared successfully!")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
