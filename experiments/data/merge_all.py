import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "data_merged_text_label.csv"

SOURCES = [
    {"path": DATA_DIR / "Fake.csv", "label": "FAKE", "text_col": "text"},
    {"path": DATA_DIR / "True.csv", "label": "REAL", "text_col": "text"},
    {"path": DATA_DIR / "data.csv", "label": None, "text_col": None},
    {"path": DATA_DIR / "dataset.csv", "label": None, "text_col": None},
]


def _normalize_frame(df: pd.DataFrame, forced_label: str | None, text_col_hint: str | None) -> pd.DataFrame:
    # Pick a text column
    text_col = None
    for col in ([text_col_hint] if text_col_hint else []) + ["text", "content"]:
        if col and col in df.columns:
            text_col = col
            break
    if text_col is None:
        raise ValueError("No text/content column found")

    # If label provided externally, use it; otherwise read from column
    if forced_label:
        df = df.rename(columns={text_col: "content"})
        df["label"] = forced_label
    else:
        if "label" not in df.columns:
            raise ValueError("Label column missing and no forced label provided")
        df = df.rename(columns={text_col: "content"})
        df["label"] = df["label"].astype(str)

    # Normalize label to REAL/FAKE
    df["label"] = df["label"].str.upper().apply(lambda x: "REAL" if "REAL" in x else "FAKE")

    return df[["content", "label"]]


def main() -> None:
    frames: list[pd.DataFrame] = []

    for src in SOURCES:
        path = src["path"]
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        print(f"Loading {path}")
        df = pd.read_csv(path)
        try:
            normalized = _normalize_frame(df, src["label"], src["text_col"])
            frames.append(normalized)
        except Exception as exc:
            raise SystemExit(f"Failed to normalize {path}: {exc}")

    if not frames:
        raise SystemExit("No data loaded; nothing to merge")

    merged = pd.concat(frames, ignore_index=True)
    merged.dropna(subset=["content", "label"], inplace=True)
    merged["content"] = merged["content"].astype(str).str.strip()
    merged["label"] = merged["label"].astype(str).str.upper().apply(lambda x: "REAL" if "REAL" in x else "FAKE")

    # Remove duplicates and shuffle
    merged = merged.drop_duplicates(subset=["content", "label"]).sample(frac=1.0, random_state=42)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Merged rows: {len(merged)}")
    print("Label distribution:\n", merged["label"].value_counts())
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
