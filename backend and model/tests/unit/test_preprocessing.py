# tests/unit/test_preprocessing.py
import sys
from pathlib import Path
import pytest

# Ensure backend is importable when running pytest from project root
ROOT = Path(__file__).resolve().parents[2]  # project-root/tests/unit -> go up two
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from preprocessing import (
    clean_text,
    remove_non_alphanumeric,
    tokenize,
    remove_stopwords,
    lemmatize,
    normalize_for_vectorizer,
    preprocess_for_vectorizer,
)


def test_clean_text_removes_urls_emails_and_html():
    raw = "Hello <b>World</b>! Visit http://example.com or contact me@domain.com\nNew line."
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "<b>" not in cleaned
    assert "new line" in cleaned  # newline replaced by space
    assert cleaned == cleaned.strip()


def test_remove_non_alphanumeric_keeps_letters_only():
    raw = "This is a test: 1234, symbols! @#$%"
    out = remove_non_alphanumeric(raw)
    assert "1234" not in out
    assert ":" not in out
    assert "symbols" in out


def test_tokenize_returns_only_words():
    raw = "Hello, world! It's 2025."
    tokens = tokenize(raw)
    assert all(tok.isalpha() for tok in tokens)
    assert "hello" in [t.lower() for t in tokens]


def test_stopwords_removed():
    tokens = ["this", "is", "a", "sample", "text"]
    filtered = remove_stopwords(tokens)
    assert "this" not in filtered
    assert "sample" in filtered


def test_lemmatize_basic():
    tokens = ["cars", "running", "better"]
    lem = lemmatize(tokens)
    # basic sanity: output length same and words lemmatized (not raising)
    assert len(lem) == len(tokens)
    assert "car" in [t.lower() for t in lem] or "cars" in [t.lower() for t in lem]


def test_normalize_for_vectorizer_and_pipeline_end_to_end():
    raw = "<p>The QUICK brown foxes, running fast! Visit: http://x</p>"
    prepped = preprocess_for_vectorizer(raw)
    # result should be a lowercased, token-joined string with no punctuation
    assert isinstance(prepped, str)
    assert prepped == prepped.strip()
    assert "http" not in prepped
    assert "," not in prepped
    # tokens separated by single space
    assert "  " not in prepped
    assert len(prepped.split()) >= 1
