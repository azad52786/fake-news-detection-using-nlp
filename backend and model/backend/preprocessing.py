# backend/preprocessing.py
"""
Deterministic preprocessing functions used identically during training and inference.
Keep functions pure and unit-testable.

Functions:
- clean_text(text) -> str
- remove_non_alphanumeric(text) -> str
- tokenize(text) -> List[str]
- remove_stopwords(tokens) -> List[str]
- lemmatize(tokens) -> List[str]
- normalize_for_vectorizer(tokens) -> str
- preprocess_for_vectorizer(text) -> str   # top-level: string in -> cleaned string out
"""

import re
from typing import List
import os

# Use NLTK for lemmatization and stopwords (lightweight)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:
    raise RuntimeError(
        "NLTK not available. Install requirements and run `python -m nltk.downloader wordnet stopwords punkt`"
    )

# Ensure necessary NLTK corpora are available; if not, try download (best-effort)
def _ensure_nltk():
    import warnings
    warnings.filterwarnings('ignore')
    
    resources_needed = [
        ('corpora', 'stopwords'),
        ('corpora', 'wordnet'),
        ('tokenizers', 'punkt')
    ]
    
    for category, resource in resources_needed:
        try:
            nltk.data.find(f"{category}/{resource}")
        except LookupError:
            try:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True, raise_on_error=False)
            except Exception as e:
                print(f"Warning: Could not download {resource}: {e}")
    
    # Initialize after downloads
    try:
        _ = stopwords.words("english")
    except Exception:
        pass
    
    warnings.filterwarnings('default')


_ensure_nltk()
_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs and emails
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace newlines and tabs with space
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Collapse multiple spaces to one
    text = re.sub(r"\s+", " ", text)

    text = text.strip()
    return text


def remove_non_alphanumeric(text: str) -> str:
    # Keep letters and spaces only. Remove digits and punctuation.
    # For languages needing accents, modify regex accordingly.
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    # simple whitespace + punctuation aware tokenizer
    # Use regex to capture words (equivalent to \w+ but only letters)
    tokens = re.findall(r"[a-zA-Z]+", text)
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _STOPWORDS]


def lemmatize(tokens: List[str]) -> List[str]:
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def normalize_for_vectorizer(tokens: List[str]) -> str:
    # join tokens as space-separated string expected by sklearn TfidfVectorizer
    return " ".join(tokens)


def preprocess_for_vectorizer(text: str) -> str:
    """
    Full pipeline: text -> cleaned string suitable for TF-IDF vectorizer input.
    Deterministic and idempotent.
    """
    text = clean_text(text)
    text = remove_non_alphanumeric(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return normalize_for_vectorizer(tokens)
