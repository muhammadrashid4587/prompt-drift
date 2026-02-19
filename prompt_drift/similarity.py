"""Similarity metrics for comparing prompt outputs.

All functions return a float in [0, 1] where 1 means identical.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Levenshtein ratio
# ---------------------------------------------------------------------------

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise.
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def levenshtein_ratio(text_a: str, text_b: str) -> float:
    """Return normalized Levenshtein similarity in [0, 1].

    1.0 means the strings are identical.

    Edge cases:
        - Two empty strings return 1.0 because they are identical
          (edit distance is 0, and identical strings are maximally similar).
        - One empty and one non-empty string returns 0.0 because every
          character must be inserted/deleted.

    This is consistent with ``bleu_score``, which also returns 1.0 for
    two identical empty strings (caught by the ``reference == candidate``
    fast-path) and 0.0 when only one side is empty.
    """
    # Two identical strings (including both empty) are maximally similar.
    if text_a == text_b:
        return 1.0
    max_len = max(len(text_a), len(text_b))
    if max_len == 0:
        # Both empty -- already handled by the equality check above, but
        # kept as a defensive guard.
        return 1.0
    distance = _levenshtein_distance(text_a, text_b)
    return 1.0 - (distance / max_len)


# ---------------------------------------------------------------------------
# BLEU score (self-contained, no nltk dependency)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _modified_precision(reference_tokens: List[str], candidate_tokens: List[str], n: int) -> float:
    """Compute modified n-gram precision (clipped counts)."""
    candidate_ngrams = _ngrams(candidate_tokens, n)
    reference_ngrams = _ngrams(reference_tokens, n)

    if not candidate_ngrams:
        return 0.0

    ref_counts: Counter = Counter(reference_ngrams)
    cand_counts: Counter = Counter(candidate_ngrams)

    clipped_count = 0
    for ngram, count in cand_counts.items():
        clipped_count += min(count, ref_counts.get(ngram, 0))

    return clipped_count / len(candidate_ngrams)


def bleu_score(
    reference: str,
    candidate: str,
    max_n: int = 4,
    weights: Optional[List[float]] = None,
) -> float:
    """Compute a BLEU-like score between reference and candidate texts.

    Returns a float in [0, 1] where 1 means perfect n-gram overlap.
    This is a self-contained implementation with no nltk dependency.

    Edge cases:
        - Two identical strings (including both empty) return 1.0.
        - One empty and one non-empty string returns 0.0 because no
          tokens can be extracted from the empty side.
    """
    if reference == candidate:
        return 1.0

    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)

    if not ref_tokens or not cand_tokens:
        return 0.0

    if weights is None:
        weights = [1.0 / max_n] * max_n

    # Compute precisions for each n-gram order.
    log_precisions = []
    for n in range(1, max_n + 1):
        p = _modified_precision(ref_tokens, cand_tokens, n)
        if p == 0:
            return 0.0  # If any precision is 0, BLEU is 0
        log_precisions.append(math.log(p))

    # Brevity penalty
    bp = 1.0
    if len(cand_tokens) < len(ref_tokens):
        bp = math.exp(1.0 - len(ref_tokens) / len(cand_tokens))

    # Weighted geometric mean of precisions
    log_avg = sum(w * lp for w, lp in zip(weights, log_precisions))
    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# Cosine similarity (TF-IDF based, with sklearn fallback)
# ---------------------------------------------------------------------------

def _build_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term-frequency vector."""
    counts: Counter = Counter(tokens)
    total = len(tokens)
    return {t: c / total for t, c in counts.items()}


def _cosine_sim_vectors(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors (dicts)."""
    all_keys = set(vec_a.keys()) | set(vec_b.keys())
    dot = sum(vec_a.get(k, 0.0) * vec_b.get(k, 0.0) for k in all_keys)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _cosine_sklearn(text_a: str, text_b: str) -> float:
    """Cosine similarity using sklearn TF-IDF vectoriser."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    vectoriser = TfidfVectorizer()
    tfidf_matrix = vectoriser.fit_transform([text_a, text_b])
    sim = sk_cosine(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(sim[0][0])


def _cosine_pure_python(text_a: str, text_b: str) -> float:
    """Pure-Python TF-IDF cosine similarity (fallback)."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    # Simple TF-based similarity (approximate TF-IDF without corpus IDF).
    # For a two-document comparison, IDF is less meaningful, so TF suffices.
    all_tokens = set(tokens_a) | set(tokens_b)

    # IDF across the two-document "corpus"
    doc_a_set = set(tokens_a)
    doc_b_set = set(tokens_b)
    idf: Dict[str, float] = {}
    for token in all_tokens:
        df = int(token in doc_a_set) + int(token in doc_b_set)
        idf[token] = math.log(2.0 / df) + 1.0  # smoothed IDF

    tf_a = _build_tf(tokens_a)
    tf_b = _build_tf(tokens_b)

    tfidf_a = {t: tf_a.get(t, 0.0) * idf.get(t, 1.0) for t in all_tokens}
    tfidf_b = {t: tf_b.get(t, 0.0) * idf.get(t, 1.0) for t in all_tokens}

    return _cosine_sim_vectors(tfidf_a, tfidf_b)


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute TF-IDF cosine similarity in [0, 1].

    Uses sklearn if available, otherwise falls back to a pure-Python
    implementation.
    """
    if text_a == text_b:
        return 1.0
    if not text_a.strip() or not text_b.strip():
        return 0.0

    try:
        return _cosine_sklearn(text_a, text_b)
    except ImportError:
        return _cosine_pure_python(text_a, text_b)


# ---------------------------------------------------------------------------
# Composite drift score
# ---------------------------------------------------------------------------

def composite_drift_score(
    text_a: str,
    text_b: str,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], float]:
    """Compute all similarity metrics and a weighted composite score.

    Args:
        text_a: First text (baseline output).
        text_b: Second text (candidate output).
        weights: Optional weights for each metric. Defaults to equal weights.

    Returns:
        Tuple of (individual_scores dict, composite_score float).
        All values are in [0, 1] where 1 = identical.
    """
    if weights is None:
        weights = {
            "levenshtein": 0.3,
            "bleu": 0.3,
            "cosine": 0.4,
        }

    scores = {
        "levenshtein": levenshtein_ratio(text_a, text_b),
        "bleu": bleu_score(text_a, text_b),
        "cosine": cosine_similarity(text_a, text_b),
    }

    total_weight = sum(weights.values())
    if total_weight == 0:
        return scores, 0.0
    composite = sum(weights.get(k, 0.0) * scores[k] for k in scores) / total_weight

    return scores, composite
