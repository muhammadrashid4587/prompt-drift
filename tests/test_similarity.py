"""Unit tests for similarity metrics."""

import pytest

from prompt_drift.similarity import (
    bleu_score,
    composite_drift_score,
    cosine_similarity,
    levenshtein_ratio,
    _levenshtein_distance,
    _cosine_pure_python,
)


# ---------------------------------------------------------------------------
# Levenshtein
# ---------------------------------------------------------------------------

class TestLevenshteinDistance:
    def test_identical_strings(self):
        assert _levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert _levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert _levenshtein_distance("abc", "") == 3
        assert _levenshtein_distance("", "abc") == 3

    def test_single_char_diff(self):
        assert _levenshtein_distance("cat", "car") == 1

    def test_insertion(self):
        assert _levenshtein_distance("abc", "abcd") == 1

    def test_deletion(self):
        assert _levenshtein_distance("abcd", "abc") == 1


class TestLevenshteinRatio:
    def test_identical(self):
        assert levenshtein_ratio("hello world", "hello world") == 1.0

    def test_completely_different(self):
        ratio = levenshtein_ratio("aaa", "zzz")
        assert ratio == 0.0

    def test_empty_strings(self):
        assert levenshtein_ratio("", "") == 1.0

    def test_partial_match(self):
        ratio = levenshtein_ratio("hello", "hallo")
        assert 0.5 < ratio < 1.0

    def test_returns_float_in_range(self):
        ratio = levenshtein_ratio("test string one", "test string two")
        assert 0.0 <= ratio <= 1.0

    def test_symmetry(self):
        a, b = "quantum computing", "quantum mechanics"
        assert levenshtein_ratio(a, b) == levenshtein_ratio(b, a)


# ---------------------------------------------------------------------------
# BLEU score
# ---------------------------------------------------------------------------

class TestBleuScore:
    def test_identical(self):
        text = "the cat sat on the mat"
        assert bleu_score(text, text) == 1.0

    def test_completely_different(self):
        score = bleu_score("alpha beta gamma", "one two three")
        assert score == 0.0

    def test_partial_overlap(self):
        ref = "the quick brown fox jumps over the lazy dog"
        cand = "the fast brown fox leaps over the sleepy dog"
        score = bleu_score(ref, cand)
        assert 0.0 < score < 1.0

    def test_empty_candidate(self):
        assert bleu_score("hello world", "") == 0.0

    def test_empty_reference(self):
        assert bleu_score("", "hello world") == 0.0

    def test_returns_float_in_range(self):
        score = bleu_score(
            "Explain quantum computing simply",
            "Quantum computing uses qubits for computation",
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical(self):
        text = "machine learning is a subset of artificial intelligence"
        assert cosine_similarity(text, text) == 1.0

    def test_empty_strings(self):
        assert cosine_similarity("", "") == 0.0
        assert cosine_similarity("hello", "") == 0.0

    def test_similar_texts(self):
        a = "machine learning algorithms process data"
        b = "machine learning models analyze data"
        sim = cosine_similarity(a, b)
        assert 0.3 < sim < 1.0

    def test_different_texts(self):
        a = "the weather is sunny today"
        b = "quantum physics explains particle behavior"
        sim = cosine_similarity(a, b)
        assert sim < 0.5

    def test_returns_float_in_range(self):
        sim = cosine_similarity("hello world foo bar", "hello world baz qux")
        assert 0.0 <= sim <= 1.0


class TestCosinePurePython:
    def test_identical(self):
        text = "machine learning is great"
        assert _cosine_pure_python(text, text) > 0.99

    def test_empty(self):
        assert _cosine_pure_python("", "hello") == 0.0

    def test_similar(self):
        a = "the cat sat on the mat"
        b = "the dog sat on the mat"
        sim = _cosine_pure_python(a, b)
        assert 0.5 < sim < 1.0


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class TestCompositeDriftScore:
    def test_identical_texts(self):
        scores, composite = composite_drift_score("hello world", "hello world")
        assert composite == 1.0
        assert scores["levenshtein"] == 1.0
        assert scores["bleu"] == 1.0
        assert scores["cosine"] == 1.0

    def test_different_texts(self):
        scores, composite = composite_drift_score(
            "the quick brown fox", "completely unrelated text"
        )
        assert 0.0 <= composite <= 1.0
        for key in ("levenshtein", "bleu", "cosine"):
            assert 0.0 <= scores[key] <= 1.0

    def test_custom_weights(self):
        weights = {"levenshtein": 1.0, "bleu": 0.0, "cosine": 0.0}
        scores, composite = composite_drift_score("abc", "abd", weights=weights)
        assert abs(composite - scores["levenshtein"]) < 1e-6

    def test_partial_similarity(self):
        a = "Quantum computing uses qubits for parallel computation"
        b = "Quantum computing leverages qubits for simultaneous processing"
        scores, composite = composite_drift_score(a, b)
        assert 0.2 < composite < 1.0
