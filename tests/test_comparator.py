"""Tests for the comparison logic."""

import pytest

from prompt_drift.comparator import compare_snapshots
from prompt_drift.models import Snapshot, SnapshotEntry, TestInput


def _make_snapshot(name: str, outputs: list[str], prompt: str = "test prompt") -> Snapshot:
    """Helper to build a Snapshot with given outputs."""
    entries = [
        SnapshotEntry(
            input=TestInput(role="user", content=f"input_{i}"),
            output=output,
            model="mock",
        )
        for i, output in enumerate(outputs)
    ]
    return Snapshot(
        name=name,
        prompt_template=prompt,
        entries=entries,
        model="mock",
    )


class TestCompareSnapshots:
    def test_identical_snapshots(self):
        outputs = ["Hello, world!", "Quantum computing uses qubits.", "SOLID stands for..."]
        baseline = _make_snapshot("v1", outputs)
        candidate = _make_snapshot("v2", outputs)

        report = compare_snapshots(baseline, candidate)

        assert report.mean_drift == 0.0
        assert report.max_drift == 0.0
        assert report.regression_count == 0
        assert len(report.entries) == 3
        for entry in report.entries:
            assert entry.drift_score == 0.0
            assert entry.similarity.composite == 1.0

    def test_completely_different_outputs(self):
        baseline = _make_snapshot("v1", ["aaa bbb ccc ddd eee"])
        candidate = _make_snapshot("v2", ["xxx yyy zzz www qqq"])

        report = compare_snapshots(baseline, candidate)

        assert report.mean_drift > 0.5
        assert report.max_drift > 0.5
        assert report.regression_count >= 1

    def test_partial_drift(self):
        baseline = _make_snapshot("v1", [
            "Quantum computing uses qubits for parallel computation.",
            "A haiku about code: Bugs in the system, silent errors creep at night, debug saves the day.",
        ])
        candidate = _make_snapshot("v2", [
            "Quantum computing leverages qubits for simultaneous processing.",
            "A haiku about code: Bugs in the system, silent errors creep at night, debug saves the day.",
        ])

        report = compare_snapshots(baseline, candidate)

        # First entry drifted, second is identical
        assert report.entries[0].drift_score > 0.0
        assert report.entries[1].drift_score == 0.0
        assert report.mean_drift > 0.0

    def test_different_lengths_uses_min(self):
        baseline = _make_snapshot("v1", ["a", "b", "c"])
        candidate = _make_snapshot("v2", ["a", "b"])

        report = compare_snapshots(baseline, candidate)
        assert len(report.entries) == 2

    def test_custom_threshold(self):
        baseline = _make_snapshot("v1", ["hello world"])
        candidate = _make_snapshot("v2", ["hello earth"])

        # Very low threshold should flag minor changes as regressions
        report_low = compare_snapshots(baseline, candidate, threshold=0.01)
        report_high = compare_snapshots(baseline, candidate, threshold=0.99)

        assert report_low.regression_count >= report_high.regression_count

    def test_report_names(self):
        baseline = _make_snapshot("baseline-v1", ["x"])
        candidate = _make_snapshot("candidate-v2", ["x"])

        report = compare_snapshots(baseline, candidate)
        assert report.baseline_name == "baseline-v1"
        assert report.candidate_name == "candidate-v2"

    def test_empty_snapshots(self):
        baseline = _make_snapshot("v1", [])
        candidate = _make_snapshot("v2", [])

        report = compare_snapshots(baseline, candidate)
        assert report.mean_drift == 0.0
        assert report.regression_count == 0
        assert len(report.entries) == 0

    def test_has_regressions_property(self):
        baseline = _make_snapshot("v1", ["aaa bbb ccc"])
        candidate = _make_snapshot("v2", ["xxx yyy zzz"])

        report = compare_snapshots(baseline, candidate, threshold=0.01)
        assert report.has_regressions is True

        report2 = compare_snapshots(baseline, _make_snapshot("v3", ["aaa bbb ccc"]))
        assert report2.has_regressions is False
