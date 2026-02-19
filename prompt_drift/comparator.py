"""Compare two snapshots and produce a drift report."""

from __future__ import annotations

import logging
from typing import Optional

from .models import (
    ComparisonEntry,
    DriftReport,
    SimilarityScores,
    Snapshot,
)
from .similarity import composite_drift_score

logger = logging.getLogger(__name__)


def compare_snapshots(
    baseline: Snapshot,
    candidate: Snapshot,
    threshold: float = 0.2,
    weights: Optional[dict] = None,
) -> DriftReport:
    """Compare two snapshots entry-by-entry and produce a DriftReport.

    The comparison is done by matching entries in order. If the snapshots
    have different numbers of entries, only the overlapping prefix is compared
    and a warning is logged.

    Args:
        baseline: The reference snapshot.
        candidate: The new snapshot to compare against baseline.
        threshold: Drift scores above this are counted as regressions.
        weights: Optional metric weights for composite scoring.

    Returns:
        A DriftReport with per-entry and aggregate results.
    """
    entries: list[ComparisonEntry] = []
    baseline_count = len(baseline.entries)
    candidate_count = len(candidate.entries)
    n = min(baseline_count, candidate_count)

    if baseline_count != candidate_count:
        logger.warning(
            "Snapshot entry count mismatch: baseline '%s' has %d entries, "
            "candidate '%s' has %d entries. Only the first %d entries will "
            "be compared.",
            baseline.name,
            baseline_count,
            candidate.name,
            candidate_count,
            n,
        )

    for i in range(n):
        base_entry = baseline.entries[i]
        cand_entry = candidate.entries[i]

        scores_dict, composite = composite_drift_score(
            base_entry.output,
            cand_entry.output,
            weights=weights,
        )

        # Drift = 1 - similarity (higher drift = more different)
        drift = 1.0 - composite

        similarity = SimilarityScores(
            levenshtein=scores_dict["levenshtein"],
            bleu=scores_dict["bleu"],
            cosine=scores_dict["cosine"],
            composite=composite,
        )

        entries.append(
            ComparisonEntry(
                input=base_entry.input,
                output_before=base_entry.output,
                output_after=cand_entry.output,
                similarity=similarity,
                drift_score=round(drift, 6),
            )
        )

    drift_scores = [e.drift_score for e in entries]

    if drift_scores:
        mean_drift = sum(drift_scores) / len(drift_scores)
        max_drift = max(drift_scores)
        min_drift = min(drift_scores)
        regression_count = sum(1 for d in drift_scores if d > threshold)
    else:
        mean_drift = 0.0
        max_drift = 0.0
        min_drift = 0.0
        regression_count = 0

    metadata: dict = {}
    if baseline_count != candidate_count:
        metadata["entry_count_mismatch"] = {
            "baseline_entries": baseline_count,
            "candidate_entries": candidate_count,
            "compared_entries": n,
            "skipped_entries": abs(baseline_count - candidate_count),
        }

    return DriftReport(
        baseline_name=baseline.name,
        candidate_name=candidate.name,
        entries=entries,
        mean_drift=round(mean_drift, 6),
        max_drift=round(max_drift, 6),
        min_drift=round(min_drift, 6),
        regression_count=regression_count,
        threshold=threshold,
        metadata=metadata,
    )
