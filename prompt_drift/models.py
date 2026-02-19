"""Pydantic models for prompt-drift data structures."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TestInput(BaseModel):
    """A single test input message."""

    role: str = "user"
    content: str


class SnapshotEntry(BaseModel):
    """One input/output pair within a snapshot."""

    input: TestInput
    output: str
    model: str
    latency_ms: Optional[float] = None


class Snapshot(BaseModel):
    """A complete snapshot of prompt outputs across all test inputs."""

    name: str
    prompt_template: str
    entries: List[SnapshotEntry]
    model: str
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimilarityScores(BaseModel):
    """Similarity scores between two outputs."""

    levenshtein: float = Field(ge=0.0, le=1.0)
    bleu: float = Field(ge=0.0, le=1.0)
    cosine: float = Field(ge=0.0, le=1.0)
    composite: float = Field(ge=0.0, le=1.0)


class ComparisonEntry(BaseModel):
    """Comparison result for a single test input."""

    input: TestInput
    output_before: str
    output_after: str
    similarity: SimilarityScores
    drift_score: float = Field(ge=0.0, le=1.0)


class DriftReport(BaseModel):
    """Full drift report comparing two snapshots."""

    baseline_name: str
    candidate_name: str
    entries: List[ComparisonEntry]
    mean_drift: float
    max_drift: float
    min_drift: float
    regression_count: int
    threshold: float = 0.2
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_regressions(self) -> bool:
        return self.regression_count > 0
