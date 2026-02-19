"""Tests for snapshot storage."""

import json
import pytest
from pathlib import Path

from prompt_drift.models import Snapshot, SnapshotEntry, TestInput
from prompt_drift.store import (
    save_snapshot,
    load_snapshot,
    list_snapshots,
    delete_snapshot,
    snapshot_exists,
)


@pytest.fixture
def store_dir(tmp_path):
    """Provide a temporary store directory."""
    d = tmp_path / "test_store"
    d.mkdir()
    return str(d)


@pytest.fixture
def sample_snapshot():
    """Create a sample snapshot for testing."""
    return Snapshot(
        name="test-snap",
        prompt_template="You are a helpful assistant.",
        entries=[
            SnapshotEntry(
                input=TestInput(role="user", content="Hello"),
                output="Hi there! How can I help you?",
                model="gpt-4o",
                latency_ms=150.5,
            ),
            SnapshotEntry(
                input=TestInput(role="user", content="What is 2+2?"),
                output="2 + 2 equals 4.",
                model="gpt-4o",
                latency_ms=120.3,
            ),
        ],
        model="gpt-4o",
        metadata={"version": "1.0"},
    )


class TestSaveAndLoad:
    def test_save_creates_file(self, sample_snapshot, store_dir):
        path = save_snapshot(sample_snapshot, store_dir)
        assert path.exists()
        assert path.suffix == ".json"

    def test_roundtrip(self, sample_snapshot, store_dir):
        save_snapshot(sample_snapshot, store_dir)
        loaded = load_snapshot(sample_snapshot.name, store_dir)

        assert loaded.name == sample_snapshot.name
        assert loaded.prompt_template == sample_snapshot.prompt_template
        assert loaded.model == sample_snapshot.model
        assert len(loaded.entries) == len(sample_snapshot.entries)
        assert loaded.entries[0].output == sample_snapshot.entries[0].output
        assert loaded.entries[1].input.content == "What is 2+2?"
        assert loaded.metadata == {"version": "1.0"}

    def test_load_nonexistent_raises(self, store_dir):
        with pytest.raises(FileNotFoundError):
            load_snapshot("nonexistent", store_dir)

    def test_saved_file_is_valid_json(self, sample_snapshot, store_dir):
        path = save_snapshot(sample_snapshot, store_dir)
        data = json.loads(path.read_text())
        assert data["name"] == "test-snap"
        assert len(data["entries"]) == 2

    def test_overwrite_snapshot(self, store_dir):
        snap1 = Snapshot(
            name="overwrite-test",
            prompt_template="v1",
            entries=[],
            model="gpt-4o",
        )
        snap2 = Snapshot(
            name="overwrite-test",
            prompt_template="v2",
            entries=[],
            model="gpt-4o",
        )
        save_snapshot(snap1, store_dir)
        save_snapshot(snap2, store_dir)

        loaded = load_snapshot("overwrite-test", store_dir)
        assert loaded.prompt_template == "v2"


class TestListSnapshots:
    def test_empty_store(self, store_dir):
        assert list_snapshots(store_dir) == []

    def test_lists_saved(self, sample_snapshot, store_dir):
        save_snapshot(sample_snapshot, store_dir)
        names = list_snapshots(store_dir)
        assert sample_snapshot.name in names

    def test_lists_multiple(self, store_dir):
        for name in ["alpha", "beta", "gamma"]:
            snap = Snapshot(name=name, prompt_template="t", entries=[], model="m")
            save_snapshot(snap, store_dir)

        names = list_snapshots(store_dir)
        assert set(names) == {"alpha", "beta", "gamma"}


class TestDeleteSnapshot:
    def test_delete_existing(self, sample_snapshot, store_dir):
        save_snapshot(sample_snapshot, store_dir)
        assert delete_snapshot(sample_snapshot.name, store_dir) is True
        assert not snapshot_exists(sample_snapshot.name, store_dir)

    def test_delete_nonexistent(self, store_dir):
        assert delete_snapshot("nope", store_dir) is False


class TestSnapshotExists:
    def test_exists(self, sample_snapshot, store_dir):
        save_snapshot(sample_snapshot, store_dir)
        assert snapshot_exists(sample_snapshot.name, store_dir) is True

    def test_not_exists(self, store_dir):
        assert snapshot_exists("missing", store_dir) is False
