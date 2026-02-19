"""Tests for snapshot creation with mocked LLM."""

import json
import pytest
from pathlib import Path

from prompt_drift.models import TestInput
from prompt_drift.snapshot import create_snapshot, load_test_inputs, _mock_llm


class TestMockLLM:
    def test_deterministic(self):
        """Same inputs always produce the same output."""
        result1 = _mock_llm("prompt", "input", "model")
        result2 = _mock_llm("prompt", "input", "model")
        assert result1 == result2

    def test_different_prompts_give_different_outputs(self):
        result1 = _mock_llm("prompt A", "input", "model")
        result2 = _mock_llm("prompt B", "input", "model")
        assert result1 != result2

    def test_different_inputs_give_different_outputs(self):
        result1 = _mock_llm("prompt", "input A", "model")
        result2 = _mock_llm("prompt", "input B", "model")
        assert result1 != result2

    def test_returns_nonempty_string(self):
        result = _mock_llm("prompt", "input", "model")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_ends_with_period(self):
        result = _mock_llm("prompt", "input", "model")
        assert result.endswith(".")

    def test_first_word_capitalised(self):
        result = _mock_llm("prompt", "input", "model")
        assert result[0].isupper()


class TestCreateSnapshot:
    def test_dry_run_creates_snapshot(self):
        inputs = [
            TestInput(role="user", content="Explain quantum computing"),
            TestInput(role="user", content="Write a haiku"),
        ]
        snap = create_snapshot(
            name="test-v1",
            prompt_template="You are a helpful assistant.",
            test_inputs=inputs,
            model="gpt-4o",
            dry_run=True,
        )

        assert snap.name == "test-v1"
        assert snap.model == "gpt-4o"
        assert len(snap.entries) == 2
        assert snap.entries[0].input.content == "Explain quantum computing"
        assert len(snap.entries[0].output) > 0
        assert snap.entries[0].latency_ms is not None

    def test_custom_llm_caller(self):
        def custom_caller(prompt, content, model):
            return f"CUSTOM: {content}"

        inputs = [TestInput(role="user", content="test")]
        snap = create_snapshot(
            name="custom",
            prompt_template="prompt",
            test_inputs=inputs,
            llm_caller=custom_caller,
        )

        assert snap.entries[0].output == "CUSTOM: test"

    def test_dry_run_is_deterministic(self):
        inputs = [TestInput(role="user", content="hello")]
        snap1 = create_snapshot("s1", "prompt", inputs, dry_run=True)
        snap2 = create_snapshot("s2", "prompt", inputs, dry_run=True)

        assert snap1.entries[0].output == snap2.entries[0].output

    def test_different_prompt_gives_different_output(self):
        inputs = [TestInput(role="user", content="hello")]
        snap1 = create_snapshot("s1", "prompt version 1", inputs, dry_run=True)
        snap2 = create_snapshot("s2", "prompt version 2", inputs, dry_run=True)

        assert snap1.entries[0].output != snap2.entries[0].output

    def test_metadata(self):
        inputs = [TestInput(role="user", content="hello")]
        snap = create_snapshot(
            "s1", "p", inputs, dry_run=True,
            metadata={"author": "test"},
        )
        assert snap.metadata == {"author": "test"}

    def test_snapshot_has_created_at(self):
        inputs = [TestInput(role="user", content="hello")]
        snap = create_snapshot("s1", "p", inputs, dry_run=True)
        assert snap.created_at is not None
        assert len(snap.created_at) > 0


class TestLoadTestInputs:
    def test_load_valid(self, tmp_path):
        data = [
            {"role": "user", "content": "Question 1"},
            {"role": "user", "content": "Question 2"},
        ]
        path = tmp_path / "inputs.json"
        path.write_text(json.dumps(data))

        inputs = load_test_inputs(str(path))
        assert len(inputs) == 2
        assert inputs[0].content == "Question 1"
        assert inputs[1].role == "user"

    def test_load_invalid_format(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not": "an array"}')

        with pytest.raises(ValueError, match="Expected a JSON array"):
            load_test_inputs(str(path))

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_test_inputs("/nonexistent/path.json")
