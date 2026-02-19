"""Snapshot creation: run a prompt against test inputs and capture outputs."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import Snapshot, SnapshotEntry, TestInput


# ---------------------------------------------------------------------------
# Mock LLM for --dry-run mode
# ---------------------------------------------------------------------------

def _mock_llm(prompt: str, input_content: str, model: str) -> str:
    """Deterministic mock LLM for testing and dry-run mode.

    Produces a repeatable output based on a hash of the prompt and input,
    so the same prompt+input always yields the same mock response.
    """
    seed = hashlib.sha256(f"{prompt}||{input_content}||{model}".encode()).hexdigest()[:12]
    word_bank = [
        "the", "quantum", "approach", "enables", "efficient", "processing",
        "of", "complex", "data", "structures", "through", "parallel",
        "computation", "and", "advanced", "algorithms", "that", "leverage",
        "fundamental", "principles", "to", "solve", "previously",
        "intractable", "problems", "in", "polynomial", "time", "with",
        "high", "accuracy", "and", "reliability",
    ]
    # Use the hash to select words deterministically.
    indices = [int(seed[i : i + 2], 16) % len(word_bank) for i in range(0, len(seed), 2)]
    n_words = 20 + (int(seed[:4], 16) % 30)
    words = []
    for i in range(n_words):
        idx = indices[i % len(indices)]
        words.append(word_bank[(idx + i) % len(word_bank)])
    # Capitalise first word and add a period.
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


# ---------------------------------------------------------------------------
# OpenAI-compatible LLM caller
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, input_content: str, model: str) -> str:
    """Call an OpenAI-compatible API."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for live LLM calls. "
            "Install it with: pip install openai"
        )

    client = openai.OpenAI()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": input_content},
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_snapshot(
    name: str,
    prompt_template: str,
    test_inputs: List[TestInput],
    model: str = "gpt-4o",
    dry_run: bool = False,
    llm_caller: Optional[Callable[[str, str, str], str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Snapshot:
    """Create a snapshot by running a prompt against all test inputs.

    Args:
        name: Unique name for this snapshot (e.g., "v1", "refactored-prompt").
        prompt_template: The system prompt / template text.
        test_inputs: List of TestInput messages to send.
        model: Model identifier (e.g., "gpt-4o").
        dry_run: If True, use a deterministic mock LLM instead of real API.
        llm_caller: Optional custom function(prompt, input, model) -> output.
        metadata: Optional extra metadata dict to attach.

    Returns:
        A Snapshot containing all input/output pairs.
    """
    caller = llm_caller
    if caller is None:
        caller = _mock_llm if dry_run else _call_openai

    entries: List[SnapshotEntry] = []
    for test_input in test_inputs:
        start = time.monotonic()
        output = caller(prompt_template, test_input.content, model)
        elapsed_ms = (time.monotonic() - start) * 1000.0

        entries.append(
            SnapshotEntry(
                input=test_input,
                output=output,
                model=model,
                latency_ms=round(elapsed_ms, 2),
            )
        )

    return Snapshot(
        name=name,
        prompt_template=prompt_template,
        entries=entries,
        model=model,
        metadata=metadata or {},
    )


def load_test_inputs(path: str) -> List[TestInput]:
    """Load test inputs from a JSON file.

    Expected format: a JSON array of objects with 'role' and 'content' keys.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return [TestInput(**item) for item in data]
