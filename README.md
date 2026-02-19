# prompt-drift

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-orange.svg)](https://pypi.org/project/prompt-drift/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**Detect when prompt template changes cause output regressions.**

You tweak a prompt. Outputs change. But did they get *better* or *worse*? `prompt-drift` snapshots your prompt's outputs, lets you modify the prompt, re-run, and gives you a precise drift report -- with semantic similarity scores, not just text diffs.

```
                         prompt-drift workflow

  +-----------+       +-------------+       +-------------+
  |  Prompt   | ----> |  Snapshot   | ----> |   Stored    |
  | Template  |       |   Engine    |       |  Snapshot   |
  +-----------+       +------+------+       |  (v1.json)  |
       |                     |              +------+------+
       | (edit)              |                     |
       v                     v                     |
  +-----------+       +-------------+              |
  |  Prompt   | ----> |  Snapshot   |              |
  | Template' |       |   Engine    |              |
  +-----------+       +------+------+              |
                             |                     |
                             v                     v
                      +-------------+       +------+------+
                      |   Stored    | ----> | Comparator  |
                      |  Snapshot   |       |   Engine    |
                      |  (v2.json)  |       +------+------+
                      +-------------+              |
                                                   v
                                            +------+------+
                                            |    Drift    |
                                            |   Report    |
                                            +-------------+
                                            | - Levenshtein
                                            | - BLEU score
                                            | - Cosine sim
                                            | - Composite
                                            +-------------+
```

## Why?

Prompt engineering is iterative. Every change risks breaking what already works. `prompt-drift` gives you a safety net:

- **Snapshot** your prompt's outputs across standardised test inputs
- **Compare** snapshots with multiple similarity metrics
- **Watch** a prompt file and get instant feedback on every save
- **Catch regressions** before they reach production

## Installation

```bash
pip install prompt-drift
```

For OpenAI API support:

```bash
pip install "prompt-drift[openai]"
```

For enhanced TF-IDF cosine similarity (sklearn):

```bash
pip install "prompt-drift[all]"
```

### From source

```bash
git clone https://github.com/muhammadrashid4587/prompt-drift.git
cd prompt-drift
pip install -e ".[dev]"
```

## Quick Start

### 1. Create test inputs

Create a `test_inputs.json` file with the inputs you want to test:

```json
[
  {"role": "user", "content": "Explain quantum computing in simple terms"},
  {"role": "user", "content": "Write a haiku about programming"},
  {"role": "user", "content": "What are the SOLID principles?"}
]
```

### 2. Snapshot your current prompt

```bash
# With a real LLM
prompt-drift snapshot --name "v1" --prompt prompt.txt --inputs test_inputs.json --model gpt-4o

# Dry run (mock LLM, no API key needed)
prompt-drift snapshot --name "v1" --prompt prompt.txt --inputs test_inputs.json --dry-run
```

### 3. Edit your prompt, take another snapshot

```bash
prompt-drift snapshot --name "v2" --prompt prompt.txt --inputs test_inputs.json --dry-run
```

### 4. Compare

```bash
prompt-drift compare v1 v2
```

Output:

```
 +---------------------------------------------------------+
 | Drift Report: v1 -> v2                                  |
 | Threshold: 20%                                          |
 +---------------------------------------------------------+

       Aggregate Metrics
 ┌─────────────────┬────────┐
 │ Metric          │  Value │
 ├─────────────────┼────────┤
 │ Mean Drift      │ 34.21% │
 │ Max Drift       │ 52.10% │
 │ Min Drift       │ 12.03% │
 │ Total Inputs    │      3 │
 │ Regressions     │      2 │
 └─────────────────┴────────┘

              Per-Input Results
 ┌───┬──────────────────────┬─────────┬──────┬────────┬────────┬───────────┐
 │ # │ Input                │ Levensh │ BLEU │ Cosine │ Drift  │  Status   │
 ├───┼──────────────────────┼─────────┼──────┼────────┼────────┼───────────┤
 │ 1 │ Explain quantum c... │  65.20% │ 42%  │ 71.30% │ 40.12% │ REGRESSION│
 │ 2 │ Write a haiku abo... │  88.00% │ 81%  │ 90.20% │ 12.03% │ MINOR     │
 │ 3 │ What are the SOLID...│  41.30% │ 35%  │ 55.80% │ 52.10% │ REGRESSION│
 └───┴──────────────────────┴─────────┴──────┴────────┴────────┴───────────┘

 2 regression(s) detected (drift > 20%).
```

### 5. Watch mode

Automatically re-run and compare whenever you save changes to your prompt file:

```bash
prompt-drift watch prompt.txt --baseline v1 --inputs test_inputs.json --dry-run
```

## CLI Reference

### `prompt-drift snapshot`

Take a snapshot of prompt outputs against test inputs.

```
Options:
  -n, --name TEXT       Unique snapshot name (required)
  -p, --prompt PATH     Path to prompt template file (required)
  -i, --inputs PATH     Path to test inputs JSON (required)
  -m, --model TEXT      Model identifier (default: gpt-4o)
  --dry-run             Use mock LLM (no API calls)
  --store-dir PATH      Override storage directory
  --force               Overwrite existing snapshot
```

### `prompt-drift compare`

Compare two snapshots and display a drift report.

```
Arguments:
  BASELINE              Name of the baseline snapshot
  CANDIDATE             Name of the candidate snapshot

Options:
  -t, --threshold FLOAT  Regression threshold (default: 0.2)
  --no-diff              Hide text diffs
  -v, --verbose          Show full output text
  --json-out PATH        Export report as JSON
  --store-dir PATH       Override storage directory
```

The command exits with code 1 if regressions are detected, making it CI-friendly.

### `prompt-drift watch`

Watch a prompt file for changes and auto-compare.

```
Arguments:
  PROMPT_FILE           Path to the prompt file to watch

Options:
  -b, --baseline TEXT    Baseline snapshot name (required)
  -i, --inputs PATH     Test inputs JSON (required)
  -m, --model TEXT       Model identifier (default: gpt-4o)
  --interval FLOAT       Polling interval in seconds (default: 2)
  -t, --threshold FLOAT  Regression threshold (default: 0.2)
  --dry-run              Use mock LLM
  --store-dir PATH       Override storage directory
```

### `prompt-drift list`

List all saved snapshots.

### `prompt-drift delete`

Delete a saved snapshot.

## Similarity Metrics

| Metric | What it measures | Best for |
|--------|-----------------|----------|
| **Levenshtein ratio** | Character-level edit distance, normalised to [0,1] | Detecting small wording changes |
| **BLEU score** | N-gram overlap (1-4 grams) with brevity penalty | Measuring structural similarity |
| **Cosine similarity** | TF-IDF vector similarity | Semantic topic overlap |
| **Composite** | Weighted average (30% Lev, 30% BLEU, 40% Cosine) | Overall drift score |

All metrics return a value in **[0, 1]** where **1 = identical**.

The **drift score** is `1 - composite_similarity`, so higher drift = more different.

## Architecture

```
prompt_drift/
├── cli.py          # Click CLI entry point
├── snapshot.py     # Prompt execution + snapshot creation
├── comparator.py   # Snapshot comparison engine
├── similarity.py   # Levenshtein, BLEU, cosine, composite
├── store.py        # JSON file storage in .prompt-drift/
├── reporter.py     # Rich console output formatting
└── models.py       # Pydantic data models
```

**Design principles:**

- **Provider-agnostic:** Works with any LLM. Snapshots store outputs, not API details.
- **Zero mandatory ML dependencies:** BLEU is self-contained. Cosine has a pure-Python fallback.
- **Deterministic dry-run:** Mock LLM is hash-based, so same inputs always give same outputs.
- **CI-ready:** `compare` exits with code 1 on regressions. Export reports as JSON.

## Use in CI/CD

```yaml
# GitHub Actions example
- name: Check for prompt drift
  run: |
    prompt-drift snapshot --name "pr-${{ github.sha }}" \
      --prompt prompts/system.txt \
      --inputs tests/prompt_inputs.json \
      --dry-run
    prompt-drift compare baseline "pr-${{ github.sha }}" \
      --threshold 0.15 \
      --json-out drift-report.json
```

## Development

```bash
git clone https://github.com/muhammadrashid4587/prompt-drift.git
cd prompt-drift
pip install -e ".[dev]"
pytest
```

Run tests with coverage:

```bash
pytest --cov=prompt_drift --cov-report=term-missing
```

## License

MIT License. See [LICENSE](LICENSE) for details.
