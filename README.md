# Privacy Score Engine

[![PyPI version](https://img.shields.io/pypi/v/privacy-score-engine.svg)](https://pypi.org/project/privacy-score-engine/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/usc-tk/privacy-score-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/usc-tk/privacy-score-engine/actions)

An open-source scoring engine for evaluating privacy policies against the **Australian Privacy Principles (APPs)** under the *Privacy Act 1988*.

The engine extracts structured claims from privacy policy text and scores them across 10 dimensions, producing an overall weighted score (0–100) and a letter grade (A–F). It is designed to be **LLM-agnostic**: you inject your own callable, making it compatible with any provider.

---

## What is this?

This package is the **open-source scoring core** of the [Australian Privacy Score](https://privacy.theucu.com) platform. It provides:

- **Claim extraction** — Analyse raw privacy policy text and extract structured claims per APP dimension
- **Dimension scoring** — Score each dimension (0–10) via an injected LLM evaluator
- **Overall scoring** — Compute a deterministic weighted score (0–100) and letter grade
- **Zero operational dependencies** — No Supabase, no URL fetching, no API keys required

The engine is a pure function: `policy_text + llm_client → claims → scores`. All I/O, storage, and orchestration live in the closed-source scanner layer.

---

## Installation

```bash
pip install privacy-score-engine
```

Or with `uv`:

```bash
uv add privacy-score-engine
```

**Requirements:** Python ≥ 3.12

---

## Quickstart

```python
import anthropic
from engine.extractor import extract_claims
from engine.scorer import compute_scores

# 1. Inject your LLM client as a callable: str → str
client = anthropic.Anthropic()

def llm_client(prompt: str) -> str:
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text

# 2. Extract claims from raw policy text
with open("privacy_policy.txt") as f:
    policy_text = f.read()

extraction = extract_claims(policy_text, llm_client)
print(f"Extracted {len(extraction.claims)} claims")

# 3. Score the claims
result = compute_scores(extraction.claims, llm_client)
print(f"Overall score: {result.overall_score:.1f} / 100  ({result.letter_grade})")

for ds in result.dimension_scores:
    print(f"  {ds.dimension}: {ds.score}/10")
```

---

## How It Works

```
Policy Text
    │
    ▼
┌─────────────────────────────────────────────┐
│  extract_claims(policy_text, llm_client)    │
│                                             │
│  For each of 10 APP dimensions:             │
│  • Build dimension-specific prompt          │
│  • Call llm_client(prompt) → JSON           │
│  • Parse and validate Claim objects         │
└──────────────────┬──────────────────────────┘
                   │  List[Claim]
                   ▼
┌─────────────────────────────────────────────┐
│  compute_scores(claims, llm_client)         │
│                                             │
│  For each dimension:                        │
│  • Gather relevant claims                   │
│  • Ask LLM to score 0–10 with rationale    │
│                                             │
│  Then (pure/deterministic):                 │
│  • Weighted sum → overall score 0–100       │
│  • Threshold lookup → letter grade A–F      │
└──────────────────┬──────────────────────────┘
                   │  ScoringResult
                   ▼
         overall_score, letter_grade,
         dimension_scores[]
```

---

## 10 Privacy Dimensions

| Dimension | Display Name | Weight | APP Reference |
|-----------|--------------|--------|---------------|
| `transparency_clarity` | Transparency & Clarity | 15% | APP 1 |
| `data_collection` | Data Collection Disclosure | 15% | APP 3, APP 6 |
| `third_party_sharing` | Third-Party Sharing & Disclosure | 15% | APP 6, APP 8 |
| `purpose_limitation` | Purpose Limitation & Use | 10% | APP 6 |
| `consumer_rights` | Consumer Rights & Control | 10% | APP 12, APP 13 |
| `data_security` | Data Security | 10% | APP 11 |
| `automated_decision_making` | Automated Decision-Making | 10% | APP 1.4 |
| `childrens_data` | Children's Data | 5% | APP 3.5 |
| `cross_border_flows` | Cross-Border Data Flows | 5% | APP 8 |
| `policy_maintenance` | Policy Maintenance & Accountability | 5% | APP 1 |

---

## Grade Thresholds

| Grade | Score Range | Interpretation |
|-------|-------------|----------------|
| **A** | 80–100 | Excellent — comprehensive, transparent, user-friendly |
| **B** | 65–79 | Good — meets most obligations with minor gaps |
| **C** | 50–64 | Fair — partial disclosure, some obligations unmet |
| **D** | 35–49 | Poor — significant gaps across multiple dimensions |
| **F** | 0–34 | Failing — inadequate privacy disclosures |

---

## API Reference

### `extract_claims(policy_text, llm_client) → ExtractionResult`

Extracts structured claims from raw privacy policy text across all 10 dimensions.

```python
from engine.extractor import extract_claims

result = extract_claims(policy_text, llm_client)
# result.claims: list[Claim]
# result.policy_text_hash: str (SHA-256)
# result.engine_version: str
# result.extracted_at: datetime
```

> **Note:** Each dimension is attempted independently. If the LLM returns an unparseable response for a dimension, that dimension's claims are silently skipped and `result.claims` will contain fewer entries than the maximum possible. No exception is raised. Check `len(result.claims)` if completeness matters for your use case.

### `compute_scores(claims, llm_client) → ScoringResult`

Scores extracted claims across all 10 dimensions and computes the overall weighted score.

```python
from engine.scorer import compute_scores

result = compute_scores(claims, llm_client)
# result.dimension_scores: list[DimensionScore]
# result.overall_score: float  (0.0–100.0)
# result.letter_grade: str     ("A" | "B" | "C" | "D" | "F")
# result.engine_version: str
# result.scored_at: datetime
```

### `Claim` model

| Field | Type | Description |
|-------|------|-------------|
| `dimension` | `str` | One of the 10 dimension keys |
| `claim_type` | `str` | Short snake_case label (e.g. `data_retention_period`) |
| `claim_value` | `dict` | Structured assertion details |
| `confidence` | `float` | 0.0–1.0 extraction certainty |
| `app_reference` | `str` | Specific APP provision (e.g. `"APP 8.1"`) |
| `source_text` | `str` | Verbatim excerpt from the policy |

---

## Security & Limits

The engine analyses privacy policy text scraped from arbitrary third-party
websites — i.e. untrusted input. Two guardrails apply:

- **Prompt-injection containment** — policy text and extracted claims are passed
  to the LLM inside explicit `<untrusted_policy_document>` / `<extracted_claims>`
  delimiters, and the system prompts instruct the model to treat that content as
  data only, never obeying instructions embedded within it. This bounds — but
  cannot wholly eliminate — manipulation by a hostile policy. Scores are always
  clamped to 0–10, so the worst case is an inaccurate score, never out-of-range
  output or code execution.
- **Input size cap** — `extract_claims` truncates policy text longer than
  500,000 characters (logging a warning) to bound LLM cost and memory. A genuine
  privacy policy is far shorter.

---

## Extending the Engine

### Using a Different LLM

The engine accepts any `Callable[[str], str]` as `llm_client`. Example with OpenAI:

```python
from openai import OpenAI

openai = OpenAI()

def llm_client(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

### Scoring a Single Dimension

```python
from engine.scorer import score_dimension

dim_score = score_dimension("data_security", claims, llm_client)
# dim_score.score: int  (0–10)
# dim_score.rationale: str
```

### Computing the Overall Score Without LLM

The overall score and grade are deterministic — no LLM required:

```python
from engine.scorer import compute_overall_score, assign_grade

overall = compute_overall_score(dimension_scores)  # float
grade = assign_grade(overall)                       # "A" | "B" | "C" | "D" | "F"
```

---

## Development

```bash
# Clone and set up
git clone https://github.com/usc-tk/privacy-score-engine
cd privacy-score-engine
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check src/
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

The most impactful contributions are improvements to the **extraction prompts** in `src/engine/prompts/`. Better prompts lead to more accurate claim extraction across all scored services.

---

## License

MIT — see [LICENSE](LICENSE) for details.
