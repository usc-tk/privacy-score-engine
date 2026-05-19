# Contributing to Privacy Score Engine

Thank you for your interest in improving the Privacy Score Engine! This is an open-source project and community contributions are what make it better.

---

## Welcome

The Privacy Score Engine scores privacy policies against the Australian Privacy Principles (APPs). The most impactful thing you can contribute is **better extraction prompts** — they directly improve how accurately we classify privacy policy language across all services scored by the platform.

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/<your-username>/privacy-score-engine
cd privacy-score-engine

# Install dependencies
uv sync
```

### Run the Tests

```bash
uv run pytest
```

All tests must pass before submitting a pull request.

### Run the Linter

```bash
uv run ruff check src/
```

Fix any lint errors before submitting.

---

## Open-Source Boundary

This engine is the **open-source half** of a larger system. To keep it that way, there are strict rules about what can and cannot be added.

### What belongs in this engine

- Claim extraction prompts (`src/engine/prompts/`)
- Confidence scoring logic
- Dimension scoring rubric and weights (`src/engine/models.py`)
- Score computation and grade assignment (`src/engine/scorer.py`)
- Pydantic models for claims and scores (`src/engine/models.py`)
- Tests for all of the above

### What must NEVER be added

The following have no place in this package:

| Category | Examples |
|----------|---------|
| Database clients | `supabase`, `asyncpg`, `psycopg2`, `sqlalchemy` |
| HTTP clients | `httpx`, `requests`, `aiohttp`, `urllib` |
| Browser automation | `playwright`, `selenium`, `pyppeteer` |
| Cloud SDKs | `boto3`, `google-cloud-*`, `azure-*` |
| Environment variables | `os.environ`, `os.getenv`, `python-dotenv` |
| API keys or credentials | Any hardcoded secret, token, or connection string |
| Operational logic | Batch processing, cost monitoring, retry orchestration |
| Agency tooling | Remediation drafting, engagement tracking |

A CI test (`tests/test_boundary.py`) enforces this. PRs that add forbidden imports will fail CI automatically.

---

## Submitting a Pull Request

1. **Branch from `main`** — create a descriptive branch name, e.g. `improve-data-security-prompt`
2. **Write tests** — every change to extraction or scoring logic needs a corresponding test
3. **Keep lint clean** — `uv run ruff check src/` must pass with zero warnings
4. **Respect the boundary** — no operational imports (see above)
5. **Write a clear PR description** — explain what changed and why it improves accuracy

### PR checklist

- [ ] Tests added or updated
- [ ] `uv run pytest` passes
- [ ] `uv run ruff check src/` passes
- [ ] No forbidden imports added
- [ ] PR description explains the change

---

## Improving Extraction Prompts

Extraction prompts live in `src/engine/prompts/extraction.py`. Each of the 10 dimensions has its own prompt.

### Tips for better prompts

- **Be specific about what to look for** — list concrete signal phrases for each dimension
- **Calibrate confidence thresholds** — `0.90+` for explicit statements, `0.75–0.89` for clear with minor inference, `0.60–0.74` for vague/indirect
- **Test against real policies** — use the fixtures in `tests/fixtures/sample_policies/` as your benchmark
- **Watch for false positives** — a claim that isn't really there is worse than a missed claim

### Testing prompt changes

```python
# Quick manual test
from engine.extractor import extract_claims_for_dimension

def my_llm(prompt: str) -> str:
    # ... your LLM client
    pass

with open("tests/fixtures/sample_policies/telstra_privacy_excerpt.txt") as f:
    text = f.read()

claims = extract_claims_for_dimension(text, "data_security", my_llm)
for c in claims:
    print(c.model_dump_json(indent=2))
```

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. By participating, you agree to uphold its standards. Report unacceptable behaviour to the project maintainers.

---

## Questions?

Open a [GitHub Issue](https://github.com/usc-tk/privacy-score-engine/issues) with the `question` label. We're happy to help.
