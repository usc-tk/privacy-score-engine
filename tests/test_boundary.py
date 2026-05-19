"""Boundary enforcement tests for the open-source engine (NFR15).

These tests ensure the engine package never acquires operational imports,
environment variable access, or hardcoded credentials. They run on every CI
pass to prevent accidental boundary violations.

The engine must be a pure function: policy_text + llm_client → claims/scores.
No I/O, no database, no HTTP, no env vars.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# Root of engine source (relative to this test file)
ENGINE_SRC = Path(__file__).parent.parent / "src" / "engine"

# All .py files under src/engine/
ENGINE_FILES = list(ENGINE_SRC.rglob("*.py"))

# Sanity guard: if the directory is empty or missing, tests would pass vacuously.
assert ENGINE_FILES, (
    f"ENGINE_FILES is empty — expected Python source files under {ENGINE_SRC}. "
    "Check that the test is running from the correct working directory."
)


def _get_imports(path: Path) -> list[tuple[str, int]]:
    """Return (module_name, lineno) pairs for all import statements in a file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module, node.lineno))
    return imports


def _file_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ─── Forbidden import roots ───────────────────────────────────────────────────

FORBIDDEN_IMPORT_PREFIXES = [
    "supabase",
    "httpx",
    "requests",
    "aiohttp",
    "urllib",       # covers urllib.request, urllib.parse, urllib.error, etc.
    "http.client",  # stdlib HTTP (not covered by urllib prefix)
    "playwright",
    "selenium",
    "pyppeteer",
    "boto3",
    "botocore",
    "google.cloud",
    "azure",
    "asyncpg",
    "psycopg",
    "sqlalchemy",
    "databases",
    "dotenv",
    "decouple",
    "anthropic",
    "openai",
    "cohere",
    "mistralai",
]


def test_no_forbidden_imports() -> None:
    """Engine source must not import any operational or LLM-specific libraries.

    The engine accepts an injected llm_client callable — it must never import
    a specific LLM SDK. All HTTP, database, and cloud SDKs are forbidden.
    """
    violations: list[str] = []

    for py_file in ENGINE_FILES:
        if "__pycache__" in str(py_file):
            continue
        for module, lineno in _get_imports(py_file):
            for forbidden in FORBIDDEN_IMPORT_PREFIXES:
                if module == forbidden or module.startswith(forbidden + "."):
                    rel = py_file.relative_to(ENGINE_SRC.parent.parent)
                    violations.append(
                        f"{rel}:{lineno} — forbidden import '{module}' "
                        f"(matched rule: '{forbidden}')"
                    )

    assert not violations, (
        "Engine source contains forbidden imports (NFR15 violation):\n"
        + "\n".join(f"  • {v}" for v in violations)
    )


# ─── Dynamic import bypass detection ─────────────────────────────────────────

# Patterns that bypass static AST import analysis
DYNAMIC_IMPORT_PATTERNS = [
    r"\b__import__\s*\(",
    r"\bimportlib\.import_module\s*\(",
    r"\bimportlib\.util\.spec_from_file_location\s*\(",
]


def test_no_dynamic_imports() -> None:
    """Engine source must not use dynamic import mechanisms.

    Dynamic imports (`__import__`, `importlib.import_module`) bypass the AST
    import check in `test_no_forbidden_imports` and could smuggle in forbidden
    dependencies invisibly at runtime.
    """
    violations: list[str] = []

    for py_file in ENGINE_FILES:
        if "__pycache__" in str(py_file):
            continue
        source = _file_source(py_file)
        for pattern in DYNAMIC_IMPORT_PATTERNS:
            for match in re.finditer(pattern, source):
                lineno = source[: match.start()].count("\n") + 1
                rel = py_file.relative_to(ENGINE_SRC.parent.parent)
                violations.append(
                    f"{rel}:{lineno} — dynamic import via '{match.group().strip()}'"
                )

    assert not violations, (
        "Engine source uses dynamic imports (potential boundary bypass):\n"
        + "\n".join(f"  • {v}" for v in violations)
    )


# ─── Forbidden env-var access ─────────────────────────────────────────────────

ENV_VAR_PATTERNS = [
    r"\bos\.environ\b",
    r"\bos\.getenv\b",
    r"\bos\.environ\.get\b",
    r"\bgetenv\b",
]


def test_no_environment_variable_access() -> None:
    """Engine source must not access environment variables.

    The engine is a pure function: all configuration (e.g. LLM client) must be
    injected by the caller. Reading env vars would couple the engine to
    deployment environment and break open-source usability.
    """
    violations: list[str] = []

    for py_file in ENGINE_FILES:
        if "__pycache__" in str(py_file):
            continue
        source = _file_source(py_file)
        for pattern in ENV_VAR_PATTERNS:
            for match in re.finditer(pattern, source):
                lineno = source[: match.start()].count("\n") + 1
                rel = py_file.relative_to(ENGINE_SRC.parent.parent)
                violations.append(
                    f"{rel}:{lineno} — env var access via '{match.group()}'"
                )

    assert not violations, (
        "Engine source accesses environment variables (NFR15 violation):\n"
        + "\n".join(f"  • {v}" for v in violations)
    )


# ─── Forbidden credential patterns ────────────────────────────────────────────

# Patterns that indicate hardcoded secrets or Supabase URLs
CREDENTIAL_PATTERNS = [
    # Supabase URLs
    (r"https://[a-z0-9]+\.supabase\.co", "Supabase URL"),
    # Generic API key assignment patterns (long alphanumeric secrets)
    (r'(?:api_key|apikey|secret|token|password)\s*=\s*["\'][A-Za-z0-9_\-]{20,}["\']', "hardcoded credential"),
    # Service role key prefix (Supabase)
    (r"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "JWT token (likely Supabase service role key)"),
]


def test_no_hardcoded_credentials() -> None:
    """Engine source must not contain hardcoded API keys, tokens, or URLs.

    This enforces NFR15: the open-source engine repository must not contain
    API keys, database credentials, or agency client data.
    """
    violations: list[str] = []

    for py_file in ENGINE_FILES:
        if "__pycache__" in str(py_file):
            continue
        source = _file_source(py_file)
        for pattern, label in CREDENTIAL_PATTERNS:
            for match in re.finditer(pattern, source, re.IGNORECASE):
                lineno = source[: match.start()].count("\n") + 1
                rel = py_file.relative_to(ENGINE_SRC.parent.parent)
                violations.append(
                    f"{rel}:{lineno} — {label}: '{match.group()[:60]}...'"
                    if len(match.group()) > 60
                    else f"{rel}:{lineno} — {label}: '{match.group()}'"
                )

    assert not violations, (
        "Engine source contains hardcoded credentials (NFR15 violation):\n"
        + "\n".join(f"  • {v}" for v in violations)
    )
