#Fixes ModuleNotFoundError: No module named 'app' when running pytest from repo root
#Central place for helpers like finding root level run CLI
#Prevents repeated path logic in each file
#!Run all tests using pytest in venv using cmd: python -m pytest -q -rs

# tests/conftest.py
import sys
import subprocess
from pathlib import Path
import pytest

# Repo root (…/Team_Repo-*)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Prefer the src/ layout (src/app/...), else allow root-level app/ if you ever use it
if (SRC / "app").exists():
    sys.path.insert(0, str(SRC))
elif (ROOT / "app").exists():
    sys.path.insert(0, str(ROOT))
# If neither exists, no change; tests that import 'app' will fail/skip accordingly.

# Locate the root-level CLI (run or run.py)
CLI = ROOT / ("run" if (ROOT / "run").exists() else "run.py")

@pytest.fixture(scope="session")
def cli_path():
    """Path to the root-level CLI script, or skip if it doesn't exist."""
    if not CLI.exists():
        pytest.skip("Root-level CLI not found (run / run.py)")
    return CLI

@pytest.fixture
def run_cli(cli_path):
    """Run the CLI and return (code, stdout, stderr)."""
    def _run(*argv: str):
        r = subprocess.run(
            [sys.executable, str(cli_path), *argv],
            capture_output=True, text=True
        )
        return r.returncode, r.stdout, r.stderr
    return _run

def pytest_report_header(config):
    """Nice to have: show which CLI/path pytest is using."""
    first = sys.path[0] if sys.path else "<empty>"
    return [f"CLI: {CLI if CLI.exists() else 'not found'}", f"sys.path[0]: {first}"]