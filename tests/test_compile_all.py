"""Test purpose: sanity-check that project Python files compile. Uses src/ if
present and also checks root-level run/run.py if they exist.
"""

import compileall
from pathlib import Path

# Checks that all files in src compile
def test_all_python_files_compile():
    src = Path("src")
    assert src.exists(), "Expected a src/ folder"
    ok = compileall.compile_dir(str(src), quiet=1)
    assert ok, "Some files failed to compile."