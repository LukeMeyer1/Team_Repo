#!/usr/bin/env python3
"""Test purpose: end-to-end checks for the metrics system.

What this file verifies
- Registry: all expected metric implementations are registered (e.g., license,
  ramp_up_time, bus_factor, performance_claims, size_score, dataset_and_code_score,
  dataset_quality, code_quality).
- Metric contracts: each metric’s factory().compute(ResourceBundle) returns a result
  with:
    • score in [0, 1]
    • latency_ms >= 0
    • notes present (string or similar)
- Engine output: running the metrics engine (run_bundle) returns JSON/NDJSON with:
    • required fields: URL, NetScore, NetScore_Latency
    • for each metric: <name> and <name>_Latency
    • NetScore ∈ [0,1] and lies within min/max of individual metric scores
    • weights sum to ~1.0
- Ecosystem awareness: scores respond to inputs as expected:
    • dataset_quality increases when dataset URLs are present
    • code_quality increases when code URLs are present

How to run
- Pytest (recommended):
    python -m pytest -q -rs tests\\test_metrics.py
- With coverage:
    python -m pytest --cov=src --cov-report=term-missing -q tests\\test_metrics.py
- Script mode (prints a colored summary without pytest):
    python tests\\test_metrics.py

Notes
- Some checks are “pending-safe”: if certain metrics are not implemented yet,
  the test skips instead of failing.
- Uses the src/ layout via tests/conftest.py so `import app...` works without install.
"""

import sys
from pathlib import Path
import json

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Make pytest optional since it may not be available
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Mock pytest functions if not available
    class MockPytest:
        def skip(self, reason):
            print(f"SKIPPING: {reason}")
            return
        def xfail(self, reason):
            print(f"EXPECTED FAIL: {reason}")
            return
    pytest = MockPytest()


# ANSI color codes for terminal output
# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def colored(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"

def success(text: str) -> str: return colored(f"[PASS] {text}", Colors.GREEN)
def failure(text: str) -> str: return colored(f"[FAIL] {text}", Colors.RED)
def warning(text: str) -> str: return colored(f"[WARN] {text}", Colors.YELLOW)
def info(text: str)    -> str: return colored(f"[INFO] {text}", Colors.BLUE)
def header(text: str)  -> str: return colored(text, Colors.PURPLE + Colors.BOLD)

def test_metric_registration():
    print(header("=== Testing Metric Registration ==="))

    # Import to trigger registrations (side effects)
    from app.metrics.registry import all_metrics

    registered = all_metrics()
    print(info(f"Registered metrics: {sorted(registered.keys())}"))

    expected = {
        "license",
        "ramp_up_time",
        "bus_factor",
        "performance_claims",
        "size_score",
        "dataset_and_code_score",
        "dataset_quality",
        "code_quality",
    }

    # Assert each expected metric is present
    missing = expected - set(registered.keys())
    extra   = set(registered.keys()) - expected

    if missing:
        print(failure(f"Missing metrics: {sorted(missing)}"))
    if extra:
        print(warning(f"Unexpected metrics found: {sorted(extra)}"))

    assert not missing, f"Registry is missing: {sorted(missing)}"
    # Extra metrics aren’t necessarily wrong; fail only if you want strict parity:
    # assert not extra, f"Unexpected metrics: {sorted(extra)}"

def test_individual_metrics():
    print(header("\n=== Testing Individual Metrics ==="))

    from app.metrics.registry import all_metrics
    from app.metrics.base import ResourceBundle

    rb = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[
            "https://huggingface.co/datasets/squad",
            "https://huggingface.co/datasets/glue",
        ],
        code_urls=[
            "https://github.com/huggingface/transformers",
            "https://github.com/google-research/bert",
        ],
    )

    factories = all_metrics()
    assert factories, "No metrics registered."

    for name, factory in sorted(factories.items()):
        metric = factory()
        result = metric.compute(rb)
        print(success(f"{name}: score={result.score:.3f}, latency={result.latency_ms}ms"))

        assert hasattr(result, "score"), f"{name}: result missing 'score'"
        assert 0 <= result.score <= 1, f"{name}: score {result.score} not in [0,1]"
        assert hasattr(result, "latency_ms"), f"{name}: result missing 'latency_ms'"
        assert result.latency_ms >= 0, f"{name}: latency {result.latency_ms} must be >= 0"
        assert hasattr(result, "notes"), f"{name}: result missing 'notes'"

def test_engine_functionality():
    print(header("\n=== Testing Engine Functionality ==="))

    from app.metrics.registry import all_metrics
    from app.metrics.engine import run_bundle
    from app.metrics.base import ResourceBundle

    rb = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=["https://huggingface.co/datasets/squad"],
        code_urls=["https://github.com/huggingface/transformers"],
    )

    factories = all_metrics()
    metrics = [f() for f in factories.values()]
    assert metrics, "No metrics available to run."

    weights = {
        "license": 0.15,
        "ramp_up_time": 0.15,
        "bus_factor": 0.10,
        "performance_claims": 0.20,
        "size_score": 0.15,
        "dataset_and_code_score": 0.10,
        "dataset_quality": 0.10,
        "code_quality": 0.05,
    }
    # Optional: ensure weights sum ~ 1.0
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    ndjson = run_bundle(rb, metrics, weights)
    print(info(f"Engine output: {ndjson!r}"))

    # Accept either a single JSON object or NDJSON (one JSON per line)
    lines = [ln for ln in (ndjson or "").splitlines() if ln.strip()]
    assert lines, "Engine returned empty output."

    # Use the last line as the summary record (common pattern)
    parsed = json.loads(lines[-1])

    # Required fields (updated to match actual output format)
    for field in ("name", "net_score", "net_score_latency"):
        assert field in parsed, f"Missing required field: {field}"

    # All metric fields (score + latency)
    missing_fields = []
    for m in weights.keys():
        if m not in parsed:
            missing_fields.append(m)
        if f"{m}_latency" not in parsed:  # Updated to use lowercase
            missing_fields.append(f"{m}_latency")
    assert not missing_fields, f"Missing metric fields: {missing_fields}"

    # NetScore sanity (updated field name)
    net = parsed["net_score"]
    assert 0 <= net <= 1, f"net_score {net} not in [0,1]"

    # Extract individual metric scores (handle size_score being an object)
    indiv = []
    for m in weights.keys():
        if m == "size_score":
            # Size score is an object with device scores, use desktop_pc as representative
            size_obj = parsed[m]
            if isinstance(size_obj, dict):
                indiv.append(size_obj.get("desktop_pc", 0.0))
            else:
                indiv.append(size_obj)
        else:
            indiv.append(parsed[m])
    
    if indiv:  # weighted average should lie within min/max of components
        mn, mx = min(indiv), max(indiv)
        assert mn - 1e-9 <= net <= mx + 1e-9, f"net_score {net} outside [{mn}, {mx}]"

    print(success(f"NDJSON structure and scores validated (net_score: {net:.3f})"))

def test_metric_ecosystem_awareness():
    print(header("\n=== Testing Ecosystem Awareness ==="))

    from app.metrics.registry import all_metrics
    from app.metrics.base import ResourceBundle

    bundles = {
        "no_extras":     ResourceBundle("https://huggingface.co/model1", [], []),
        "with_datasets": ResourceBundle("https://huggingface.co/model2", ["https://dataset1"], []),
        "with_code":     ResourceBundle("https://huggingface.co/model3", [], ["https://code1"]),
        "complete":      ResourceBundle("https://huggingface.co/model4", ["https://dataset1"], ["https://code1"]),
    }

    eco_metrics = ["dataset_and_code_score", "dataset_quality", "code_quality"]
    factories = all_metrics()

    # If these metrics aren't implemented yet, skip rather than fail.
    not_found = [m for m in eco_metrics if m not in factories]
    if not_found:
        pytest.skip(f"Ecosystem metrics not implemented yet: {not_found}")

    results = {}
    for bname, bundle in bundles.items():
        results[bname] = {}
        for m in eco_metrics:
            metric = factories[m]()
            results[bname][m] = metric.compute(bundle).score

    # dataset_quality should be lower without datasets
    dq_none = results["no_extras"]["dataset_quality"]
    dq_some = results["with_datasets"]["dataset_quality"]
    assert dq_none <= dq_some, f"dataset_quality should increase with datasets (got {dq_none} vs {dq_some})"

    # code_quality should be lower without code
    cq_none = results["no_extras"]["code_quality"]
    cq_some = results["with_code"]["code_quality"]
    assert cq_none <= cq_some, f"code_quality should increase with code (got {cq_none} vs {cq_some})"

# --- Optional script-mode runner (nice colored summary when not using pytest) ---
def main():
    print(colored("Testing Metrics System (All 8 Metrics)", Colors.CYAN + Colors.BOLD))
    print()
    tests = [
        test_metric_registration,
        test_individual_metrics,
        test_engine_functionality,
        test_metric_ecosystem_awareness,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(success(f"{t.__name__} PASSED"))
            passed += 1
        except Exception as e:
            print(failure(f"{t.__name__} FAILED: {e}"))
    print(header("\n=== FINAL RESULTS ==="))
    print(info(f"Passed: {passed}/{len(tests)}"))
    return passed == len(tests)

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)