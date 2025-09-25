#!/usr/bin/env python3
"""
Test for the code quality metric implementation.

This test verifies:
- Code quality metric registration and basic functionality
- Repository structure analysis
- Code quality assessment logic
- Error handling for invalid/inaccessible repositories
- Integration with the metrics system
- Mock testing for git operations

Run from repo root: python tests/test_code_quality.py
Or with pytest: pytest tests/test_code_quality.py -v
"""

import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports (consistent with other tests)
ROOT = Path(__file__).resolve().parent.parent  # Go up from tests/ to repo root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.metrics.code_quality import CodeQualityMetric
from app.metrics.base import ResourceBundle
from app.metrics.registry import all_metrics


def test_code_quality_metric_registration():
    """Test that code quality metric is properly registered."""
    print("Testing code quality metric registration...")

    # Test registry lookup
    metrics = all_metrics()
    assert "code_quality" in metrics, "Code quality metric should be registered"

    metric = metrics["code_quality"]()
    assert isinstance(metric, CodeQualityMetric), "Should return CodeQualityMetric instance"
    assert metric.name == "code_quality", "Metric name should be 'code_quality'"

    print("✓ Code quality metric properly registered")


def test_no_code_repositories():
    """Test behavior when no code repositories are provided."""
    print("Testing no code repositories...")

    metric = CodeQualityMetric()
    resource = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=[]
    )

    result = metric.compute(resource)

    assert result.score == 0.0, f"Expected 0.0 for no repositories, got {result.score}"
    assert "No code repositories found" in result.notes

    print("✓ No code repositories handling working correctly")


def test_non_github_repositories():
    """Test behavior with non-GitHub repositories."""
    print("Testing non-GitHub repositories...")

    metric = CodeQualityMetric()
    resource = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=["https://gitlab.com/user/repo", "https://bitbucket.org/user/repo"]
    )

    result = metric.compute(resource)

    assert result.score == 0.0, f"Expected 0.0 for non-GitHub repos, got {result.score}"
    assert "No GitHub repositories found" in result.notes

    print("✓ Non-GitHub repositories handling working correctly")


def test_repository_structure_analysis():
    """Test repository structure analysis with mock directory."""
    print("Testing repository structure analysis...")

    metric = CodeQualityMetric()

    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)

        # Create files that should score points
        (repo_path / "README.md").touch()
        (repo_path / "LICENSE").touch()
        (repo_path / "requirements.txt").touch()
        (repo_path / ".gitignore").touch()

        # Create test directory
        test_dir = repo_path / "tests"
        test_dir.mkdir()
        (test_dir / "test_example.py").touch()

        # Create CI config
        github_dir = repo_path / ".github" / "workflows"
        github_dir.mkdir(parents=True)
        (github_dir / "ci.yml").touch()

        # Create documentation
        docs_dir = repo_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.md").touch()

        # Create source structure
        src_dir = repo_path / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").touch()

        score = metric._analyze_repository_structure(repo_path)

        # Should score highly with all these quality indicators
        assert 0.8 <= score <= 1.0, f"Expected high score for complete structure, got {score}"

    print("✓ Repository structure analysis working correctly")


def test_code_quality_analysis():
    """Test code quality analysis with mock Python files."""
    print("Testing code quality analysis...")

    metric = CodeQualityMetric()

    # Create a temporary directory with Python files
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)

        # Create a Python file with good practices
        good_file = repo_path / "good_code.py"
        good_file.write_text('''
"""
This is a well-documented module with type hints.
"""

def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b

class Calculator:
    """A simple calculator class."""

    def __init__(self) -> None:
        """Initialize calculator."""
        pass
''')

        # Create another file
        (repo_path / "__init__.py").touch()
        (repo_path / "main.py").write_text("# Main module")

        # Create quality config files
        (repo_path / ".flake8").touch()
        (repo_path / "mypy.ini").touch()

        score = metric._analyze_code_quality(repo_path)

        # Should score well with docstrings, type hints, and structure
        assert 0.6 <= score <= 1.0, f"Expected good score for quality code, got {score}"

    print("✓ Code quality analysis working correctly")


def test_code_quality_analysis_no_python():
    """Test code quality analysis with no Python files."""
    print("Testing code quality analysis with no Python files...")

    metric = CodeQualityMetric()

    # Create a temporary directory with non-Python files
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)

        # Create some JavaScript files
        (repo_path / "app.js").touch()
        (repo_path / "index.html").touch()

        score = metric._analyze_code_quality(repo_path)

        # Should get partial credit for having code files
        assert 0.0 < score < 0.5, f"Expected low but non-zero score for non-Python, got {score}"

    print("✓ Code quality analysis with no Python files working correctly")


@patch('subprocess.run')
def test_successful_repository_analysis(mock_subprocess):
    """Test successful repository analysis with mocked git clone."""
    print("Testing successful repository analysis...")

    # Mock successful git clone
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_subprocess.return_value = mock_result

    metric = CodeQualityMetric()

    with patch.object(metric, '_analyze_repository_structure', return_value=0.8):
        with patch.object(metric, '_analyze_code_quality', return_value=0.7):
            with patch('tempfile.mkdtemp', return_value='/fake/temp'):
                with patch('shutil.rmtree'):
                    score = metric._analyze_repository("https://github.com/user/repo")

    expected_score = (0.8 * 0.6) + (0.7 * 0.4)  # Weighted average
    assert abs(score - expected_score) < 0.01, f"Expected {expected_score}, got {score}"

    print("✓ Successful repository analysis working correctly")


@patch('subprocess.run')
def test_failed_repository_clone(mock_subprocess):
    """Test handling of failed git clone."""
    print("Testing failed repository clone...")

    # Mock failed git clone
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_subprocess.return_value = mock_result

    metric = CodeQualityMetric()

    with patch('tempfile.mkdtemp', return_value='/fake/temp'):
        with patch('shutil.rmtree'):
            score = metric._analyze_repository("https://github.com/user/nonexistent-repo")

    assert score is None, f"Expected None for failed clone, got {score}"

    print("✓ Failed repository clone handling working correctly")


@patch('subprocess.run')
def test_repository_timeout(mock_subprocess):
    """Test handling of git clone timeout."""
    print("Testing repository clone timeout...")

    # Mock timeout exception
    mock_subprocess.side_effect = subprocess.TimeoutExpired("git", 60)

    metric = CodeQualityMetric()

    with patch('tempfile.mkdtemp', return_value='/fake/temp'):
        with patch('shutil.rmtree'):
            score = metric._analyze_repository("https://github.com/user/slow-repo")

    assert score is None, f"Expected None for timeout, got {score}"

    print("✓ Repository timeout handling working correctly")


@patch.object(CodeQualityMetric, '_analyze_repository')
def test_compute_score_with_repositories(mock_analyze):
    """Test score computation with multiple repositories."""
    print("Testing score computation with repositories...")

    # Mock repository analysis returning different scores
    mock_analyze.side_effect = [0.8, 0.6, 0.9]

    metric = CodeQualityMetric()
    resource = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=[
            "https://github.com/user/repo1",
            "https://github.com/user/repo2",
            "https://github.com/user/repo3"
        ]
    )

    score = metric._compute_score(resource)

    # Should average the three scores plus diversity bonus
    expected_base = (0.8 + 0.6 + 0.9) / 3  # 0.767
    expected_bonus = min(0.1, (3 - 1) * 0.05)  # 0.1
    expected_total = expected_base + expected_bonus

    assert abs(score - expected_total) < 0.01, f"Expected {expected_total}, got {score}"

    print("✓ Score computation with repositories working correctly")


@patch.object(CodeQualityMetric, '_analyze_repository')
def test_compute_score_with_failed_repositories(mock_analyze):
    """Test score computation when all repositories fail analysis."""
    print("Testing score computation with failed repositories...")

    # Mock all repository analyses failing
    mock_analyze.return_value = None

    metric = CodeQualityMetric()
    resource = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=["https://github.com/user/broken-repo"]
    )

    score = metric._compute_score(resource)

    assert score == 0.0, f"Expected 0.0 for all failed analyses, got {score}"

    print(" Failed repositories handling working correctly")


def test_computation_notes():
    """Test computation notes generation."""
    print("Testing computation notes...")

    metric = CodeQualityMetric()

    # Test with no repositories
    resource_no_code = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=[]
    )
    notes_no_code = metric._get_computation_notes(resource_no_code)
    assert "No code repositories found" in notes_no_code

    # Test with GitHub repositories
    resource_with_code = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=["https://github.com/user/repo1", "https://github.com/user/repo2"]
    )
    notes_with_code = metric._get_computation_notes(resource_with_code)
    assert "Analyzed" in notes_with_code
    assert "GitHub repositories" in notes_with_code

    # Test with non-GitHub repositories
    resource_non_github = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=["https://gitlab.com/user/repo"]
    )
    notes_non_github = metric._get_computation_notes(resource_non_github)
    assert "No GitHub repositories found" in notes_non_github

    print("✓ Computation notes working correctly")


def test_metric_computation():
    """Test full metric computation with ResourceBundle."""
    print("Testing full metric computation...")

    metric = CodeQualityMetric()

    # Test with GitHub repositories (will likely fail but should handle gracefully)
    resource = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=["https://github.com/huggingface/transformers"]
    )

    result = metric.compute(resource)

    # Should return valid MetricResult
    assert hasattr(result, 'score'), "Result should have score attribute"
    assert hasattr(result, 'latency_ms'), "Result should have latency_ms attribute"
    assert hasattr(result, 'notes'), "Result should have notes attribute"

    # Score should be between 0 and 1
    assert 0.0 <= result.score <= 1.0, f"Score should be between 0-1, got {result.score}"

    # Should have timing information
    assert isinstance(result.latency_ms, int), "Latency should be integer milliseconds"
    assert result.latency_ms >= 0, "Latency should be non-negative"

    # Should have descriptive notes
    assert isinstance(result.notes, str), "Notes should be string"
    assert len(result.notes) > 0, "Notes should not be empty"

    print(f"✓ Metric computation working (score: {result.score}, latency: {result.latency_ms}ms)")
    print(f"  Notes: {result.notes}")


def run_all_tests():
    """Run all code quality metric tests."""
    print("=== Code Quality Metric Tests ===\n")

    try:
        test_code_quality_metric_registration()
        test_no_code_repositories()
        test_non_github_repositories()
        test_repository_structure_analysis()
        test_code_quality_analysis()
        test_code_quality_analysis_no_python()
        test_successful_repository_analysis()
        test_failed_repository_clone()
        test_repository_timeout()
        test_compute_score_with_repositories()
        test_compute_score_with_failed_repositories()
        test_computation_notes()
        test_metric_computation()

        print("\n All code quality metric tests passed!")
        return True

    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)