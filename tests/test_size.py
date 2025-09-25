#!/usr/bin/env python3
"""
Test for the size metric implementation.

This test verifies:
- Size metric registration and basic functionality
- Device scoring logic with known inputs
- Error handling for invalid URLs
- Integration with the metrics system
- URL extraction functionality
- Weight file detection
- Model download and size calculation

Run from repo root: python tests/test_size.py
Or with pytest: pytest tests/test_size.py -v
"""

import sys
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports (consistent with other tests)
ROOT = Path(__file__).resolve().parent.parent  # Go up from tests/ to repo root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.metrics.size import SizeScoreMetric, _is_weight_file, _extract_model_id_from_url
from app.metrics.base import ResourceBundle
from app.metrics.registry import all_metrics


def test_size_metric_registration():
    """Test that size metric is properly registered."""
    print("Testing size metric registration...")

    # Test registry lookup
    metrics = all_metrics()
    assert "size_score" in metrics, "Size metric should be registered"

    metric = metrics["size_score"]()
    assert isinstance(metric, SizeScoreMetric), "Should return SizeScoreMetric instance"
    assert metric.name == "size_score", "Metric name should be 'size_score'"

    print("✓ Size metric properly registered")


def test_weight_file_detection():
    """Test weight file pattern matching."""
    print("Testing weight file detection...")

    # Weight files should return True
    weight_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model-00001-of-00002.bin",
        "model-00001-of-00002.safetensors",
        "model.gguf",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json"
    ]

    for filename in weight_files:
        assert _is_weight_file(filename), f"{filename} should be detected as weight file"

    # Non-weight files should return False
    other_files = [
        "config.json",
        "README.md",
        "tokenizer.json",
        "vocab.txt",
        "special_tokens_map.json",
        "added_tokens.json"
    ]

    for filename in other_files:
        assert not _is_weight_file(filename), f"{filename} should NOT be detected as weight file"

    print("✓ Weight file detection working correctly")


def test_url_extraction():
    """Test URL extraction from HuggingFace URLs."""
    print("Testing URL extraction...")

    test_cases = [
        ("https://huggingface.co/bert-base-uncased", "bert-base-uncased"),
        ("https://huggingface.co/microsoft/DialoGPT-medium", "microsoft/DialoGPT-medium"),
        ("https://huggingface.co/models/bert-base-uncased", "bert-base-uncased"),
        ("https://huggingface.co/distilbert-base-uncased/", "distilbert-base-uncased"),
        ("https://huggingface.co/facebook/bart-large-cnn?tab=files", "facebook/bart-large-cnn"),
        ("not-a-url", None),
        ("https://github.com/user/repo", None),
        ("", None),
        ("https://huggingface.co/", None),
        ("https://huggingface.co/single-model", "single-model"),  # Single part models are valid
    ]

    for url, expected in test_cases:
        result = _extract_model_id_from_url(url)
        assert result == expected, f"URL {url}: expected {expected}, got {result}"

    print("✓ URL extraction working correctly")


def test_device_scoring():
    """Test device compatibility scoring logic."""
    print("Testing device scoring logic...")

    metric = SizeScoreMetric()

    # Test cases: (size_mb, expected_scores)
    test_cases = [
        # Very small model - perfect scores
        (50, {"raspberry_pi": 1.0, "jetson_nano": 1.0, "desktop_pc": 1.0, "aws_server": 1.0}),

        # Small model - good for most devices
        (100, {"raspberry_pi": 1.0, "jetson_nano": 1.0, "desktop_pc": 1.0, "aws_server": 1.0}),

        # Medium model - can't run on Pi
        (1500, {"raspberry_pi": 0.0, "jetson_nano": 0.833, "desktop_pc": 1.0, "aws_server": 1.0}),

        # Large model - only desktop/server
        (10000, {"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.667, "aws_server": 1.0}),

        # Very large model - server only
        (100000, {"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.0, "aws_server": 0.667}),

        # Extremely large model - no device compatibility
        (300000, {"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.0, "aws_server": 0.0}),
    ]

    for size_mb, expected in test_cases:
        scores = metric._compute_device_scores(size_mb)

        for device, expected_score in expected.items():
            actual_score = scores[device]
            # Allow small floating point differences
            assert abs(actual_score - expected_score) < 0.01, \
                f"Size {size_mb}MB, device {device}: expected {expected_score}, got {actual_score}"

    print("✓ Device scoring logic working correctly")


@patch('app.metrics.size.hf_client')
def test_fetch_model_size_success(mock_hf_client):
    """Test successful model size fetching."""
    print("Testing successful model size fetching...")

    # Mock successful download
    mock_download_result = {
        'success': True,
        'local_path': '/fake/path/model'
    }
    mock_hf_client.download_model.return_value = mock_download_result

    # Mock file system
    mock_path = MagicMock()
    mock_file1 = MagicMock()
    mock_file1.name = "pytorch_model.bin"
    mock_file1.is_file.return_value = True
    mock_file1.stat.return_value.st_size = 100 * 1024 * 1024  # 100MB

    mock_file2 = MagicMock()
    mock_file2.name = "config.json"
    mock_file2.is_file.return_value = True
    mock_file2.stat.return_value.st_size = 1024  # 1KB

    mock_path.exists.return_value = True
    mock_path.rglob.return_value = [mock_file1, mock_file2]

    with patch('app.metrics.size.Path', return_value=mock_path):
        with patch('shutil.rmtree'):  # Mock shutil.rmtree directly
            metric = SizeScoreMetric()
            size = metric._fetch_model_size("https://huggingface.co/bert-base-uncased")

    assert size == 100.0, f"Expected 100.0 MB, got {size}"
    mock_hf_client.download_model.assert_called_once()

    print("✓ Model size fetching working correctly")


@patch('app.metrics.size.hf_client')
def test_fetch_model_size_download_failure(mock_hf_client):
    """Test model size fetching with download failure."""
    print("Testing model size fetching with download failure...")

    # Mock failed download
    mock_download_result = {
        'success': False,
        'error': 'Connection failed'
    }
    mock_hf_client.download_model.return_value = mock_download_result

    metric = SizeScoreMetric()
    size = metric._fetch_model_size("https://huggingface.co/invalid-model")

    assert size is None, "Should return None for failed download"

    print("✓ Download failure handling working correctly")


def test_fetch_model_size_invalid_url():
    """Test model size fetching with invalid URL."""
    print("Testing model size fetching with invalid URL...")

    metric = SizeScoreMetric()
    size = metric._fetch_model_size("not-a-valid-url")

    assert size is None, "Should return None for invalid URL"

    print("✓ Invalid URL handling working correctly")


@patch('app.metrics.size.hf_client')
def test_fetch_model_size_exception(mock_hf_client):
    """Test model size fetching with exception."""
    print("Testing model size fetching with exception...")

    # Mock exception during download
    mock_hf_client.download_model.side_effect = Exception("Network error")

    metric = SizeScoreMetric()
    size = metric._fetch_model_size("https://huggingface.co/bert-base-uncased")

    assert size is None, "Should return None when exception occurs"

    print("✓ Exception handling working correctly")


@patch.object(SizeScoreMetric, '_fetch_model_size')
def test_compute_score_success(mock_fetch_size):
    """Test successful score computation."""
    print("Testing successful score computation...")

    # Mock successful size fetching
    mock_fetch_size.return_value = 1500.0  # 1.5GB

    metric = SizeScoreMetric()
    resource = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=[]
    )

    score = metric._compute_score(resource)

    # Should compute average of device scores for 1.5GB model
    expected_scores = metric._compute_device_scores(1500.0)
    expected_average = sum(expected_scores.values()) / len(expected_scores)

    assert abs(score - expected_average) < 0.001, f"Expected {expected_average}, got {score}"

    print("✓ Score computation working correctly")


@patch.object(SizeScoreMetric, '_fetch_model_size')
def test_compute_score_failure(mock_fetch_size):
    """Test score computation with fetch failure."""
    print("Testing score computation with fetch failure...")

    # Mock failed size fetching
    mock_fetch_size.return_value = None

    metric = SizeScoreMetric()
    resource = ResourceBundle(
        model_url="https://huggingface.co/invalid-model",
        dataset_urls=[],
        code_urls=[]
    )

    score = metric._compute_score(resource)

    assert score == 0.0, f"Expected 0.0 for failed fetch, got {score}"

    print("✓ Failed fetch handling working correctly")


@patch.object(SizeScoreMetric, '_fetch_model_size')
def test_computation_notes(mock_fetch_size):
    """Test computation notes generation."""
    print("Testing computation notes...")

    metric = SizeScoreMetric()
    resource = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=[]
    )

    # Test successful case
    mock_fetch_size.return_value = 150.5
    notes = metric._get_computation_notes(resource)
    assert "150.5MB" in notes
    assert "Device compatibility computed from downloaded model weight files" in notes

    # Test failure case
    mock_fetch_size.return_value = None
    notes = metric._get_computation_notes(resource)
    assert "Could not fetch model size" in notes
    assert "unevaluable" in notes

    print("✓ Computation notes working correctly")


def test_metric_computation():
    """Test full metric computation with ResourceBundle."""
    print("Testing full metric computation...")

    metric = SizeScoreMetric()

    # Test with a model URL (will likely fail to fetch but should handle gracefully)
    resource = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=[]
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


def test_invalid_url_handling():
    """Test handling of invalid URLs."""
    print("Testing invalid URL handling...")

    metric = SizeScoreMetric()

    invalid_urls = [
        "https://invalid-domain.com/not-a-model",
        "not-a-url-at-all",
        "https://huggingface.co/",  # Missing model path
        ""  # Empty string
    ]

    for url in invalid_urls:
        resource = ResourceBundle(model_url=url, dataset_urls=[], code_urls=[])
        result = metric.compute(resource)

        # Should handle gracefully and return 0.0 score
        assert result.score == 0.0, f"Invalid URL {url} should return score 0.0, got {result.score}"
        assert "unevaluable" in result.notes.lower() or "could not fetch" in result.notes.lower(), \
            f"Notes should indicate failure for URL {url}: {result.notes}"

    print("✓ Invalid URL handling working correctly")


def run_all_tests():
    """Run all size metric tests."""
    print("=== Size Metric Tests ===\n")

    try:
        test_size_metric_registration()
        test_weight_file_detection()
        test_url_extraction()
        test_device_scoring()
        test_fetch_model_size_success()
        test_fetch_model_size_download_failure()
        test_fetch_model_size_invalid_url()
        test_fetch_model_size_exception()
        test_compute_score_success()
        test_compute_score_failure()
        test_computation_notes()
        test_metric_computation()
        test_invalid_url_handling()

        print("\n🎉 All size metric tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)