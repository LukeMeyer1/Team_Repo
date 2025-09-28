#!/usr/bin/env python3
"""
Test for the dataset quality metric implementation.

This test verifies:
- Dataset quality metric registration and basic functionality
- Dataset information fetching and analysis
- Quality scoring algorithms for popularity, documentation, metadata, diversity, licensing
- Error handling for invalid/inaccessible datasets
- Integration with the metrics system
- Mock testing for HuggingFace API operations

Run from repo root: python tests/test_dataset_quality.py
Or with pytest: pytest tests/test_dataset_quality.py -v
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports (consistent with other tests)
ROOT = Path(__file__).resolve().parent.parent  # Go up from tests/ to repo root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.metrics.implementations.dataset_quality import DatasetQualityMetric
from app.metrics.base import ResourceBundle
from app.metrics.registry import all_metrics


def test_dataset_quality_metric_registration():
    """Test that dataset quality metric is properly registered."""
    print("Testing dataset quality metric registration...")

    # Test registry lookup
    metrics = all_metrics()
    assert "dataset_quality" in metrics, "Dataset quality metric should be registered"

    metric = metrics["dataset_quality"]()
    assert isinstance(metric, DatasetQualityMetric), "Should return DatasetQualityMetric instance"
    assert metric.name == "dataset_quality", "Metric name should be 'dataset_quality'"

    print("✓ Dataset quality metric properly registered")


def test_no_datasets():
    """Test behavior when no datasets are provided."""
    print("Testing no datasets...")

    metric = DatasetQualityMetric()
    resource = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=[]
    )

    result = metric.compute(resource)

    assert result.score == 0.0, f"Expected 0.0 for no datasets, got {result.score}"
    assert "No datasets found" in result.notes

    print("✓ No datasets handling working correctly")


def test_hf_client_unavailable():
    """Test behavior when HuggingFace client is unavailable."""
    print("Testing HuggingFace client unavailable...")

    metric = DatasetQualityMetric()

    # Mock hf_client as None
    with patch('app.metrics.implementations.dataset_quality.hf_client', None):
        resource = ResourceBundle(
            model_url="https://huggingface.co/bert-base-uncased",
            dataset_urls=["https://huggingface.co/datasets/squad"],
            code_urls=[]
        )

        result = metric.compute(resource)

        assert result.score == 0.1, f"Expected 0.1 for unavailable client, got {result.score}"
        assert "HuggingFace client unavailable" in result.notes

    print("✓ HuggingFace client unavailable handling working correctly")


def test_extract_dataset_id():
    """Test dataset ID extraction from various URL formats."""
    print("Testing dataset ID extraction...")

    metric = DatasetQualityMetric()

    # Test various URL formats
    test_cases = [
        ("https://huggingface.co/datasets/squad", "squad"),
        ("https://huggingface.co/datasets/glue/cola", "glue/cola"),
        ("https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k", "microsoft/orca-math-word-problems-200k"),
        ("squad", "squad"),  # Direct dataset ID
        ("glue/cola", "glue/cola"),  # Direct dataset ID with org
        ("https://gitlab.com/datasets/something", None),  # Non-HF URL
        ("", None),  # Empty string
        (None, None),  # None input
    ]

    for url, expected in test_cases:
        result = metric._extract_dataset_id(url)
        assert result == expected, f"Expected {expected} for {url}, got {result}"

    print("✓ Dataset ID extraction working correctly")


def test_popularity_score_calculation():
    """Test popularity score calculation based on downloads and likes."""
    print("Testing popularity score calculation...")

    metric = DatasetQualityMetric()

    # Test various popularity scenarios - updated for new logarithmic scoring
    test_cases = [
        (0, 0, (0.0, 0.0)),           # No popularity
        (1000, 10, (0.6, 0.8)),       # Low popularity (updated)
        (10000, 100, (0.8, 1.0)),     # Medium popularity (updated)
        (100000, 1000, (0.9, 1.0)),   # High popularity
        (1000000, 5000, (0.9, 1.0)),  # Very high popularity
    ]

    for downloads, likes, (min_expected, max_expected) in test_cases:
        score = metric._calculate_popularity_score(downloads, likes)
        assert min_expected <= score <= max_expected, f"Expected {min_expected}-{max_expected} for {downloads} downloads, {likes} likes, got {score}"

    print("✓ Popularity score calculation working correctly")


def test_documentation_score_calculation():
    """Test documentation quality score calculation."""
    print("Testing documentation score calculation...")

    metric = DatasetQualityMetric()

    # Test with no documentation
    score_no_docs = metric._calculate_documentation_score({}, None)
    assert score_no_docs == 0.0, f"Expected 0.0 for no docs, got {score_no_docs}"

    # Test with basic README
    basic_readme = "This is a dataset for machine learning."
    score_basic = metric._calculate_documentation_score({}, basic_readme)
    assert 0.3 < score_basic < 0.6, f"Expected medium score for basic README, got {score_basic}"

    # Test with comprehensive README
    comprehensive_readme = """
# Dataset Description

This dataset contains comprehensive information for machine learning research.

## Usage

Instructions for using this dataset in your projects.

## Citation

Please cite this work if you use it.

## License

This dataset is released under MIT license.
"""
    score_comprehensive = metric._calculate_documentation_score({}, comprehensive_readme)
    assert score_comprehensive > 0.6, f"Expected high score for comprehensive README, got {score_comprehensive}"

    # Test with metadata
    metadata = {
        'task_categories': ['text-classification'],
        'language': ['en'],
        'size_categories': ['100K<n<1M'],
        'source_datasets': ['original']
    }
    score_with_metadata = metric._calculate_documentation_score(metadata, basic_readme)
    assert score_with_metadata > score_basic, "Score should be higher with metadata"

    print("✓ Documentation score calculation working correctly")


def test_metadata_score_calculation():
    """Test metadata completeness score calculation."""
    print("Testing metadata score calculation...")

    metric = DatasetQualityMetric()

    # Test with no metadata
    score_no_metadata = metric._calculate_metadata_score({})
    assert score_no_metadata == 0.0, f"Expected 0.0 for no metadata, got {score_no_metadata}"

    # Test with partial metadata
    partial_metadata = {
        'task_categories': ['text-classification'],
        'language': ['en'],
        'tags': ['nlp', 'classification']
    }
    score_partial = metric._calculate_metadata_score(partial_metadata)
    assert 0.6 <= score_partial <= 0.8, f"Expected medium-high score for partial metadata, got {score_partial}"

    # Test with comprehensive metadata
    full_metadata = {
        'task_categories': ['text-classification'],
        'task_ids': ['topic-classification'],
        'language': ['en'],
        'multilinguality': 'monolingual',
        'size_categories': ['100K<n<1M'],
        'source_datasets': ['original'],
        'tags': ['nlp', 'classification'],
        'created_at': '2023-01-01'
    }
    score_full = metric._calculate_metadata_score(full_metadata)
    assert score_full == 1.0, f"Expected 1.0 for full metadata, got {score_full}"

    print("✓ Metadata score calculation working correctly")


def test_diversity_score_calculation():
    """Test data diversity score calculation."""
    print("Testing diversity score calculation...")

    metric = DatasetQualityMetric()

    # Test with no diversity info
    score_no_diversity = metric._calculate_diversity_score({})
    assert score_no_diversity == 0.0, f"Expected 0.0 for no diversity info, got {score_no_diversity}"

    # Test with language diversity
    multilingual_data = {'language': ['en', 'es', 'fr']}
    score_multilingual = metric._calculate_diversity_score(multilingual_data)
    assert score_multilingual >= 0.3, f"Expected high score for multilingual, got {score_multilingual}"

    # Test with size diversity
    large_size_data = {'size_categories': ['1M<n<10M']}
    score_large = metric._calculate_diversity_score(large_size_data)
    assert score_large >= 0.4, f"Expected high score for large dataset, got {score_large}"

    # Test with task diversity
    multitask_data = {'task_categories': ['text-classification', 'question-answering']}
    score_multitask = metric._calculate_diversity_score(multitask_data)
    assert score_multitask >= 0.2, f"Expected decent score for multitask, got {score_multitask}"

    # Test comprehensive diversity
    diverse_data = {
        'language': ['en', 'es'],
        'size_categories': ['10M<n<100M'],
        'task_categories': ['text-classification', 'ner'],
        'source_datasets': ['original', 'derived']
    }
    score_diverse = metric._calculate_diversity_score(diverse_data)
    assert score_diverse >= 0.8, f"Expected very high score for diverse dataset, got {score_diverse}"

    print("✓ Diversity score calculation working correctly")


def test_licensing_score_calculation():
    """Test licensing and ethics score calculation."""
    print("Testing licensing score calculation...")

    metric = DatasetQualityMetric()

    # Test with no license
    score_no_license = metric._calculate_licensing_score({})
    assert score_no_license == 0.2, f"Expected 0.2 for no license, got {score_no_license}"

    # Test with open license
    open_license_data = {'license': 'mit'}
    score_open = metric._calculate_licensing_score(open_license_data)
    assert score_open >= 0.8, f"Expected high score for open license, got {score_open}"

    # Test with restrictive license
    restrictive_license_data = {'license': 'custom-restrictive'}
    score_restrictive = metric._calculate_licensing_score(restrictive_license_data)
    assert 0.4 <= score_restrictive <= 0.7, f"Expected medium score for restrictive license, got {score_restrictive}"

    # Test with ethics tags
    ethics_data = {
        'license': 'apache-2.0',
        'tags': ['bias-analysis', 'fairness', 'responsible-ai']
    }
    score_ethics = metric._calculate_licensing_score(ethics_data)
    assert score_ethics == 1.0, f"Expected 1.0 for open license + ethics, got {score_ethics}"

    print("✓ Licensing score calculation working correctly")


@patch('app.metrics.implementations.dataset_quality.hf_client')
def test_analyze_single_dataset_success(mock_hf_client):
    """Test successful single dataset analysis."""
    print("Testing successful single dataset analysis...")

    # Mock HuggingFace client responses
    mock_info_result = {
        'dataset_info': {'id': 'squad'},
        'dataset_id': 'squad'
    }

    mock_card_data = {
        'downloads': 50000,
        'likes': 500,
        'task_categories': ['question-answering'],
        'language': ['en'],
        'license': 'cc-by-4.0',
        'size_categories': ['100K<n<1M'],
        'tags': ['nlp', 'qa']
    }

    mock_readme = """
# SQuAD Dataset

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset.

## Usage

Use this dataset for training question answering models.

## Citation

Please cite the original SQuAD paper.

## License

This dataset is licensed under CC-BY-4.0.
"""

    mock_hf_client.get_dataset_info.return_value = mock_info_result
    mock_hf_client.get_dataset_card_data.return_value = mock_card_data
    mock_hf_client.get_dataset_readme.return_value = mock_readme

    metric = DatasetQualityMetric()
    score, details = metric._analyze_single_dataset("squad")

    assert score is not None, "Should return a score"
    assert 0.5 < score <= 1.0, f"Expected high score for good dataset, got {score}"
    assert details['id'] == 'squad', "Should include dataset ID in details"
    assert details['downloads'] == 50000, "Should include download count"
    assert details['likes'] == 500, "Should include like count"

    print("✓ Successful single dataset analysis working correctly")


@patch('app.metrics.implementations.dataset_quality.hf_client')
def test_analyze_single_dataset_failure(mock_hf_client):
    """Test single dataset analysis failure handling."""
    print("Testing single dataset analysis failure...")

    # Mock HuggingFace client returning error
    mock_hf_client.get_dataset_info.return_value = {'error': 'Dataset not found'}

    metric = DatasetQualityMetric()
    score, details = metric._analyze_single_dataset("nonexistent-dataset")

    assert score is None, "Should return None for failed analysis"
    assert details == {}, "Should return empty details for failed analysis"

    print("✓ Single dataset analysis failure handling working correctly")


@patch('app.metrics.implementations.dataset_quality.hf_client')
def test_compute_score_with_datasets(mock_hf_client):
    """Test score computation with multiple datasets."""
    print("Testing score computation with datasets...")

    # Mock successful responses for multiple datasets
    mock_hf_client.get_dataset_info.return_value = {'dataset_info': {}, 'dataset_id': 'test'}
    mock_hf_client.get_dataset_card_data.return_value = {
        'downloads': 10000,
        'likes': 100,
        'license': 'mit',
        'task_categories': ['text-classification']
    }
    mock_hf_client.get_dataset_readme.return_value = "Basic dataset documentation."

    metric = DatasetQualityMetric()

    # Mock the _analyze_single_dataset method to return predictable scores
    with patch.object(metric, '_analyze_single_dataset') as mock_analyze:
        mock_analyze.side_effect = [
            (0.8, {'id': 'dataset1', 'score': 0.8, 'downloads': 10000, 'likes': 100}),
            (0.6, {'id': 'dataset2', 'score': 0.6, 'downloads': 5000, 'likes': 50}),
            (0.9, {'id': 'dataset3', 'score': 0.9, 'downloads': 20000, 'likes': 200})
        ]

        resource = ResourceBundle(
            model_url="https://huggingface.co/bert-base-uncased",
            dataset_urls=[
                "https://huggingface.co/datasets/dataset1",
                "https://huggingface.co/datasets/dataset2",
                "https://huggingface.co/datasets/dataset3"
            ],
            code_urls=[]
        )

        score = metric._compute_score(resource)

        # Should average the three scores plus diversity bonus
        expected_base = (0.8 + 0.6 + 0.9) / 3  # 0.767
        expected_bonus = min(0.1, (3 - 1) * 0.05)  # 0.1
        expected_total = expected_base + expected_bonus

        assert abs(score - expected_total) < 0.01, f"Expected {expected_total}, got {score}"

    print("✓ Score computation with datasets working correctly")


@patch('app.metrics.implementations.dataset_quality.hf_client')
def test_compute_score_with_failed_datasets(mock_hf_client):
    """Test score computation when all datasets fail analysis."""
    print("Testing score computation with failed datasets...")

    metric = DatasetQualityMetric()

    # Mock all dataset analyses failing
    with patch.object(metric, '_analyze_single_dataset') as mock_analyze:
        mock_analyze.return_value = (None, {})

        resource = ResourceBundle(
            model_url="https://huggingface.co/bert-base-uncased",
            dataset_urls=["https://huggingface.co/datasets/broken-dataset"],
            code_urls=[]
        )

        score = metric._compute_score(resource)

        assert score == 0.0, f"Expected 0.0 for all failed analyses, got {score}"

    print("✓ Failed datasets handling working correctly")


def test_computation_notes():
    """Test computation notes generation."""
    print("Testing computation notes...")

    metric = DatasetQualityMetric()

    # Test with no datasets
    resource_no_datasets = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=[],
        code_urls=[]
    )
    notes_no_datasets = metric._get_computation_notes(resource_no_datasets)
    assert "No datasets found" in notes_no_datasets

    # Test with datasets (mock analysis details)
    metric._analysis_details = {
        'analyzed_count': 2,
        'total_count': 3,
        'average_score': 0.75,
        'dataset_details': [
            {'id': 'squad', 'score': 0.9, 'downloads': 50000, 'likes': 500},
            {'id': 'glue', 'score': 0.6, 'downloads': 20000, 'likes': 200}
        ]
    }

    resource_with_datasets = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=["squad", "glue", "broken"],
        code_urls=[]
    )
    notes_with_datasets = metric._get_computation_notes(resource_with_datasets)
    assert "Analyzed 2/3 datasets" in notes_with_datasets
    assert "Average dataset quality score: 0.75" in notes_with_datasets
    assert "Top datasets:" in notes_with_datasets

    print("✓ Computation notes working correctly")


@patch('app.metrics.implementations.dataset_quality.hf_client')
def test_metric_computation(mock_hf_client):
    """Test full metric computation with ResourceBundle."""
    print("Testing full metric computation...")

    # Mock HuggingFace client for successful response
    mock_hf_client.get_dataset_info.return_value = {
        'dataset_info': {'id': 'squad'},
        'dataset_id': 'squad'
    }
    mock_hf_client.get_dataset_card_data.return_value = {
        'downloads': 10000,
        'likes': 100,
        'license': 'mit',
        'task_categories': ['question-answering']
    }
    mock_hf_client.get_dataset_readme.return_value = "Dataset documentation."

    metric = DatasetQualityMetric()
    resource = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=["https://huggingface.co/datasets/squad"],
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


def run_all_tests():
    """Run all dataset quality metric tests."""
    print("=== Dataset Quality Metric Tests ===\n")

    try:
        test_dataset_quality_metric_registration()
        test_no_datasets()
        test_hf_client_unavailable()
        test_extract_dataset_id()
        test_popularity_score_calculation()
        test_documentation_score_calculation()
        test_metadata_score_calculation()
        test_diversity_score_calculation()
        test_licensing_score_calculation()
        test_analyze_single_dataset_success()
        test_analyze_single_dataset_failure()
        test_compute_score_with_datasets()
        test_compute_score_with_failed_datasets()
        test_computation_notes()
        test_metric_computation()

        print("\n All dataset quality metric tests passed!")
        return True

    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)