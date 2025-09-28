import time
import sys
import os
import Url_Parser
import Installer
import Tester
from pathlib import Path
from typing import Dict, Any, List
import json

# Suppress HuggingFace symlink warning on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Suppress transformers progress bars and warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Also suppress at the library level if already imported
try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

def install_dependencies() -> int:
    return Installer.run()
    

def run_tests(pytest_args=None) -> int:
    if pytest_args is None:
        try:
            i = sys.argv.index("test")
            pytest_args = sys.argv[i+1:]   # everything after 'test'
        except ValueError:
            pytest_args = []
    return Tester.run(pytest_args=pytest_args)


def process_urls(file_path: Path) -> int:
    # Add src to path for imports
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from app.dataset_tracker import DatasetTracker

    # Initialize dataset tracker for this session
    dataset_tracker = DatasetTracker()

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Check if line contains comma-separated URLs (grouped evaluation)
                if ',' in line:
                    process_url_group(line, dataset_tracker)
                else:
                    # Single URL - traditional processing
                    process_url(line)
    return 0

def process_url_group(url_group: str, dataset_tracker=None) -> int:
    """
    Run metrics evaluation for a group of related URLs in CSV format.

    Format: <code_link>, <dataset_link>, <model_link>
    Where code_link and dataset_link can be blank.

    Args:
        url_group: Comma-separated line with code, dataset, model URLs

    Returns:
        0 on success, non-zero on error
    """
    try:
        # Parse CSV format: code_link, dataset_link, model_link
        parts = url_group.split(',')
        if len(parts) != 3:
            print(f"Error: Expected 3 comma-separated values, got {len(parts)}", file=sys.stderr)
            return 1

        code_url = parts[0].strip() if parts[0].strip() else None
        dataset_url = parts[1].strip() if parts[1].strip() else None
        model_url = parts[2].strip() if parts[2].strip() else None

        if not model_url:
            print("Error: Model URL (third field) is required", file=sys.stderr)
            return 1

        # Handle dataset tracking and inference
        if dataset_url:
            # Dataset explicitly provided - add to tracker
            if dataset_tracker:
                dataset_tracker.add_dataset(dataset_url)
            dataset_urls = [dataset_url]
        else:
            # Dataset missing - try to infer from model README
            if dataset_tracker:
                inferred_dataset = dataset_tracker.infer_dataset(model_url)
                dataset_urls = [inferred_dataset] if inferred_dataset else []
            else:
                dataset_urls = []

        # Prepare code URL list
        code_urls = [code_url] if code_url else []

        # Run grouped evaluation - all entries in CSV are model evaluations
        result = run_metrics(
            model_url=model_url,
            dataset_urls=dataset_urls,
            code_urls=code_urls,
            category="MODEL"
        )
        print(result)
        return 0

    except Exception as e:
        print(f"Error evaluating URL group {url_group}: {e}", file=sys.stderr)
        return 1



def process_url(url: str) -> int:
    """
    Run metrics evaluation for a single URL.
    Treats all single URLs as model URLs for evaluation.

    Args:
        url: Single URL to evaluate as a model

    Returns:
        0 on success, non-zero on error
    """
    try:
        result = run_metrics(model_url=url)
        print(result)
        return 0
    except Exception as e:
        print(f"Error evaluating URL {url}: {e}", file=sys.stderr)
        return 1


def run_metrics(model_url: str, dataset_urls: List[str] = None, code_urls: List[str] = None, weights: Dict[str, float] = None, use_cache: bool = False, category: str = "MODEL") -> str:
    """
    Run the metrics system on a given model with optional dataset and code URLs.

    Args:
        model_url: The primary model URL (required)
        dataset_urls: Optional list of dataset URLs
        code_urls: Optional list of code URLs
        weights: Optional custom weights profile for metrics
        use_cache: Whether to use database caching (default: True)

    Returns:
        NDJSON string with metric results
    """
    # Add src to path for imports
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    try:
        # Import metrics system and database
        from app.metrics.registry import all_metrics
        from app.metrics.engine import run_bundle
        from app.metrics.base import ResourceBundle
        from app.database import get_database

        # Set defaults
        if dataset_urls is None:
            dataset_urls = []
        if code_urls is None:
            code_urls = []

        # Check cache first if caching is enabled
        if use_cache:
            db = get_database()
            cached_result = db.get_cached_result(model_url, dataset_urls, code_urls)
            if cached_result:
                return cached_result

        # Default weights profile if not provided
        if weights is None:
            weights = {
                "license": 0.15,
                "ramp_up_time": 0.15,
                "bus_factor": 0.10,
                "performance_claims": 0.20,
                "size_score": 0.15,
                "dataset_and_code_score": 0.10,
                "dataset_quality": 0.10,
                "code_quality": 0.05
            }

        # Extract model_id from URL for metrics that need it
        model_id = model_url.split("/")[-1] if model_url else ""

        # Enhance URLs with LLM-based metadata extraction if lists are empty
        enhanced_dataset_urls = dataset_urls[:]
        enhanced_code_urls = code_urls[:]

        if not enhanced_dataset_urls or not enhanced_code_urls:
            try:
                from app.metrics.llm_metadata_extractor import metadata_extractor
                extracted_metadata = metadata_extractor.extract_metadata(model_id)

                # Add discovered URLs if we don't already have them
                if not enhanced_dataset_urls and extracted_metadata.dataset_urls:
                    enhanced_dataset_urls.extend(extracted_metadata.dataset_urls)

                if not enhanced_code_urls and extracted_metadata.code_urls:
                    enhanced_code_urls.extend(extracted_metadata.code_urls)

                # Also try to infer dataset URLs from dataset names
                if not enhanced_dataset_urls and extracted_metadata.training_datasets:
                    for dataset_name in extracted_metadata.training_datasets:
                        # Convert common dataset names to HuggingFace URLs
                        if '/' not in dataset_name:  # Simple name, not a path
                            hf_url = f"https://huggingface.co/datasets/{dataset_name.lower()}"
                            enhanced_dataset_urls.append(hf_url)

            except Exception:
                # If metadata extraction fails, continue with original URLs
                pass

        # Create resource bundle
        resource_bundle = ResourceBundle(
            model_url=model_url,
            dataset_urls=enhanced_dataset_urls,
            code_urls=enhanced_code_urls,
            model_id=model_id
        )

        # Get all available metrics
        metric_factories = all_metrics()
        if not metric_factories:
            raise RuntimeError("No metrics registered")

        metrics = [factory() for factory in metric_factories.values()]

        # Run the metrics engine
        result = run_bundle(resource_bundle, metrics, weights, category)

        # Cache the result if caching is enabled
        if use_cache:
            db = get_database()
            db.cache_result(model_url, result, dataset_urls, code_urls)

        return result

    except Exception as e:
        # Return error in NDJSON format
        error_result = {
            "URL": model_url,
            "Error": str(e),
            "NetScore": 0.0,
            "NetScore_Latency": 0
        }
        return json.dumps(error_result, separators=(',', ':'))

