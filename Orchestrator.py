import time
import sys
import os
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
    import sys, json
    from pathlib import Path

    # Preserve current behavior: grab args after "test" if pytest_args not provided
    if pytest_args is None:
        try:
            i = sys.argv.index("test")
            pytest_args = sys.argv[i+1:]   # everything after 'test'
        except ValueError:
            pytest_args = []

    # Ensure we get machine-readable outputs to parse
    extra = []

    # Coverage: default to src if user didn't specify --cov
    if not any(a == "--cov" or a.startswith("--cov=") for a in pytest_args):
        extra += ["--cov=src"]  # change 'src' if your package path differs

    # Always produce a JSON coverage artifact
    if not any(a.startswith("--cov-report=") and "json:" in a for a in pytest_args):
        extra += ["--cov-report=json:coverage.json"]
    # Keep terminal coverage too (optional)
    if not any(a.startswith("--cov-report=") and ("term" in a or "term-missing" in a) for a in pytest_args):
        extra += ["--cov-report=term-missing"]

    # Emit a JSON pytest report (requires pytest-json-report)
    if "--json-report" not in pytest_args:
        extra += ["--json-report"]
    if not any(a.startswith("--json-report-file=") for a in pytest_args):
        extra += ["--json-report-file=.report.json"]

    # Run tests via your existing Tester
    exit_code = Tester.run(pytest_args=pytest_args + extra)

    # Parse and print: "X/Y test cases passed. Z% line coverage achieved."
    try:
        rpt = json.loads(Path(".report.json").read_text())
        cov = json.loads(Path("coverage.json").read_text())

        passed = int(rpt["summary"]["passed"])
        total  = int(rpt["summary"]["total"])

        # pytest-cov JSON schema: totals.percent_covered (float)
        pct = float(cov["totals"]["percent_covered"])
        pct_str = f"{pct:.2f}".rstrip("0").rstrip(".")

        print(f"{passed}/{total} test cases passed. {pct_str}% line coverage achieved.")
    except FileNotFoundError:
        print("Note: missing .report.json or coverage.json. "
              "Install/enable plugins: pytest-json-report and pytest-cov.")
    except Exception as e:
        print(f"Summary generation failed: {e}")

    return exit_code


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

        # Clean up model URL (remove /tree/main etc.)
        model_url = model_url.rstrip('/').split('/tree/')[0].split('/blob/')[0]

        # Extract model_id for LLM inference
        model_id = None
        if model_url and "huggingface.co" in model_url:
            parts = model_url.rstrip("/").split("/")
            if len(parts) >= 2:
                model_id = "/".join(parts[-2:])  # Get org/model

        # Use LLM metadata extractor to infer missing URLs
        inferred_dataset_urls = []
        inferred_code_urls = []

        if model_id and (not dataset_url or not code_url):
            try:
                # Import LLM metadata extractor
                ROOT = Path(__file__).resolve().parent
                SRC = ROOT / "src"
                if str(SRC) not in sys.path:
                    sys.path.insert(0, str(SRC))

                from app.metrics.llm_metadata_extractor import metadata_extractor

                # Extract metadata to infer missing URLs
                extracted_metadata = metadata_extractor.extract_metadata(model_id)

                if not dataset_url and extracted_metadata:
                    # Try dataset URLs first, then training datasets
                    inferred_dataset_urls = extracted_metadata.dataset_urls or []
                    if not inferred_dataset_urls and extracted_metadata.training_datasets:
                        # Convert dataset names to URLs if possible
                        known_mappings = {
                            'librispeech': 'https://huggingface.co/datasets/librispeech_asr',
                            'common_voice': 'https://huggingface.co/datasets/common_voice',
                            'squad': 'https://huggingface.co/datasets/squad',
                            'glue': 'https://huggingface.co/datasets/glue',
                        }
                        for dataset_name in extracted_metadata.training_datasets:
                            mapping_key = dataset_name.lower().replace('-', '_')
                            if mapping_key in known_mappings:
                                inferred_dataset_urls.append(known_mappings[mapping_key])

                if not code_url and extracted_metadata:
                    inferred_code_urls = extracted_metadata.code_urls or []

                # print(f"LLM inference for {model_id}: datasets={inferred_dataset_urls}, code={inferred_code_urls}", file=sys.stderr)

            except Exception as e:
                pass  # Silently continue if LLM inference fails

        # Handle dataset URLs - explicit or inferred
        if dataset_url:
            # Dataset explicitly provided - add to tracker
            if dataset_tracker:
                dataset_tracker.add_dataset(dataset_url)
            dataset_urls = [dataset_url]
        elif inferred_dataset_urls:
            # Use LLM-inferred dataset URLs
            dataset_urls = inferred_dataset_urls
            if dataset_tracker:
                for url in dataset_urls:
                    dataset_tracker.add_dataset(url)
        else:
            # Fallback to old dataset tracker inference
            if dataset_tracker:
                inferred_dataset = dataset_tracker.infer_dataset(model_url)
                dataset_urls = [inferred_dataset] if inferred_dataset else []
            else:
                dataset_urls = []

        # Handle code URLs - explicit or inferred
        if code_url:
            code_urls = [code_url]
        elif inferred_code_urls:
            # Use LLM-inferred code URLs
            code_urls = inferred_code_urls
        else:
            code_urls = []

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
    Uses LLM inference to discover missing dataset/code URLs.

    Args:
        url: Single URL to evaluate as a model

    Returns:
        0 on success, non-zero on error
    """
    try:
        # Clean up model URL (remove /tree/main etc.)
        model_url = url.rstrip('/').split('/tree/')[0].split('/blob/')[0]

        # Extract model_id for LLM inference
        model_id = None
        if model_url and "huggingface.co" in model_url:
            parts = model_url.rstrip("/").split("/")
            if len(parts) >= 2:
                model_id = "/".join(parts[-2:])  # Get org/model

        # Use LLM metadata extractor to infer dataset and code URLs
        dataset_urls = []
        code_urls = []

        if model_id:
            try:
                # Import LLM metadata extractor
                ROOT = Path(__file__).resolve().parent
                SRC = ROOT / "src"
                if str(SRC) not in sys.path:
                    sys.path.insert(0, str(SRC))

                from app.metrics.llm_metadata_extractor import metadata_extractor

                # Extract metadata to infer URLs
                extracted_metadata = metadata_extractor.extract_metadata(model_id)

                if extracted_metadata:
                    # Get dataset URLs
                    dataset_urls = extracted_metadata.dataset_urls or []
                    if not dataset_urls and extracted_metadata.training_datasets:
                        # Convert dataset names to URLs if possible
                        known_mappings = {
                            'librispeech': 'https://huggingface.co/datasets/librispeech_asr',
                            'common_voice': 'https://huggingface.co/datasets/common_voice',
                            'squad': 'https://huggingface.co/datasets/squad',
                            'glue': 'https://huggingface.co/datasets/glue',
                        }
                        for dataset_name in extracted_metadata.training_datasets:
                            mapping_key = dataset_name.lower().replace('-', '_')
                            if mapping_key in known_mappings:
                                dataset_urls.append(known_mappings[mapping_key])

                    # Get code URLs
                    code_urls = extracted_metadata.code_urls or []

            except Exception as e:
                pass  # Silently continue if LLM inference fails

        result = run_metrics(
            model_url=model_url,
            dataset_urls=dataset_urls,
            code_urls=code_urls
        )
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
        # For HuggingFace URLs, we need org/model format, not just the model name
        if model_url and "huggingface.co" in model_url:
            parts = model_url.rstrip("/").split("/")
            if len(parts) >= 2:
                model_id = "/".join(parts[-2:])  # Get org/model
            else:
                model_id = parts[-1] if parts else ""
        else:
            model_id = model_url.split("/")[-1] if model_url else ""

        # Create resource bundle
        resource_bundle = ResourceBundle(
            model_url=model_url,
            dataset_urls=dataset_urls,
            code_urls=code_urls,
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
        return json.dumps(error_result)

