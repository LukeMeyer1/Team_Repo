# Trustworthy Model Re-use System

A comprehensive system for evaluating machine learning models based on trustworthiness metrics including licensing, documentation quality, maintainer health, performance claims, and more.

## Team Members
- Simon Greenaway
- Luke Miller
- Ayddan Hartle
- Luke Meyer

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Metrics System](#metrics-system)
- [Testing](#testing)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

This system provides automated evaluation of machine learning models across multiple trustworthiness dimensions. It analyzes models, their associated datasets, and code repositories to generate comprehensive trust scores.

### Key Features
- **Multi-dimensional Evaluation**: 8 different metrics covering licensing, documentation, maintainer health, performance claims, and more
- **Automated Analysis**: Processes models from URLs or batch files
- **Caching System**: Database-backed caching for efficient re-evaluation
- **Extensible Architecture**: Plugin-based metrics system for easy expansion
- **Comprehensive Testing**: Full test suite with coverage tracking

## Architecture

### Core Components
```
run (CLI entry point)
├── Orchestrator.py (Main controller)
│   ├── Installer.py (Dependency management)
│   ├── Tester.py (Test runner with coverage)
│   └── Url_Parser.py (URL processing)
└── src/app/
    ├── metrics/ (Evaluation system)
    ├── database/ (Caching layer)
    ├── integrations/ (External APIs)
    └── dataset_tracker.py (Dataset inference)
```

### Data Flow
1. **Input**: URLs or batch files containing model/dataset/code references
2. **Processing**: Extract metadata, analyze documentation, evaluate metrics
3. **Scoring**: Apply weighted scoring across all metrics
4. **Output**: NDJSON format with detailed scores and timing information

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git (for repository analysis)

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd team_repo

# Install dependencies
python run install

# Verify installation
python run test
```

### Manual Installation
```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Required Packages
- `huggingface_hub[hf_xet]` - HuggingFace API integration
- `transformers` - Model metadata extraction
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-json-report` - Machine-readable test reports

## Usage

### Command Line Interface

#### Single Model Evaluation
```bash
# Evaluate a single model
echo "https://huggingface.co/microsoft/DialoGPT-medium" > model.txt
python run model.txt
```

#### Batch Processing
```bash
# Create a file with multiple URLs
cat > models.txt << EOF
https://huggingface.co/microsoft/DialoGPT-medium
https://huggingface.co/facebook/bart-large
https://huggingface.co/google/flan-t5-base
EOF

# Process all models
python run models.txt
```

#### Grouped Evaluation (CSV Format)
```bash
# Format: code_url,dataset_url,model_url
cat > grouped_models.txt << EOF
https://github.com/microsoft/DialoGPT,https://huggingface.co/datasets/daily_dialog,https://huggingface.co/microsoft/DialoGPT-medium
,https://huggingface.co/datasets/squad,https://huggingface.co/facebook/bart-large
https://github.com/google-research/text-to-text-transfer-transformer,,https://huggingface.co/google/flan-t5-base
EOF

python run grouped_models.txt
```

#### Testing and Coverage
```bash
# Run all tests
python run test

# Run with coverage report
python run test --cov=src --cov-report=html

# Run specific test file
python run test tests/test_metrics.py

# Run with verbose output
python run test -v
```

### Output Format

The system outputs NDJSON (newline-delimited JSON) with the following structure:
```json
{
    "URL": "https://huggingface.co/microsoft/DialoGPT-medium",
    "NetScore": 0.75,
    "NetScore_Latency": 2.34,
    "license": 0.8,
    "license_Latency": 0.12,
    "ramp_up_time": 0.9,
    "ramp_up_time_Latency": 0.45,
    "bus_factor": 0.6,
    "bus_factor_Latency": 0.23,
    "performance_claims": 0.85,
    "performance_claims_Latency": 0.67,
    "size_score": 0.7,
    "size_score_Latency": 0.15,
    "dataset_and_code_score": 0.8,
    "dataset_and_code_score_Latency": 0.34,
    "dataset_quality": 0.75,
    "dataset_quality_Latency": 0.28,
    "code_quality": 0.65,
    "code_quality_Latency": 0.41
}
```

## Configuration

### Environment Variables

#### Logging Configuration
```bash
# Log file path (must exist and be writable)
export LOG_FILE="/path/to/logfile.log"

# Log verbosity level
export LOG_LEVEL=1  # 0=silent, 1=info, 2=debug
```

#### API Tokens (Optional)
```bash
# GitHub token for enhanced repository analysis
export GITHUB_TOKEN="your_github_token_here"

# HuggingFace token for private models
export HUGGINGFACE_TOKEN="your_hf_token_here"
```

#### System Configuration
```bash
# Disable HuggingFace warnings
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
```

### Logging System
This project includes a configurable logging system to help track errors, warnings, and debug messages. Logging is controlled entirely through environment variables.

**LOG_FILE**: Path to the file where logs will be written
- Example: `/tmp/myapp.log` (Linux/macOS) or `C:\temp\myapp.log` (Windows)
- Default: `app.log` in the project root if not set
- **Important**: File must exist and be writable

**LOG_LEVEL**: Verbosity of the logs
- `0` → Silent (no logs written)
- `1` → Informational messages (INFO, WARNING, ERROR, CRITICAL)
- `2` → Debug messages (DEBUG and above)
- Default: 0 (silent)

## Metrics System Implementation

```
src/app/metrics/
├── implementations/          # Individual metric implementations
│   ├── license.py           # License compatibility (LGPLv2.1)
│   ├── documentation.py     # Ramp-up time / docs quality  
│   ├── maintainer.py        # Bus factor / contributor health
│   ├── performance.py       # Performance claims evidence
│   ├── size.py              # Device compatibility scoring
│   ├── dataset.py           # Dataset quality metrics
│   └── code_quality.py      # Code maintainability analysis
├── base.py                  # Core data structures (ResourceBundle, MetricResult)
├── base_metric.py           # Abstract base class
├── registry.py              # Auto-discovery system (@register decorator)
└── engine.py                # Parallel orchestration + NDJSON output
```

### Metrics Implemented

| Metric | Purpose | Scoring Focus |
|--------|---------|---------------|
| `license` | LGPLv2.1 compatibility analysis | License clarity, commercial use permissions |
| `ramp_up_time` | Documentation quality assessment | README completeness, examples, tutorials |
| `bus_factor` | Maintainer health evaluation | Contributor diversity, project sustainability |
| `performance_claims` | Benchmark evidence validation | Performance metrics credibility |
| `size_score` | Device compatibility analysis | Model size vs deployment targets |
| `dataset_and_code_score` | Ecosystem completeness | Linked datasets and code repositories |
| `dataset_quality` | Dataset reliability metrics | Data quality, popularity, maintenance |
| `code_quality` | Repository maintainability | Code style, testing, CI/CD practices |

### Metrics Directory Structure
```
src/app/metrics/
├── base.py                  # Core data structures (ResourceBundle, MetricResult)
├── base_metric.py          # Abstract base class for all metrics
├── registry.py             # Auto-discovery system (@register decorator)
├── engine.py               # Parallel orchestration + NDJSON output
├── bus_factor.py           # Maintainer health evaluation
├── code_quality.py         # Repository maintainability analysis
├── dataset_and_code.py     # Ecosystem completeness scoring
├── dataset_quality.py      # Dataset reliability metrics
├── license_metric.py       # License compatibility analysis
├── llm_metadata_extractor.py # LLM-powered metadata extraction
├── performance_claims.py   # Benchmark evidence validation
├── ramp_up_time.py        # Documentation quality assessment
└── size.py                # Device compatibility analysis
```

### Metrics Implemented

| Metric | Weight | Purpose | Scoring Focus |
|--------|--------|---------|---------------|
| `license` | 15% | LGPLv2.1 compatibility analysis | License clarity, commercial use permissions |
| `ramp_up_time` | 15% | Documentation quality assessment | README completeness, examples, tutorials |
| `bus_factor` | 10% | Maintainer health evaluation | Contributor diversity, project sustainability |
| `performance_claims` | 20% | Benchmark evidence validation | Performance metrics credibility |
| `size_score` | 15% | Device compatibility analysis | Model size vs deployment targets |
| `dataset_and_code_score` | 10% | Ecosystem completeness | Linked datasets and code repositories |
| `dataset_quality` | 10% | Dataset reliability metrics | Data quality, popularity, maintenance |
| `code_quality` | 5% | Repository maintainability | Code style, testing, CI/CD practices |

### Adding New Metrics

1. **Create metric file** in `src/app/metrics/`:
```python
from app.metrics.base import ResourceBundle
from app.metrics.registry import register
from app.metrics.base_metric import BaseMetric

@register("my_new_metric")
class MyNewMetric(BaseMetric):
    name = "my_new_metric"

    def _compute_score(self, resource: ResourceBundle) -> float:
        # Your scoring logic here
        return 0.5  # Return score between 0-1

    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        return f"Analysis notes for {resource.model_url}"
```

2. **Update weights** in `Orchestrator.py` line 241-250
3. **Add tests** in `tests/test_metrics.py`

## Testing

### Running Tests
```bash
# Run all tests with coverage
python run test

# Run specific test categories
python run test tests/test_metrics.py        # Metrics system
python run test tests/test_database_basic.py # Database functionality
python run test tests/test_orchestrator.py   # Core orchestration

# Generate HTML coverage report
python run test --cov=src --cov-report=html
open htmlcov/index.html  # View detailed coverage
```

### Test Structure
```
tests/
├── conftest.py                 # Shared test fixtures
├── test_metrics.py            # Core metrics system tests
├── test_metrics_fast.py       # Fast metric validation tests
├── test_orchestrator.py       # Orchestrator functionality
├── test_database_basic.py     # Database operations
├── test_dataset_quality.py    # Dataset metric specific tests
├── test_code_quality.py       # Code quality metric tests
├── test_size.py              # Size metric tests
└── test_cli_*.py             # Command-line interface tests
```

### Coverage Requirements
- Minimum 70% line coverage across all modules
- Critical components (Orchestrator, metrics engine) should have 90%+ coverage
- All new metrics must include comprehensive tests

## Development

### Project Structure
```
team_repo/
├── run                        # CLI entry point (executable)
├── Orchestrator.py           # Main controller and workflow
├── Installer.py              # Dependency installation
├── Tester.py                 # Test runner with reporting
├── Url_Parser.py             # URL processing utilities
├── Logger.py                 # Logging configuration
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project metadata
├── mypy.ini                 # Type checking configuration
├── .flake8                  # Code style configuration
├── src/app/                 # Main application code
│   ├── metrics/             # Metrics evaluation system
│   ├── database/            # Caching and persistence
│   ├── integrations/        # External API clients
│   └── dataset_tracker.py   # Dataset inference system
└── tests/                   # Test suite
```

### Development Workflow
1. **Setup development environment**:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Run quality checks**:
```bash
python run test                    # Run tests
python -m flake8 src/             # Code style
python -m mypy src/               # Type checking
```

3. **Add new features**:
   - Create feature branch
   - Implement changes with tests
   - Ensure coverage doesn't decrease
   - Update documentation

### Code Style Guidelines
- Follow PEP 8 style guide
- Use type hints for all public functions
- Maximum line length: 100 characters
- Use descriptive variable and function names
- Add docstrings for all public methods

### Database Schema
The system uses SQLite for caching with the following structure:
```sql
CREATE TABLE metrics_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_url TEXT NOT NULL,
    dataset_urls TEXT,
    code_urls TEXT,
    result TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_url, dataset_urls, code_urls)
);
```

## Troubleshooting

### Common Issues

#### Installation Problems
**Issue**: `ModuleNotFoundError` when running
```bash
# Solution: Ensure dependencies are installed
python run install
# Or manually:
pip install -r requirements.txt
```

**Issue**: Permission denied on log file
```bash
# Solution: Create writable log file or unset LOG_FILE
touch app.log
chmod 666 app.log
# Or disable logging:
unset LOG_FILE
```

#### Runtime Errors
**Issue**: Network timeouts when fetching model data
```bash
# Solution: Check internet connection and try again
# Models are fetched from HuggingFace Hub - ensure access
```

**Issue**: "No metrics registered" error
```bash
# Solution: Verify metrics are properly imported
python -c "from src.app.metrics.registry import all_metrics; print(list(all_metrics().keys()))"
```

**Issue**: Database lock errors
```bash
# Solution: Remove cache database to reset
rm -f src/app/database/metrics_cache.db
```

#### Testing Issues
**Issue**: Tests fail with import errors
```bash
# Solution: Ensure Python path includes src/
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python run test
```

**Issue**: Coverage reports missing files
```bash
# Solution: Run tests from project root with proper coverage config
python run test --cov=src --cov-report=term-missing
```

### Performance Optimization
- **Enable caching**: Use `use_cache=True` in `run_metrics()` calls
- **Batch processing**: Process multiple models in single runs
- **Parallel evaluation**: Metrics run in parallel automatically
- **Network optimization**: Set appropriate timeouts for API calls

### Debugging
1. **Enable debug logging**:
```bash
export LOG_LEVEL=2
export LOG_FILE=debug.log
python run test
```

2. **Check metric computation**:
```python
# In Python console
from src.app.metrics.registry import all_metrics
from src.app.metrics.base import ResourceBundle

metrics = all_metrics()
bundle = ResourceBundle(model_url="https://huggingface.co/gpt2")
for name, factory in metrics.items():
    metric = factory()
    score = metric.compute_score(bundle)
    print(f"{name}: {score}")
```

## API Reference

### Core Functions

#### `Orchestrator.run_metrics()`
```python
def run_metrics(
    model_url: str,
    dataset_urls: List[str] = None,
    code_urls: List[str] = None,
    weights: Dict[str, float] = None,
    use_cache: bool = False,
    category: str = "MODEL"
) -> str:
    """
    Run metrics evaluation on a model.

    Args:
        model_url: Primary model URL (required)
        dataset_urls: Associated dataset URLs
        code_urls: Associated code repository URLs
        weights: Custom metric weights (defaults provided)
        use_cache: Enable database caching
        category: Evaluation category (MODEL, DATASET, etc.)

    Returns:
        NDJSON string with metric results
    """
```

#### `Orchestrator.process_urls()`
```python
def process_urls(file_path: Path) -> int:
    """
    Process URLs from a file.

    Args:
        file_path: Path to file containing URLs or CSV data

    Returns:
        0 on success, non-zero on error
    """
```

### Metric Base Classes

#### `BaseMetric`
```python
class BaseMetric:
    name: str  # Metric identifier

    def compute_score(self, resource: ResourceBundle) -> MetricResult:
        """Compute metric score with timing."""

    def _compute_score(self, resource: ResourceBundle) -> float:
        """Override this method in subclasses."""

    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        """Override to provide computation details."""
```

#### `ResourceBundle`
```python
@dataclass
class ResourceBundle:
    model_url: str
    dataset_urls: List[str] = field(default_factory=list)
    code_urls: List[str] = field(default_factory=list)
    model_id: str = ""
```

### Database API

#### `get_database()`
```python
def get_database() -> Database:
    """Get database instance with caching support."""

class Database:
    def cache_result(self, model_url: str, result: str,
                    dataset_urls: List[str] = None,
                    code_urls: List[str] = None) -> None:
        """Cache evaluation result."""

    def get_cached_result(self, model_url: str,
                         dataset_urls: List[str] = None,
                         code_urls: List[str] = None) -> Optional[str]:
        """Retrieve cached result."""
```
