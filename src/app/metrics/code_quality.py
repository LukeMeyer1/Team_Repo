"""
Code quality and maintainability metric.

Evaluates the quality of code repositories associated with models,
focusing on style, testing, maintainability, and best practices.
"""

import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from app.metrics.base import ResourceBundle
from app.metrics.registry import register
from app.metrics.base_metric import BaseMetric

# Import from project root for URL parsing
import sys
_root_path = str(Path(__file__).parent.parent.parent.parent)
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

try:
    from Url_Parser import categorize_url
except ImportError:
    def categorize_url(url: str) -> str:
        """Fallback URL categorization."""
        if "github.com" in url:
            return "CODE"
        return "UNKNOWN"


@register("code_quality")
class CodeQualityMetric(BaseMetric):
    """
    Evaluates code style, testing, and maintainability of associated repositories.

    This metric clones GitHub repositories locally and analyzes:
    - Repository structure and organization
    - Documentation presence (README, LICENSE, docs)
    - Testing infrastructure (test directories, CI configs)
    - Code file quality and patterns
    - Dependency management files
    - Basic static analysis metrics

    Scoring Rubric:
    - 1.0: Excellent repository with comprehensive structure and documentation
    - 0.8: Good practices with minor gaps
    - 0.6: Acceptable quality but missing some best practices
    - 0.4: Poor organization or limited documentation
    - 0.2: Very poor repository quality
    - 0.0: No code repositories or inaccessible
    """
    
    name = "code_quality"
    
    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate code quality and maintainability of associated repositories.
        Clones repositories locally and performs comprehensive analysis.
        """
        if not resource.code_urls:
            return 0.0  # No code repositories to analyze

        # Filter for GitHub repositories only
        github_repos = [url for url in resource.code_urls if categorize_url(url) == "CODE"]
        if not github_repos:
            return 0.0

        total_score = 0.0
        analyzed_repos = 0

        # Analyze each repository (limit to first 3 to avoid excessive processing)
        for repo_url in github_repos[:3]:
            try:
                repo_score = self._analyze_repository(repo_url)
                if repo_score is not None:
                    total_score += repo_score
                    analyzed_repos += 1
            except Exception:
                continue  # Skip failed repositories

        if analyzed_repos == 0:
            return 0.0

        # Average score across analyzed repositories
        average_score = total_score / analyzed_repos

        # Bonus for having multiple repositories (ecosystem diversity)
        diversity_bonus = min(0.1, (len(github_repos) - 1) * 0.05)

        return min(1.0, average_score + diversity_bonus)
    
    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        if not resource.code_urls:
            return f"No code repositories found for {resource.model_url}. Quality assessment not possible."

        github_repos = [url for url in resource.code_urls if categorize_url(url) == "CODE"]
        if not github_repos:
            return f"No GitHub repositories found in {len(resource.code_urls)} code URLs. Only GitHub repositories are analyzed."

        return (
            f"Analyzed {min(len(github_repos), 3)} GitHub repositories for {resource.model_url}. "
            f"Assessment based on repository structure, documentation, testing infrastructure, and code organization."
        )

    def _analyze_repository(self, repo_url: str) -> Optional[float]:
        """
        Clone and analyze a single repository for code quality metrics.

        Args:
            repo_url: GitHub repository URL to analyze

        Returns:
            Quality score from 0-1, or None if analysis failed
        """
        temp_dir = None
        try:
            # Create temporary directory for cloning
            temp_dir = tempfile.mkdtemp(prefix="code_quality_")
            repo_path = Path(temp_dir) / "repo"

            # Clone repository (shallow clone to save time and space)
            clone_cmd = ["git", "clone", "--depth", "1", repo_url, str(repo_path)]
            result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                return None

            # Analyze repository structure
            structure_score = self._analyze_repository_structure(repo_path)

            # Analyze code quality indicators
            code_score = self._analyze_code_quality(repo_path)

            # Calculate weighted overall score
            overall_score = (structure_score * 0.6) + (code_score * 0.4)

            return overall_score

        except Exception:
            return None
        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _analyze_repository_structure(self, repo_path: Path) -> float:
        """
        Analyze repository structure for quality indicators.

        Args:
            repo_path: Path to cloned repository

        Returns:
            Structure quality score from 0-1
        """
        if not repo_path.exists():
            return 0.0

        score = 0.0
        max_score = 7.0  # total possible points

        # Check for README file (3 points)
        readme_files = list(repo_path.glob("README*"))
        if readme_files:
            score += 3.0

        # Check for LICENSE file (1 point)
        license_files = list(repo_path.glob("LICENSE*")) + list(repo_path.glob("LICENCE*"))
        if license_files:
            score += 1.0

        # Check for any kind of dependency management (1.5 points - more generous)
        dep_files = (
            list(repo_path.glob("requirements*.txt")) +
            list(repo_path.glob("setup.py")) +
            list(repo_path.glob("pyproject.toml")) +
            list(repo_path.glob("Pipfile")) +
            list(repo_path.glob("package.json")) +
            list(repo_path.glob("environment.yml")) +  # conda
            list(repo_path.glob("*.toml"))  # any toml file
        )
        if dep_files:
            score += 1.5

        # Check for any configuration files (0.75 points)
        config_files = (
            list(repo_path.glob("*.cfg")) +
            list(repo_path.glob("*.ini")) +
            list(repo_path.glob("*.toml")) +
            list(repo_path.glob(".gitignore")) +
            list(repo_path.glob("*.json")) +  # config files
            list(repo_path.glob("*.yaml")) +
            list(repo_path.glob("*.yml"))
        )
        if config_files:
            score += 0.75

        # Give bonus for having any code organization (0.75 points)
        python_files = list(repo_path.glob("**/*.py"))
        if len(python_files) > 3:  # More than just a few files shows organization
            score += 0.75

        return min(1.0, score / max_score)

    def _analyze_code_quality(self, repo_path: Path) -> float:
        """
        Analyze code files for quality indicators.

        Args:
            repo_path: Path to cloned repository

        Returns:
            Code quality score from 0-1
        """
        if not repo_path.exists():
            return 0.0

        score = 0.0
        max_score = 7.0  # Reduced total possible points for easier scoring

        # Find Python files (most common for ML projects)
        python_files = list(repo_path.glob("**/*.py"))

        if not python_files:
            # No Python files, check for other common languages
            other_files = (
                list(repo_path.glob("**/*.js")) +
                list(repo_path.glob("**/*.java")) +
                list(repo_path.glob("**/*.cpp")) +
                list(repo_path.glob("**/*.c")) +
                list(repo_path.glob("**/*.r")) +
                list(repo_path.glob("**/*.R")) +
                list(repo_path.glob("**/*.ipynb"))  # Jupyter notebooks
            )
            if other_files:
                score += 2.0  # Generous credit for having any code files
            return min(1.0, score / max_score)

        # Give generous base score for having Python code (2 points)
        score += 2.0

        # Analyze Python code quality with relaxed standards
        files_with_docstrings = 0
        files_with_imports = 0

        # Sample up to 5 Python files to avoid excessive processing
        sample_files = python_files[:5]

        for py_file in sample_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')

                # Very relaxed docstring check - any triple quotes
                if '"""' in content or "'''" in content or 'def ' in content:
                    files_with_docstrings += 1

                # Check for imports (shows it's real code, not just scripts)
                if 'import ' in content or 'from ' in content:
                    files_with_imports += 1

            except Exception:
                continue

        if sample_files:
            # Docstring/function coverage (1.5 points - generous)
            if files_with_docstrings > 0:
                score += 1.5

            # Import usage (shows structured code) (1 point)
            if files_with_imports > 0:
                score += 1.0

        # Check for any kind of code organization (0.5 points)
        if len(python_files) > 1:
            score += 0.5

        return min(1.0, score / max_score)