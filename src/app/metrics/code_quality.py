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
        code_urls = resource.code_urls

        # LLM inference is now handled by the orchestrator - URLs are pre-populated in ResourceBundle

        if not code_urls:
            return 0.1  # Give small score instead of 0 - model might have implicit code

        # Filter for GitHub repositories only
        github_repos = [url for url in code_urls if categorize_url(url) == "CODE"]
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
            return f"No code repositories found for {resource.model_url}. Quality assessment limited to baseline score."

        code_urls = resource.code_urls

        github_repos = [url for url in code_urls if categorize_url(url) == "CODE"]
        if not github_repos:
            return f"No GitHub repositories found in {len(code_urls)} code URLs. Only GitHub repositories are analyzed."

        return (
            f"Analyzed {min(len(github_repos), 3)} GitHub repositories for {resource.model_url}{inferred_note}. "
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
        max_score = 10.0  # Total possible points

        # Check for README file (2 points)
        readme_files = list(repo_path.glob("README*"))
        if readme_files:
            score += 2.0

        # Check for LICENSE file (1 point)
        license_files = list(repo_path.glob("LICENSE*")) + list(repo_path.glob("LICENCE*"))
        if license_files:
            score += 1.0

        # Check for test directory or test files (2 points)
        test_dirs = list(repo_path.glob("test*")) + list(repo_path.glob("Test*"))
        test_files = list(repo_path.glob("**/test_*.py")) + list(repo_path.glob("**/*_test.py"))
        if test_dirs or test_files:
            score += 2.0

        # Check for CI/CD configuration (1 point)
        ci_configs = (
            list(repo_path.glob(".github/workflows/*")) +
            list(repo_path.glob(".gitlab-ci.yml")) +
            list(repo_path.glob("*.yml")) +
            list(repo_path.glob(".travis.yml"))
        )
        if ci_configs:
            score += 1.0

        # Check for dependency management (1 point)
        dep_files = (
            list(repo_path.glob("requirements*.txt")) +
            list(repo_path.glob("setup.py")) +
            list(repo_path.glob("pyproject.toml")) +
            list(repo_path.glob("Pipfile")) +
            list(repo_path.glob("package.json"))
        )
        if dep_files:
            score += 1.0

        # Check for documentation directory (1 point)
        doc_dirs = list(repo_path.glob("doc*")) + list(repo_path.glob("Doc*"))
        if doc_dirs:
            score += 1.0

        # Check for proper project structure (1 point)
        src_dirs = list(repo_path.glob("src")) + list(repo_path.glob("lib"))
        python_packages = list(repo_path.glob("*/__init__.py"))
        if src_dirs or python_packages:
            score += 1.0

        # Check for configuration files (1 point)
        config_files = (
            list(repo_path.glob("*.cfg")) +
            list(repo_path.glob("*.ini")) +
            list(repo_path.glob("*.toml")) +
            list(repo_path.glob(".gitignore"))
        )
        if config_files:
            score += 1.0

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
        max_score = 8.0  # Total possible points

        # Find Python files (most common for ML projects)
        python_files = list(repo_path.glob("**/*.py"))

        if not python_files:
            # No Python files, check for other common languages
            other_files = (
                list(repo_path.glob("**/*.js")) +
                list(repo_path.glob("**/*.java")) +
                list(repo_path.glob("**/*.cpp")) +
                list(repo_path.glob("**/*.c"))
            )
            if other_files:
                score += 1.0  # Partial credit for having code files
            return min(1.0, score / max_score)

        # Analyze Python code quality
        total_lines = 0
        files_with_docstrings = 0
        files_with_type_hints = 0
        average_line_length = 0
        total_line_length = 0

        # Sample up to 10 Python files to avoid excessive processing
        sample_files = python_files[:10]

        for py_file in sample_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                total_lines += len(lines)

                # Check for docstrings
                if '"""' in content or "'''" in content:
                    files_with_docstrings += 1

                # Check for type hints
                if ':' in content and '->' in content:
                    files_with_type_hints += 1

                # Calculate average line length
                for line in lines:
                    total_line_length += len(line.strip())

            except Exception:
                continue

        if sample_files:
            # Docstring coverage (2 points)
            docstring_ratio = files_with_docstrings / len(sample_files)
            score += 2.0 * docstring_ratio

            # Type hint usage (1 point)
            type_hint_ratio = files_with_type_hints / len(sample_files)
            score += 1.0 * type_hint_ratio

            # Code organization (1 point for having multiple files)
            if len(python_files) > 1:
                score += 1.0

            # Line length reasonableness (1 point if average < 100 chars)
            if total_lines > 0:
                avg_line_length = total_line_length / total_lines
                if avg_line_length < 100:
                    score += 1.0

        # Check for common Python quality files (3 points)
        quality_files = (
            list(repo_path.glob("*.flake8")) +
            list(repo_path.glob(".flake8")) +
            list(repo_path.glob("pylintrc")) +
            list(repo_path.glob(".pylintrc")) +
            list(repo_path.glob("mypy.ini")) +
            list(repo_path.glob("tox.ini"))
        )
        if quality_files:
            score += 1.0

        # Check for __init__.py files (proper package structure) (1 point)
        init_files = list(repo_path.glob("**/__init__.py"))
        if init_files:
            score += 1.0

        # Check for main module pattern (1 point)
        main_patterns = list(repo_path.glob("**/main.py")) + list(repo_path.glob("**/__main__.py"))
        if main_patterns:
            score += 1.0

        return min(1.0, score / max_score)