import re
import sys
from pathlib import Path
from typing import Optional
from app.metrics.base import ResourceBundle
from app.metrics.registry import register
from app.metrics.base_metric import BaseMetric

# Import HuggingFace client
try:
    from app.integrations.huggingface_client import hf_client
except ImportError:
    hf_client = None

# Import LLM metadata extractor
try:
    from app.metrics.llm_metadata_extractor import metadata_extractor
except ImportError:
    metadata_extractor = None

@register("dataset_and_code_score")
class DatasetAndCodeScoreMetric(BaseMetric):
    """
    Evaluates presence of dataset and code references in model documentation.
    Enhanced with LLM-based metadata extraction for bonus scoring.
    """

    name = "dataset_and_code_score"

    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate documentation quality of datasets and code using LLM analysis.
        Focus on how well the training/benchmark datasets and example code are documented.
        """
        # Start with generous base score for any model
        score = 0.5

        # Get README content for analysis
        readme_content = ""
        if hf_client and resource.model_id:
            readme_content = hf_client.get_model_readme(resource.model_id) or ""

        # Documentation quality scoring based on length and content
        doc_quality_score = self._analyze_documentation_quality(readme_content)
        score += doc_quality_score * 0.3  # Up to 0.3 points for good documentation

        # LLM Enhancement: Analyze quality and completeness of dataset/code documentation
        llm_analysis_score = self._analyze_with_llm(readme_content, resource.model_id)
        score += llm_analysis_score * 0.2  # Up to 0.2 points for LLM analysis

        # Bonus for any discovered or provided URLs (shows linked resources)
        if resource.dataset_urls or resource.code_urls:
            score += 0.1

        return min(score, 1.0)

    def _analyze_documentation_quality(self, readme_content: str) -> float:
        """
        Analyze the quality and completeness of documentation.

        Args:
            readme_content: README content to analyze

        Returns:
            Score from 0-1 based on documentation quality
        """
        if not readme_content:
            return 0.0

        score = 0.0
        readme_lower = readme_content.lower()

        # Length indicates thoroughness - be generous
        if len(readme_content) > 500:
            score += 0.4
        elif len(readme_content) > 200:
            score += 0.2

        # Look for key documentation sections (very generous keywords)
        documentation_indicators = [
            # Dataset documentation
            "dataset", "training data", "data", "corpus", "benchmark",
            "trained on", "fine-tuned", "evaluation", "test set",

            # Code/usage documentation
            "usage", "example", "code", "implementation", "how to", "tutorial",
            "install", "requirements", "getting started", "quickstart",

            # Model details
            "model", "architecture", "parameters", "configuration",
            "performance", "results", "accuracy", "metrics"
        ]

        found_indicators = sum(1 for indicator in documentation_indicators
                             if indicator in readme_lower)

        # Very generous scoring - if we find several indicators, assume good documentation
        if found_indicators >= 8:
            score += 0.6
        elif found_indicators >= 5:
            score += 0.4
        elif found_indicators >= 3:
            score += 0.3
        elif found_indicators >= 1:
            score += 0.2

        return min(score, 1.0)

    def _analyze_with_llm(self, readme_content: str, model_id: str) -> float:
        """
        Use LLM to analyze quality of dataset and code documentation.

        Args:
            readme_content: README content to analyze
            model_id: Model identifier

        Returns:
            Score from 0-1 based on LLM analysis of documentation quality
        """
        if not metadata_extractor or not readme_content:
            return 0.5  # Default generous score if no LLM

        try:
            extracted_metadata = metadata_extractor.extract_metadata(model_id)

            # If LLM found detailed information, documentation is likely good
            score = 0.3  # Base LLM score

            # High confidence suggests clear, well-structured documentation
            if extracted_metadata.confidence_score > 0.7:
                score += 0.4
            elif extracted_metadata.confidence_score > 0.4:
                score += 0.2

            # Discovered training details suggest good dataset documentation
            if extracted_metadata.training_details or extracted_metadata.training_datasets:
                score += 0.2

            # Discovered use cases suggest good usage documentation
            if extracted_metadata.use_cases:
                score += 0.1

            return min(score, 1.0)

        except Exception:
            # If LLM analysis fails, give benefit of the doubt
            return 0.5


    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        notes = f"Documentation quality analysis for {resource.model_url} focusing on dataset and code documentation completeness."

        # Add information about documentation analysis
        if hf_client and resource.model_id:
            readme_content = hf_client.get_model_readme(resource.model_id) or ""
            if readme_content:
                notes += f" README length: {len(readme_content)} characters."

        # Add information about LLM enhancement if available
        if metadata_extractor and resource.model_id:
            try:
                extracted_metadata = metadata_extractor.extract_metadata(resource.model_id)
                if extracted_metadata.confidence_score > 0:
                    notes += f" LLM analysis confidence: {extracted_metadata.confidence_score:.2f}."

                    quality_indicators = []
                    if extracted_metadata.training_datasets:
                        quality_indicators.append("training data documented")
                    if extracted_metadata.use_cases:
                        quality_indicators.append("usage examples found")
                    if extracted_metadata.training_details:
                        quality_indicators.append("training details provided")

                    if quality_indicators:
                        notes += f" Quality indicators: {', '.join(quality_indicators)}."
            except Exception:
                pass

        return notes