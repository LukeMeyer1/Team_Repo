"""
LLM-based metadata extraction for enhancing model evaluations.

This module uses an LLM to analyze model README and metadata to identify
additional datasets, code repositories, and other relevant information
that can improve metric scoring.
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from app.integrations.huggingface_client import hf_client
from app.integrations.genai_studio import GenAIStudio


@dataclass
class ExtractedMetadata:
    """Structured metadata extracted from model documentation."""
    dataset_urls: List[str]
    code_urls: List[str]
    training_datasets: List[str]
    evaluation_datasets: List[str]
    paper_urls: List[str]
    license_info: Optional[str]
    model_architecture: Optional[str]
    training_details: Optional[str]
    use_cases: List[str]
    limitations: List[str]
    confidence_score: float  # LLM's confidence in the extraction


class LLMMetadataExtractor:
    """
    Extracts additional metadata from model documentation using LLM analysis.

    This class fetches model README and metadata from HuggingFace, then uses
    an LLM to intelligently extract URLs, dataset references, and other
    information that can enhance metric calculations.
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize the metadata extractor.

        Args:
            use_llm: Whether to use LLM for extraction. If False, uses simpler regex patterns.
        """
        self.use_llm = use_llm
        self.genai_client = None

        if use_llm:
            try:
                self.genai_client = GenAIStudio.from_env()
            except Exception:
                # Fall back to regex-based extraction if GenAI is not available
                self.use_llm = False

    def extract_metadata(self, model_id: str) -> ExtractedMetadata:
        """
        Extract enhanced metadata for a given model.

        Args:
            model_id: HuggingFace model identifier (e.g., "bert-base-uncased")

        Returns:
            ExtractedMetadata with discovered information
        """
        # Fetch model information from HuggingFace
        model_info = hf_client.get_model_info(model_id)
        if 'error' in model_info:
            return self._empty_metadata(f"Failed to fetch model info: {model_info['error']}")

        readme_content = hf_client.get_model_readme(model_id)
        card_data = hf_client.get_model_card_data(model_id)

        if not readme_content and not card_data:
            return self._empty_metadata("No README or card data available")

        # Combine all available text for analysis
        combined_text = self._prepare_text_for_analysis(readme_content, card_data, model_info)

        if self.use_llm and self.genai_client:
            return self._extract_with_llm(combined_text, model_id)
        else:
            return self._extract_with_regex(combined_text, model_id)

    def _prepare_text_for_analysis(self, readme: Optional[str],
                                 card_data: Optional[Dict],
                                 model_info: Dict) -> str:
        """Combine and clean text from various sources for analysis."""
        text_parts = []

        # Add README content
        if readme:
            text_parts.append("=== README ===\n" + readme)

        # Add structured card data
        if card_data and not isinstance(card_data, dict) or 'error' not in card_data:
            if isinstance(card_data, dict):
                formatted_card = json.dumps(card_data, indent=2)
                text_parts.append("=== MODEL CARD DATA ===\n" + formatted_card)

        # Add basic model info
        if 'model_info' in model_info:
            info = model_info['model_info']
            info_text = f"Model ID: {getattr(info, 'modelId', 'Unknown')}\n"
            if hasattr(info, 'downloads'):
                info_text += f"Downloads: {info.downloads}\n"
            if hasattr(info, 'likes'):
                info_text += f"Likes: {info.likes}\n"
            if hasattr(info, 'tags'):
                info_text += f"Tags: {', '.join(info.tags)}\n"
            text_parts.append("=== MODEL INFO ===\n" + info_text)

        return "\n\n".join(text_parts)

    def _extract_with_llm(self, text: str, model_id: str) -> ExtractedMetadata:
        """Use LLM to extract structured metadata from text."""

        extraction_prompt = """
You are an expert at analyzing machine learning model documentation. Extract the following information from the provided text and return it as a JSON object:

{
  "dataset_urls": ["list of any HuggingFace dataset URLs mentioned"],
  "code_urls": ["list of any GitHub/GitLab repository URLs mentioned"],
  "training_datasets": ["list of dataset names used for training"],
  "evaluation_datasets": ["list of dataset names used for evaluation/benchmarking"],
  "paper_urls": ["list of any research paper URLs mentioned"],
  "license_info": "license type if mentioned",
  "model_architecture": "brief description of the model architecture",
  "training_details": "brief summary of training approach/details",
  "use_cases": ["list of mentioned use cases or applications"],
  "limitations": ["list of mentioned limitations or biases"],
  "confidence_score": 0.85
}

Guidelines:
- Only include URLs that are explicitly mentioned in the text
- For dataset names, include both formal names (e.g., "GLUE") and HuggingFace dataset IDs
- Set confidence_score based on how much relevant information you found (0.0-1.0)
- Use null for fields where no information is available
- Be conservative - only include information you're confident about

Return ONLY the JSON object, no additional text.
"""

        try:
            response = self.genai_client.evaluate(
                method=extraction_prompt,
                contents=text,
                model="llama3.1:latest",
                temperature=0.1,  # Low temperature for consistency
                max_tokens=1000
            )

            # Parse the JSON response
            metadata_dict = json.loads(response.strip())

            return ExtractedMetadata(
                dataset_urls=metadata_dict.get('dataset_urls', []),
                code_urls=metadata_dict.get('code_urls', []),
                training_datasets=metadata_dict.get('training_datasets', []),
                evaluation_datasets=metadata_dict.get('evaluation_datasets', []),
                paper_urls=metadata_dict.get('paper_urls', []),
                license_info=metadata_dict.get('license_info'),
                model_architecture=metadata_dict.get('model_architecture'),
                training_details=metadata_dict.get('training_details'),
                use_cases=metadata_dict.get('use_cases', []),
                limitations=metadata_dict.get('limitations', []),
                confidence_score=metadata_dict.get('confidence_score', 0.5)
            )

        except Exception as e:
            # Fall back to regex-based extraction if LLM fails
            return self._extract_with_regex(text, model_id)

    def _extract_with_regex(self, text: str, model_id: str) -> ExtractedMetadata:
        """Use regex patterns to extract basic information from text."""

        # URL patterns
        github_pattern = r'https?://github\.com/[^\s\)\]<>"]+'
        hf_dataset_pattern = r'https?://huggingface\.co/datasets/[^\s\)\]<>"]+'
        paper_pattern = r'https?://(?:arxiv\.org|aclanthology\.org|papers\.nips\.cc)[^\s\)\]<>"]+'

        # Dataset name patterns
        dataset_name_pattern = r'\b(?:GLUE|SQuAD|CoLA|SST-2|MRPC|QQP|MNLI|QNLI|RTE|WNLI|ImageNet|COCO|WikiText|Common Crawl|BookCorpus|OpenWebText)\b'

        # Extract URLs
        dataset_urls = re.findall(hf_dataset_pattern, text, re.IGNORECASE)
        code_urls = re.findall(github_pattern, text, re.IGNORECASE)
        paper_urls = re.findall(paper_pattern, text, re.IGNORECASE)

        # Extract dataset names
        dataset_names = re.findall(dataset_name_pattern, text, re.IGNORECASE)

        # Simple license detection
        license_info = None
        license_patterns = [
            r'license[:\s]*([a-zA-Z0-9\-\.]+)',
            r'licensed under[:\s]*([a-zA-Z0-9\-\.]+)',
            r'(MIT|Apache|BSD|GPL|CC[BY\-SA\d]*)',
        ]
        for pattern in license_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                license_info = match.group(1)
                break

        # Calculate confidence based on how much we found
        found_items = len(dataset_urls) + len(code_urls) + len(paper_urls) + len(dataset_names)
        confidence_score = min(0.8, found_items * 0.2)  # Cap at 0.8 for regex-based extraction

        return ExtractedMetadata(
            dataset_urls=list(set(dataset_urls)),  # Remove duplicates
            code_urls=list(set(code_urls)),
            training_datasets=list(set(dataset_names)),
            evaluation_datasets=list(set(dataset_names)),  # Same as training for regex
            paper_urls=list(set(paper_urls)),
            license_info=license_info,
            model_architecture=None,  # Would require more sophisticated extraction
            training_details=None,
            use_cases=[],
            limitations=[],
            confidence_score=confidence_score
        )

    def _empty_metadata(self, reason: str) -> ExtractedMetadata:
        """Return empty metadata structure with error information."""
        return ExtractedMetadata(
            dataset_urls=[],
            code_urls=[],
            training_datasets=[],
            evaluation_datasets=[],
            paper_urls=[],
            license_info=None,
            model_architecture=None,
            training_details=reason,  # Store error reason here
            use_cases=[],
            limitations=[],
            confidence_score=0.0
        )


# Singleton instance for easy use across metrics
metadata_extractor = LLMMetadataExtractor()