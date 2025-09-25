"""
Dataset tracking and inference system.

Maintains a registry of encountered datasets and provides inference capabilities
for models with blank dataset links by analyzing their README content.
"""

import re
from pathlib import Path
from typing import Set, Optional, List, Dict
from difflib import SequenceMatcher


class DatasetTracker:
    """
    Tracks encountered datasets and infers dataset references from model metadata.

    This class maintains a registry of datasets seen in previous CSV entries
    and can infer which dataset a model uses by analyzing its README content
    when the dataset_link field is blank.
    """

    def __init__(self):
        """Initialize empty dataset registry."""
        self.seen_datasets: Set[str] = set()
        self.dataset_names: Dict[str, str] = {}  # name -> full_url mapping

    def add_dataset(self, dataset_url: str) -> None:
        """
        Add a dataset URL to the registry.

        Args:
            dataset_url: Full dataset URL to track
        """
        if dataset_url:
            self.seen_datasets.add(dataset_url)
            # Extract dataset name for fuzzy matching
            dataset_name = self._extract_dataset_name(dataset_url)
            if dataset_name:
                self.dataset_names[dataset_name] = dataset_url

    def infer_dataset(self, model_url: str) -> Optional[str]:
        """
        Infer dataset URL for a model by analyzing its README content.

        Args:
            model_url: HuggingFace model URL to analyze

        Returns:
            Dataset URL if successfully inferred, None otherwise
        """
        if not self.seen_datasets:
            return None

        try:
            readme_content = self._get_model_readme(model_url)
            if not readme_content:
                return None

            # Look for dataset references in README
            dataset_references = self._extract_dataset_references(readme_content)

            # Try to match against known datasets
            for reference in dataset_references:
                matched_url = self._match_dataset_reference(reference)
                if matched_url:
                    return matched_url

            return None

        except Exception:
            # Gracefully handle errors - inference is optional
            return None

    def _extract_dataset_name(self, dataset_url: str) -> Optional[str]:
        """
        Extract dataset name from URL.

        Args:
            dataset_url: Full dataset URL

        Returns:
            Dataset name or None if extraction fails
        """
        if "huggingface.co/datasets/" in dataset_url:
            # Extract from: https://huggingface.co/datasets/org/name -> name
            parts = dataset_url.split("huggingface.co/datasets/")[-1].split("/")
            return parts[-1] if parts else None
        return None

    def _get_model_readme(self, model_url: str) -> Optional[str]:
        """
        Fetch README content for a HuggingFace model.

        Args:
            model_url: HuggingFace model URL

        Returns:
            README content as string or None if unavailable
        """
        try:
            # Add src to path for imports
            import sys
            from pathlib import Path

            ROOT = Path(__file__).resolve().parent.parent.parent
            SRC = ROOT / "src"
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))

            from app.integrations.huggingface_client import hf_client

            # Extract model_id from URL
            model_id = self._extract_model_id(model_url)
            if not model_id:
                return None

            return hf_client.get_model_readme(model_id)

        except Exception:
            return None

    def _extract_model_id(self, model_url: str) -> Optional[str]:
        """
        Extract model ID from URL.

        Args:
            model_url: Full model URL

        Returns:
            Model ID (org/model format) or None
        """
        if "huggingface.co/" in model_url:
            parts = model_url.split("huggingface.co/")[-1].split("/")
            if len(parts) >= 2:
                return "/".join(parts)
        return None

    def _extract_dataset_references(self, readme_content: str) -> List[str]:
        """
        Extract potential dataset references from README content.

        Args:
            readme_content: README text content

        Returns:
            List of potential dataset names/references
        """
        references = []

        # Pattern 1: Direct HuggingFace dataset URLs
        hf_pattern = r'https://huggingface\.co/datasets/([a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)?)'
        hf_matches = re.findall(hf_pattern, readme_content)
        references.extend(hf_matches)

        # Pattern 2: Dataset mentions in text (e.g., "trained on dataset_name")
        dataset_patterns = [
            r'trained on ([a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)?)',
            r'dataset[s]?[:\s]+([a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)?)',
            r'using ([a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)?) dataset',
        ]

        for pattern in dataset_patterns:
            matches = re.findall(pattern, readme_content, re.IGNORECASE)
            references.extend(matches)

        # Clean up references (remove common non-dataset words)
        cleaned_refs = []
        excluded_words = {'the', 'this', 'that', 'model', 'training', 'data', 'set'}

        for ref in references:
            if ref.lower() not in excluded_words and len(ref) > 2:
                cleaned_refs.append(ref)

        return cleaned_refs

    def _match_dataset_reference(self, reference: str) -> Optional[str]:
        """
        Match a dataset reference against known datasets using fuzzy matching.

        Args:
            reference: Dataset name or reference from README

        Returns:
            Matched dataset URL or None
        """
        # Direct name match
        if reference in self.dataset_names:
            return self.dataset_names[reference]

        # Fuzzy matching against known dataset names
        best_match = None
        best_ratio = 0.7  # Minimum similarity threshold

        for known_name, url in self.dataset_names.items():
            ratio = SequenceMatcher(None, reference.lower(), known_name.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = url

        return best_match

    def get_stats(self) -> Dict[str, int]:
        """
        Get tracking statistics.

        Returns:
            Dictionary with tracking statistics
        """
        return {
            'total_datasets': len(self.seen_datasets),
            'named_datasets': len(self.dataset_names)
        }