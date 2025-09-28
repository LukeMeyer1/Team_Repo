import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from app.metrics.base import ResourceBundle
from app.metrics.registry import register
from app.metrics.base_metric import BaseMetric

# Import logger from project root
_root_path = str(Path(__file__).parent.parent.parent.parent)
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

from Logger import get_logger
logger = get_logger(__name__)

# Import HuggingFace client with error handling
try:
    from app.integrations.huggingface_client import hf_client
    logger.info("HuggingFace client successfully imported")
except ImportError:
    hf_client = None
    logger.warning("HuggingFace client not available - dataset quality analysis will be limited")

@register("dataset_quality")
class DatasetQualityMetric(BaseMetric):
    """
    Evaluates dataset reliability, popularity, and quality indicators.

    This metric analyzes datasets used for training and evaluation across multiple dimensions:
    - Dataset size, diversity, and representativeness
    - Data quality indicators (completeness, accuracy, consistency)
    - Dataset popularity and community adoption (downloads, likes)
    - Documentation completeness and metadata quality
    - Data collection methodology and ethics
    - Bias analysis and fairness considerations
    - License compliance and legal considerations

    Scoring Rubric:
    - 1.0: Excellent datasets with comprehensive documentation, high popularity, and quality indicators
    - 0.8: Good datasets with solid documentation and moderate popularity
    - 0.6: Acceptable datasets but missing some quality indicators
    - 0.4: Poor datasets with limited documentation or quality concerns
    - 0.2: Very poor datasets with significant quality issues
    - 0.0: No datasets or completely inaccessible datasets
    """

    name = "dataset_quality"

    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate quality and reliability of associated datasets.

        Analyzes each dataset for:
        1. Popularity metrics (downloads, likes, community usage)
        2. Documentation quality and completeness
        3. Size categories and data diversity
        4. Licensing and ethical considerations
        5. Task alignment and utility

        Args:
            resource: Bundle with dataset URLs to analyze

        Returns:
            Score from 0-1 where 1 = high-quality, well-documented, widely-used datasets
        """
        if not resource.dataset_urls:
            logger.debug(f"No dataset URLs provided for model {resource.model_url}")
            return 0.0

        if not hf_client:
            logger.error("HuggingFace client unavailable - returning minimal score")
            return 0.1  # Minimal score if HF client unavailable

        logger.info(f"Starting dataset quality analysis for {len(resource.dataset_urls)} dataset(s) "
                   f"from model {resource.model_url}")

        total_score = 0.0
        analyzed_datasets = 0
        dataset_details = []
        failed_datasets = 0

        # Analyze each dataset (limit to first 5 to avoid excessive processing)
        for dataset_url in resource.dataset_urls[:5]:
            try:
                dataset_id = self._extract_dataset_id(dataset_url)
                if not dataset_id:
                    logger.warning(f"Could not extract dataset ID from URL: {dataset_url}")
                    continue

                logger.debug(f"Analyzing dataset: {dataset_id}")
                dataset_score, details = self._analyze_single_dataset(dataset_id)
                if dataset_score is not None:
                    total_score += dataset_score
                    analyzed_datasets += 1
                    dataset_details.append(details)
                    logger.debug(f"Dataset {dataset_id} scored {dataset_score:.2f}")
                else:
                    failed_datasets += 1

            except Exception as e:
                failed_datasets += 1
                logger.warning(f"Failed to analyze dataset {dataset_url}: {str(e)}")
                continue  # Skip failed datasets

        if analyzed_datasets == 0:
            logger.warning(f"No datasets could be analyzed for model {resource.model_url} "
                         f"({failed_datasets} failed, {len(resource.dataset_urls)} total)")
            return 0.0

        # Average score across analyzed datasets
        average_score = total_score / analyzed_datasets

        # Bonus for dataset diversity (multiple quality datasets)
        diversity_bonus = min(0.1, (analyzed_datasets - 1) * 0.05)

        final_score = min(1.0, average_score + diversity_bonus)

        # Store analysis details for notes
        self._analysis_details = {
            'analyzed_count': analyzed_datasets,
            'total_count': len(resource.dataset_urls),
            'dataset_details': dataset_details,
            'average_score': average_score,
            'diversity_bonus': diversity_bonus
        }

        logger.info(f"Dataset quality analysis complete for {resource.model_url}: "
                   f"{analyzed_datasets}/{len(resource.dataset_urls)} datasets analyzed, "
                   f"final score: {final_score:.2f} (avg: {average_score:.2f}, "
                   f"diversity bonus: {diversity_bonus:.2f})")

        if failed_datasets > 0:
            logger.warning(f"{failed_datasets} datasets failed analysis for {resource.model_url}")

        return final_score

    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        if not resource.dataset_urls:
            return f"No datasets found for {resource.model_url}. Quality assessment not possible."

        if not hf_client:
            return "HuggingFace client unavailable. Cannot analyze dataset quality."

        details = getattr(self, '_analysis_details', {})
        if not details:
            return f"Failed to analyze {len(resource.dataset_urls)} dataset(s) for {resource.model_url}."

        analyzed = details.get('analyzed_count', 0)
        total = details.get('total_count', 0)
        avg_score = details.get('average_score', 0.0)

        notes = [
            f"Analyzed {analyzed}/{total} datasets for {resource.model_url}.",
            f"Average dataset quality score: {avg_score:.2f}"
        ]

        # Add details about top datasets
        dataset_details = details.get('dataset_details', [])
        if dataset_details:
            top_datasets = sorted(dataset_details, key=lambda x: x['score'], reverse=True)[:3]
            notes.append("Top datasets:")
            for ds in top_datasets:
                notes.append(f"  - {ds['id']}: {ds['score']:.2f} (downloads: {ds['downloads']}, likes: {ds['likes']})")

        return " ".join(notes)

    def _extract_dataset_id(self, dataset_url: str) -> Optional[str]:
        """
        Extract dataset ID from HuggingFace dataset URL.

        Args:
            dataset_url: Full URL to HuggingFace dataset

        Returns:
            Dataset ID string or None if not a valid HF dataset URL
        """
        if not dataset_url:
            return None

        # Handle direct dataset IDs
        if '/' not in dataset_url or not dataset_url.startswith('http'):
            return dataset_url

        # Parse HuggingFace dataset URLs
        # https://huggingface.co/datasets/squad
        # https://huggingface.co/datasets/glue/cola
        parsed = urlparse(dataset_url)
        if 'huggingface.co' not in parsed.netloc:
            return None

        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2 and path_parts[0] == 'datasets':
            # Extract dataset name (may include organization)
            if len(path_parts) == 2:
                return path_parts[1]  # Simple case: datasets/squad
            else:
                # Complex case: datasets/org/dataset or datasets/dataset/subset
                return '/'.join(path_parts[1:3])  # Take first two parts after datasets/

        return None

    def _analyze_single_dataset(self, dataset_id: str) -> Tuple[Optional[float], Dict]:
        """
        Analyze a single dataset for quality metrics.

        Args:
            dataset_id: HuggingFace dataset identifier

        Returns:
            Tuple of (score, details_dict) or (None, {}) on failure
        """
        try:
            # Get dataset info and metadata
            logger.debug(f"Fetching dataset info for {dataset_id}")
            info_result = hf_client.get_dataset_info(dataset_id)
            if 'error' in info_result:
                logger.warning(f"Failed to get dataset info for {dataset_id}: {info_result.get('error')}")
                return None, {}

            card_data = hf_client.get_dataset_card_data(dataset_id)
            if 'error' in card_data:
                logger.debug(f"No card data available for {dataset_id}, using empty data")
                card_data = {}

            readme_content = hf_client.get_dataset_readme(dataset_id)
            if not readme_content:
                logger.debug(f"No README content available for {dataset_id}")

            # Extract basic metrics
            dataset_info = info_result.get('dataset_info')
            downloads = card_data.get('downloads', 0) or 0
            likes = card_data.get('likes', 0) or 0

            # Calculate quality score components
            popularity_score = self._calculate_popularity_score(downloads, likes)
            documentation_score = self._calculate_documentation_score(card_data, readme_content)
            metadata_score = self._calculate_metadata_score(card_data)
            diversity_score = self._calculate_diversity_score(card_data)
            licensing_score = self._calculate_licensing_score(card_data)

            # Weighted overall score
            overall_score = (
                popularity_score * 0.25 +      # 25% popularity
                documentation_score * 0.30 +   # 30% documentation
                metadata_score * 0.20 +        # 20% metadata completeness
                diversity_score * 0.15 +       # 15% data diversity
                licensing_score * 0.10         # 10% licensing
            )

            details = {
                'id': dataset_id,
                'score': overall_score,
                'downloads': downloads,
                'likes': likes,
                'popularity_score': popularity_score,
                'documentation_score': documentation_score,
                'metadata_score': metadata_score,
                'diversity_score': diversity_score,
                'licensing_score': licensing_score
            }

            logger.debug(f"Dataset {dataset_id} analysis complete: score={overall_score:.2f}, "
                        f"downloads={downloads}, likes={likes}")

            return overall_score, details

        except Exception as e:
            logger.error(f"Unexpected error analyzing dataset {dataset_id}: {str(e)}")
            return None, {}

    def _calculate_popularity_score(self, downloads: int, likes: int) -> float:
        """
        Calculate popularity score based on downloads and likes.

        Args:
            downloads: Number of downloads
            likes: Number of likes

        Returns:
            Score from 0-1 based on popularity metrics
        """
        # Normalize download counts with more reasonable thresholds
        if downloads > 0:
            # Use logarithmic scaling with lower thresholds:
            # 10K+ downloads = 1.0, 1K+ = 0.7, 100+ = 0.4, 10+ = 0.2
            import math
            download_score = min(1.0, math.log10(downloads + 1) / 4.0)  # log10(10000) = 4
        else:
            download_score = 0.0

        # Normalize like counts with more reasonable thresholds
        if likes > 0:
            # 100+ likes = 1.0, 10+ = 0.7, 5+ = 0.5, 1+ = 0.3
            import math
            like_score = min(1.0, math.log10(likes + 1) / 2.0)  # log10(100) = 2
        else:
            like_score = 0.0

        # Weighted combination (downloads more important than likes)
        return download_score * 0.7 + like_score * 0.3

    def _calculate_documentation_score(self, card_data: Dict, readme_content: Optional[str]) -> float:
        """
        Calculate documentation quality score.

        Args:
            card_data: Dataset card metadata
            readme_content: README.md content

        Returns:
            Score from 0-1 based on documentation completeness
        """
        score = 0.0

        # Check for README content - more generous base score
        if readme_content:
            score += 0.5  # Increased from 0.4

            # Bonus for any meaningful content (lowered threshold)
            if len(readme_content) > 200:  # Reduced from 1000
                score += 0.15

            # Check for key documentation sections (more flexible matching)
            readme_lower = readme_content.lower()
            key_sections = ['dataset', 'description', 'usage', 'citation', 'license']
            found_sections = sum(1 for section in key_sections if section in readme_lower)
            score += (found_sections / len(key_sections)) * 0.15  # Reduced impact

        # Check for structured metadata - more generous scoring
        metadata_fields = [
            card_data.get('task_categories'),
            card_data.get('language'),
            card_data.get('size_categories'),
            card_data.get('source_datasets')
        ]

        # Award 0.05 for each metadata field present (up to 0.2 total)
        for field in metadata_fields:
            if field:
                score += 0.05

        return min(1.0, score)

    def _calculate_metadata_score(self, card_data: Dict) -> float:
        """
        Calculate metadata completeness score.

        Args:
            card_data: Dataset card metadata

        Returns:
            Score from 0-1 based on metadata completeness
        """
        score = 0.0
        max_score = 6.0  # Reduced from 8.0 to be more lenient

        # Core metadata fields (weighted more heavily)
        if card_data.get('task_categories'):
            score += 1.5  # Most important
        if card_data.get('language'):
            score += 1.5  # Very important
        if card_data.get('size_categories'):
            score += 1.0
        if card_data.get('tags'):
            score += 1.0

        # Optional metadata fields (lighter weight)
        if card_data.get('task_ids'):
            score += 0.5
        if card_data.get('multilinguality'):
            score += 0.25
        if card_data.get('source_datasets'):
            score += 0.25
        if card_data.get('created_at') or card_data.get('last_modified'):
            score += 0.5

        return min(1.0, score / max_score)

    def _calculate_diversity_score(self, card_data: Dict) -> float:
        """
        Calculate data diversity score.

        Args:
            card_data: Dataset card metadata

        Returns:
            Score from 0-1 based on data diversity indicators
        """
        score = 0.0

        # Language diversity - be more generous
        languages = card_data.get('language', [])
        if isinstance(languages, list) and len(languages) > 1:
            score += 0.4  # Multilingual datasets
        elif languages:
            score += 0.2  # Single language specified (increased from 0.1)

        # Size categories - more generous scoring
        size_categories = card_data.get('size_categories', [])
        if isinstance(size_categories, list) and size_categories:
            # Any size category specified gets decent score
            score += 0.3
            # Bonus for larger datasets
            large_indicators = ['100K<n<1M', '1M<n<10M', '10M<n<100M', '100M<n<1B', 'n>1B']
            if any(indicator in size_categories for indicator in large_indicators):
                score += 0.2  # Bonus for large datasets

        # Task diversity - more generous
        task_categories = card_data.get('task_categories', [])
        if isinstance(task_categories, list) and len(task_categories) > 1:
            score += 0.3  # Multi-task datasets
        elif task_categories:
            score += 0.2  # Single task specified (increased from 0.1)

        # Source diversity - keep same
        source_datasets = card_data.get('source_datasets', [])
        if isinstance(source_datasets, list) and len(source_datasets) > 1:
            score += 0.1  # Derived from multiple sources

        return min(1.0, score)

    def _calculate_licensing_score(self, card_data: Dict) -> float:
        """
        Calculate licensing and ethics score.

        Args:
            card_data: Dataset card metadata

        Returns:
            Score from 0-1 based on licensing clarity and ethics
        """
        score = 0.0

        license_info = card_data.get('license')
        if license_info:
            # Open licenses get higher scores
            open_licenses = ['mit', 'apache-2.0', 'bsd', 'cc-by', 'cc0', 'unlicense']
            if any(open_lic in str(license_info).lower() for open_lic in open_licenses):
                score += 0.8
            else:
                score += 0.5  # Has license but may be restrictive (increased from 0.4)
        else:
            score += 0.2  # No license information (increased from 0.1 to be less punitive)

        # Bonus for ethics considerations
        tags = card_data.get('tags', [])
        if isinstance(tags, list):
            ethics_tags = ['ethics', 'bias', 'fairness', 'privacy', 'responsible-ai']
            if any(tag in str(tags).lower() for tag in ethics_tags):
                score += 0.2

        return min(1.0, score)