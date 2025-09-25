"""
HuggingFace API client wrapper (anonymous/no-token)
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


def _with_retries(callable_fn, *args, **kwargs):
    """Basic exponential backoff retry for 429 (rate limit) & transient 5xx."""
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return callable_fn(*args, **kwargs)
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


class HuggingFaceClient:
    """
    Centralized HuggingFace API client (anonymous).
    No token is read or used. All calls are unauthenticated.
    """

    def __init__(self):
        # Anonymous API client (token=None)
        self.token = None
        self.api = HfApi(token=None)

    def get_model_info(self, model_id: str, include_files: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetch model information from HuggingFace (public models only).

        Returns:
            {'model_info': ModelInfo, 'model_id': str, 'has_token': False}
            or {'error': str} on failure.
        """
        try:
            model_info = _with_retries(self.api.model_info, model_id, files_metadata=include_files)
            return {
                'model_info': model_info,
                'model_id': model_id,
                'has_token': False
            }
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 401:
                return {'error': 'Authentication required for this repository (gated/private).'}
            if status == 404:
                return {'error': f'Model {model_id} not found'}
            if status == 429:
                return {'error': 'Rate limited (anonymous). Please retry later.'}
            return {'error': f'HTTP error {status or "unknown"}'}
        except Exception as e:
            return {'error': f'Unexpected error: {e}'}

    def get_model_readme(self, model_id: str) -> Optional[str]:
        """
        Download and return README.md content for a public model.
        """
        try:
            readme_path = _with_retries(
                hf_hub_download,
                repo_id=model_id,
                filename="README.md",
                repo_type="model",
                token=None,
                tqdm_class=None  # Disable progress bar
            )
            with open(readme_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

    def get_model_card_data(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract structured data from the model card (public models).
        """
        try:
            info = _with_retries(self.api.model_info, model_id)
            card = getattr(info, "cardData", None)
            if not card:
                return {}

            # cardData may be a dict-like or an object depending on hub libs/versions
            def _get(src, key, default=None):
                if isinstance(src, dict):
                    return src.get(key, default)
                return getattr(src, key, default)

            result = {
                'datasets': _get(card, 'datasets', []),
                'language': _get(card, 'language', []),
                'license': _get(card, 'license', None),
                'tags': _get(card, 'tags', []),
                'metrics': _get(card, 'metrics', []),
                'library_name': _get(card, 'library_name', None),
            }

            # Add other non-private fields if present
            if not isinstance(card, dict):
                for attr in dir(card):
                    if not attr.startswith('_') and attr not in result:
                        try:
                            val = getattr(card, attr)
                            if not callable(val):
                                result[attr] = val
                        except Exception:
                            pass
            else:
                for k, v in card.items():
                    if k not in result:
                        result[k] = v

            return result

        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                return {'error': 'Authentication required for this repository (gated/private).'}
            if status == 404:
                return {'error': f'Model {model_id} not found'}
            if status == 429:
                return {'error': 'Rate limited (anonymous). Please retry later.'}
            return {'error': f'HTTP error {status or "unknown"}'}
        except Exception:
            return None

    def search_models(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Anonymous search for public models.
        """
        try:
            models = _with_retries(self.api.list_models, search=query, limit=limit)
            return [{'id': m.modelId, 'downloads': getattr(m, 'downloads', None)} for m in models]
        except Exception:
            return []

    def download_model(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Download a public model snapshot to local storage (anonymous).
        """
        try:
            # Honor optional cache_dir relative to project root (keep same behavior)
            if cache_dir:
                # Adjust this root jump if your project layout differs
                # (kept identical to your original for drop-in compatibility)
                project_root = Path(__file__).parent.parent.parent.parent
                cache_path = project_root / cache_dir
                cache_path.mkdir(parents=True, exist_ok=True)
                cache_dir = str(cache_path)

            # Disable progress bars by setting environment variable or using tqdm_class
            import os
            old_disable_progress = os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS', '0')
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

            try:
                local_path = _with_retries(
                    snapshot_download,
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    token=None,
                    allow_patterns=include_patterns,
                    ignore_patterns=ignore_patterns
                )
            finally:
                # Restore original setting
                os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = old_disable_progress

            return {
                'success': True,
                'local_path': local_path,
                'model_id': model_id,
                'cache_dir': cache_dir
            }

        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = 'Authentication required for this repository (gated/private).' if status in (401, 403) else f'HTTP error {status or "unknown"}'
            return {'success': False, 'error': msg, 'model_id': model_id}
        except Exception as e:
            return {'success': False, 'error': str(e), 'model_id': model_id}

    def load_model_for_inference(self, model_id: str, local_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model/tokenizer from local disk for inference (no remote auth).
        If local_path is None, downloads snapshot anonymously first.
        """
        try:
            try:
                from transformers import AutoModel, AutoTokenizer, AutoConfig
                # Suppress transformers logging and progress bars
                import transformers
                transformers.logging.set_verbosity_error()
            except ImportError:
                return {
                    'success': False,
                    'error': 'transformers not installed. Install with: pip install transformers'
                }

            if local_path is None:
                dl = self.download_model(model_id)
                if not dl.get('success'):
                    return dl
                local_path = dl['local_path']

            config = AutoConfig.from_pretrained(local_path, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            model = AutoModel.from_pretrained(local_path, local_files_only=True)

            return {
                'success': True,
                'model': model,
                'tokenizer': tokenizer,
                'config': config,
                'local_path': local_path,
                'model_id': model_id
            }

        except Exception as e:
            return {'success': False, 'error': str(e), 'model_id': model_id}

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch dataset information from HuggingFace (public datasets only).

        Args:
            dataset_id: Dataset identifier (e.g., "squad", "glue")

        Returns:
            Dict with dataset info or error message
        """
        try:
            dataset_info = _with_retries(self.api.dataset_info, dataset_id)
            return {
                'dataset_info': dataset_info,
                'dataset_id': dataset_id,
                'has_token': False
            }
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 401:
                return {'error': 'Authentication required for this dataset (gated/private).'}
            if status == 404:
                return {'error': f'Dataset {dataset_id} not found'}
            if status == 429:
                return {'error': 'Rate limited (anonymous). Please retry later.'}
            return {'error': f'HTTP error {status or "unknown"}'}
        except Exception as e:
            return {'error': f'Unexpected error: {e}'}

    def get_dataset_card_data(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract structured data from the dataset card (public datasets).

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dict with dataset metadata or None on failure
        """
        try:
            info = _with_retries(self.api.dataset_info, dataset_id)
            card = getattr(info, "cardData", None)
            if not card:
                return {}

            def _get(src, key, default=None):
                if isinstance(src, dict):
                    return src.get(key, default)
                return getattr(src, key, default)

            result = {
                'language': _get(card, 'language', []),
                'license': _get(card, 'license', None),
                'tags': _get(card, 'tags', []),
                'task_categories': _get(card, 'task_categories', []),
                'task_ids': _get(card, 'task_ids', []),
                'multilinguality': _get(card, 'multilinguality', None),
                'size_categories': _get(card, 'size_categories', []),
                'source_datasets': _get(card, 'source_datasets', []),
            }

            # Add dataset-specific metadata
            if hasattr(info, 'downloads'):
                result['downloads'] = info.downloads
            if hasattr(info, 'likes'):
                result['likes'] = info.likes
            if hasattr(info, 'created_at'):
                result['created_at'] = str(info.created_at)
            if hasattr(info, 'last_modified'):
                result['last_modified'] = str(info.last_modified)

            # Add other fields if present
            if not isinstance(card, dict):
                for attr in dir(card):
                    if not attr.startswith('_') and attr not in result:
                        try:
                            val = getattr(card, attr)
                            if not callable(val):
                                result[attr] = val
                        except Exception:
                            pass
            else:
                for k, v in card.items():
                    if k not in result:
                        result[k] = v

            return result

        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                return {'error': 'Authentication required for this dataset (gated/private).'}
            if status == 404:
                return {'error': f'Dataset {dataset_id} not found'}
            if status == 429:
                return {'error': 'Rate limited (anonymous). Please retry later.'}
            return {'error': f'HTTP error {status or "unknown"}'}
        except Exception:
            return None

    def search_datasets(self, query: str = None, limit: int = 10, task: str = None) -> List[Dict[str, Any]]:
        """
        Anonymous search for public datasets.

        Args:
            query: Search query string
            limit: Maximum number of results
            task: Filter by task category

        Returns:
            List of dataset info dictionaries
        """
        try:
            datasets = _with_retries(
                self.api.list_datasets,
                search=query,
                limit=limit,
                task=task
            )
            return [
                {
                    'id': d.id,
                    'downloads': getattr(d, 'downloads', None),
                    'likes': getattr(d, 'likes', None),
                    'tags': getattr(d, 'tags', [])
                }
                for d in datasets
            ]
        except Exception:
            return []

    def get_dataset_readme(self, dataset_id: str) -> Optional[str]:
        """
        Download and return README.md content for a public dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            README content as string or None on failure
        """
        try:
            readme_path = _with_retries(
                hf_hub_download,
                repo_id=dataset_id,
                filename="README.md",
                repo_type="dataset",
                token=None,
                tqdm_class=None  # Disable progress bar
            )
            with open(readme_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

    def test_connection(self) -> Dict[str, Any]:
        """
        Simple anonymous connectivity check.
        """
        result = {
            'has_token': False,
            'token_length': 0,
            'connection_ok': False,
            'error': None
        }
        try:
            models = list(_with_retries(self.api.list_models, limit=1))
            result['connection_ok'] = True
            result['sample_model'] = models[0].modelId if models else None
        except Exception as e:
            result['error'] = str(e)
        return result


# Singleton instance for easy use across metrics
hf_client = HuggingFaceClient()
