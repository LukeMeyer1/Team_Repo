"""
HuggingFace API client wrapper with authentication and error handling.

Centralized way to interact with HuggingFace APIs across metrics.

How to use this:
-----------
from integrations.huggingface_client import hf_client

# Get model information
info = hf_client.get_model_info("bert-base-uncased")
if info and 'error' not in info:
    print(f"Model has {info['model_info'].downloads} downloads")

# Get model card data
card = hf_client.get_model_card_data("bert-base-uncased")
if card:
    print(f"Datasets: {card.get('datasets', [])}")

# Get README content
readme = hf_client.get_model_readme("bert-base-uncased")
if readme:
    print(f"README length: {len(readme)} chars")

# Test connection
status = hf_client.test_connection()
print(f"Connected: {status['connection_ok']}")

# Download a model locally
download_result = hf_client.download_model("distilbert-base-uncased",
                                         cache_dir="models/distilbert")
if download_result['success']:
    print(f"Downloaded to: {download_result['local_path']}")

# Load model for inference
model_result = hf_client.load_model_for_inference("distilbert-base-uncased")
if model_result['success']:
    model = model_result['model']
    tokenizer = model_result['tokenizer']
    # Use model and tokenizer for inference...
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


def _load_env_file():
    """Load environment variables from .env file in integrations folder."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


class HuggingFaceClient:
    """
    Centralized HuggingFace API client with authentication and error handling.

    Automatically loads HUGGINGFACE_TOKEN from environment and provides
    convenient methods for common operations across metrics.
    """

    def __init__(self):
        """Initialize client with token from environment."""
        # Load .env file if it exists
        _load_env_file()

        # Get token from environment
        self.token = os.getenv('HUGGINGFACE_TOKEN')
        self.api = HfApi(token=self.token) if self.token else HfApi()

    def get_model_info(self, model_id: str, include_files: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetch model information from HuggingFace.

        Args:
            model_id: HuggingFace model identifier (e.g., "bert-base-uncased")
            include_files: Whether to include file metadata

        Returns:
            Dictionary with model information or None if failed
        """
        try:
            model_info = self.api.model_info(model_id, files_metadata=include_files)
            return {
                'model_info': model_info,
                'model_id': model_id,
                'has_token': self.token is not None
            }
        except HfHubHTTPError as e:
            error_msg = f'HTTP error {e.response.status_code}'
            if e.response.status_code == 401:
                error_msg = 'Authentication failed - check HUGGINGFACE_TOKEN'
            elif e.response.status_code == 404:
                error_msg = f'Model {model_id} not found'
            elif e.response.status_code == 429:
                error_msg = 'Rate limited - too many requests'
            print(f"HuggingFace API error for {model_id}: {error_msg}")
            return {'error': error_msg}
        except Exception as e:
            error_msg = f'Unexpected error: {str(e)}'
            print(f"HuggingFace API error for {model_id}: {error_msg}")
            return {'error': error_msg}

    def get_model_readme(self, model_id: str) -> Optional[str]:
        """
        Download and read model README.md content.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            README content as string or None if failed
        """
        try:
            readme_path = hf_hub_download(
                repo_id=model_id,
                filename="README.md",
                repo_type="model",
                token=self.token
            )

            with open(readme_path, 'r', encoding='utf-8') as f:
                return f.read()

        except Exception:
            return None

    def get_model_card_data(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract structured data from model card.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            Dictionary with card data or None if failed
        """
        try:
            model_info = self.api.model_info(model_id)

            if hasattr(model_info, 'cardData') and model_info.cardData:
                card_data = model_info.cardData

                # Extract common fields
                result = {
                    'datasets': getattr(card_data, 'datasets', []),
                    'language': getattr(card_data, 'language', []),
                    'license': getattr(card_data, 'license', None),
                    'tags': getattr(card_data, 'tags', []),
                    'metrics': getattr(card_data, 'metrics', []),
                    'library_name': getattr(card_data, 'library_name', None),
                }

                # Add any other fields that might be present
                for attr in dir(card_data):
                    if not attr.startswith('_') and attr not in result:
                        try:
                            value = getattr(card_data, attr)
                            if not callable(value):
                                result[attr] = value
                        except:
                            pass

                return result

            return {}

        except Exception:
            return None

    def search_models(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for models on HuggingFace Hub.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of model information dictionaries
        """
        try:
            models = self.api.list_models(search=query, limit=limit)
            return [{'id': model.modelId, 'downloads': model.downloads} for model in models]
        except Exception:
            return []

    def download_model(self, model_id: str, cache_dir: Optional[str] = None,
                       include_patterns: Optional[List[str]] = None,
                       ignore_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Download a complete model to local storage.

        Args:
            model_id: HuggingFace model identifier (e.g., "bert-base-uncased")
            cache_dir: Local directory to store the model (defaults to HF cache)
            include_patterns: Patterns for files to include (e.g., ["*.json", "*.bin"])
            ignore_patterns: Patterns for files to ignore (e.g., ["*.h5"])

        Returns:
            Dictionary with download results including local path
        """
        try:
            # Set up cache directory relative to project if specified
            if cache_dir:
                project_root = Path(__file__).parent.parent.parent.parent
                cache_path = project_root / cache_dir
                cache_path.mkdir(parents=True, exist_ok=True)
                cache_dir = str(cache_path)

            # Download the model
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                token=self.token,
                allow_patterns=include_patterns,
                ignore_patterns=ignore_patterns
            )

            return {
                'success': True,
                'local_path': local_path,
                'model_id': model_id,
                'cache_dir': cache_dir
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_id': model_id
            }

    def load_model_for_inference(self, model_id: str, local_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model and tokenizer for inference from local storage.

        Args:
            model_id: HuggingFace model identifier
            local_path: Path to locally downloaded model (if None, downloads automatically)

        Returns:
            Dictionary with loaded model, tokenizer, and metadata
        """
        try:
            # Import transformers here to avoid requiring it if not used
            try:
                from transformers import AutoModel, AutoTokenizer, AutoConfig
            except ImportError:
                return {
                    'success': False,
                    'error': 'transformers library not installed. Install with: pip install transformers'
                }

            # Download model if not provided locally
            if local_path is None:
                download_result = self.download_model(model_id)
                if not download_result['success']:
                    return download_result
                local_path = download_result['local_path']

            # Load config to determine model type
            config = AutoConfig.from_pretrained(local_path, local_files_only=True)

            # Load tokenizer and model
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
            return {
                'success': False,
                'error': str(e),
                'model_id': model_id
            }

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection and authentication status.

        Returns:
            Dictionary with connection status information
        """
        result = {
            'has_token': self.token is not None,
            'token_length': len(self.token) if self.token else 0,
            'connection_ok': False,
            'error': None
        }

        try:
            # Try a simple API call
            models = list(self.api.list_models(limit=1))
            result['connection_ok'] = True
            result['sample_model'] = models[0].modelId if models else None
        except Exception as e:
            result['error'] = str(e)

        return result


# Singleton instance for easy use across metrics
hf_client = HuggingFaceClient()