"""
SQLite database for caching model evaluation results.

This module provides caching functionality to avoid recomputing metrics
for the same URLs, especially expensive operations like model downloads.
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime


class MetricsDatabase:
    """SQLite database for caching model evaluation results."""

    def __init__(self, db_path: str = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Defaults to 'metrics_cache.db' in database folder.
        """
        if db_path is None:
            # Default to metrics_cache.db in the database folder
            db_path = Path(__file__).parent / "metrics_cache.db"

        self.db_path = str(db_path)
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url_hash TEXT UNIQUE NOT NULL,
                    model_url TEXT NOT NULL,
                    dataset_urls TEXT,  -- JSON array
                    code_urls TEXT,     -- JSON array
                    result_json TEXT NOT NULL,  -- Full NDJSON result
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_url_hash ON evaluation_cache(url_hash)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_url ON evaluation_cache(model_url)
            """)

    def _compute_cache_key(self, model_url: str, dataset_urls: List[str] = None,
                          code_urls: List[str] = None) -> str:
        """
        Compute cache key for the given URL combination.

        Args:
            model_url: Primary model URL
            dataset_urls: Optional dataset URLs
            code_urls: Optional code URLs

        Returns:
            SHA256 hash of the URL combination
        """
        if dataset_urls is None:
            dataset_urls = []
        if code_urls is None:
            code_urls = []

        # Create deterministic string representation
        cache_input = {
            'model_url': model_url,
            'dataset_urls': sorted(dataset_urls),  # Sort for consistency
            'code_urls': sorted(code_urls)
        }
        cache_string = json.dumps(cache_input, sort_keys=True)

        # Return SHA256 hash
        return hashlib.sha256(cache_string.encode()).hexdigest()

    def get_cached_result(self, model_url: str, dataset_urls: List[str] = None,
                         code_urls: List[str] = None) -> Optional[str]:
        """
        Get cached evaluation result if available.

        Args:
            model_url: Primary model URL
            dataset_urls: Optional dataset URLs
            code_urls: Optional code URLs

        Returns:
            Cached NDJSON result string or None if not found
        """
        cache_key = self._compute_cache_key(model_url, dataset_urls, code_urls)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT result_json FROM evaluation_cache WHERE url_hash = ?",
                (cache_key,)
            )
            row = cursor.fetchone()

            if row:
                return row[0]
            return None

    def cache_result(self, model_url: str, result_json: str,
                    dataset_urls: List[str] = None, code_urls: List[str] = None):
        """
        Cache evaluation result.

        Args:
            model_url: Primary model URL
            result_json: NDJSON result string to cache
            dataset_urls: Optional dataset URLs
            code_urls: Optional code URLs
        """
        if dataset_urls is None:
            dataset_urls = []
        if code_urls is None:
            code_urls = []

        cache_key = self._compute_cache_key(model_url, dataset_urls, code_urls)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO evaluation_cache
                (url_hash, model_url, dataset_urls, code_urls, result_json, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                cache_key,
                model_url,
                json.dumps(dataset_urls),
                json.dumps(code_urls),
                result_json
            ))

    def clear_cache(self, older_than_days: int = None):
        """
        Clear cached results.

        Args:
            older_than_days: If specified, only clear entries older than this many days
        """
        with sqlite3.connect(self.db_path) as conn:
            if older_than_days is not None:
                conn.execute("""
                    DELETE FROM evaluation_cache
                    WHERE created_at < datetime('now', '-{} days')
                """.format(older_than_days))
            else:
                conn.execute("DELETE FROM evaluation_cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM evaluation_cache")
            total_entries = cursor.fetchone()[0]

            cursor = conn.execute("""
                SELECT model_url, COUNT(*) as count
                FROM evaluation_cache
                GROUP BY model_url
                ORDER BY count DESC
                LIMIT 5
            """)
            top_models = cursor.fetchall()

            cursor = conn.execute("""
                SELECT datetime(created_at) as date, COUNT(*) as count
                FROM evaluation_cache
                GROUP BY date(created_at)
                ORDER BY date DESC
                LIMIT 7
            """)
            recent_activity = cursor.fetchall()

            return {
                'total_entries': total_entries,
                'top_models': top_models,
                'recent_activity': recent_activity,
                'database_path': self.db_path
            }


# Global database instance
_db_instance = None

def get_database() -> MetricsDatabase:
    """Get global database instance (singleton pattern)."""
    global _db_instance
    if _db_instance is None:
        _db_instance = MetricsDatabase()
    return _db_instance