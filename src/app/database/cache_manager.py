#!/usr/bin/env python3
"""
Cache management utility for the metrics database.

Usage:
    python cache_manager.py stats    # Show cache statistics
    python cache_manager.py clear    # Clear all cache entries
    python cache_manager.py clear 7  # Clear entries older than 7 days
"""

import sys
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # Go up to project root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.database import get_database

def show_stats():
    """Show cache statistics."""
    db = get_database()
    stats = db.get_cache_stats()

    print("=== METRICS CACHE STATISTICS ===")
    print(f"Database: {stats['database_path']}")
    print(f"Total cached entries: {stats['total_entries']}")
    print()

    if stats['top_models']:
        print("Top cached models:")
        for model_url, count in stats['top_models']:
            model_name = model_url.split('/')[-1] if '/' in model_url else model_url
            print(f"  {model_name}: {count} entries")
        print()

    if stats['recent_activity']:
        print("Recent cache activity:")
        for date, count in stats['recent_activity']:
            print(f"  {date}: {count} entries")

def clear_cache(older_than_days=None):
    """Clear cache entries."""
    db = get_database()

    if older_than_days:
        print(f"Clearing cache entries older than {older_than_days} days...")
        db.clear_cache(older_than_days=older_than_days)
        print("✓ Old entries cleared")
    else:
        print("Clearing all cache entries...")
        db.clear_cache()
        print("✓ Cache cleared")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    command = sys.argv[1]

    if command == "stats":
        show_stats()
    elif command == "clear":
        if len(sys.argv) > 2:
            try:
                days = int(sys.argv[2])
                clear_cache(days)
            except ValueError:
                print("Error: Days must be a number")
                return 1
        else:
            clear_cache()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())