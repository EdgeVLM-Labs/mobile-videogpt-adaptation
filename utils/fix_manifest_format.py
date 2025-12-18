#!/usr/bin/env python3
"""
Fix manifest.json to convert dict entries to string format for consistency.
This script converts augmented video entries from dict format to string format.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "dataset"
MANIFEST_FILE = BASE_DIR / "manifest.json"
BACKUP_FILE = BASE_DIR / "manifest.json.backup"

def main():
    print("="*60)
    print("Fixing manifest.json format")
    print("="*60)

    # Load manifest
    with open(MANIFEST_FILE, 'r') as f:
        manifest = json.load(f)

    # Create backup
    with open(BACKUP_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Created backup: {BACKUP_FILE}")

    # Fix entries
    fixed_count = 0
    for path, value in manifest.items():
        if isinstance(value, dict):
            # Extract exercise name from path
            if '/' in path:
                exercise_name = path.split('/')[0]
            else:
                exercise_name = 'unknown'

            manifest[path] = exercise_name
            fixed_count += 1

    # Save fixed manifest
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Fixed {fixed_count} entries")
    print(f"✓ Updated {MANIFEST_FILE}")
    print("="*60)
    print("Manifest.json format fixed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
