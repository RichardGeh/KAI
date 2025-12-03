#!/usr/bin/env python
"""Debug script to identify pytest import issue."""
import sys
import traceback
from pathlib import Path

# Mimic what conftest.py does
sys.path.insert(0, str(Path(__file__).parent))

print("Python path:", sys.path[:3])
print()

try:
    print("Importing component_1_netzwerk...")

    print("  SUCCESS")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

try:
    print("\nImporting component_11_embedding_service...")

    print("  SUCCESS")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

print("\nAll imports successful!")
