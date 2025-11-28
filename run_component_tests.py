"""
Manual test runner for components 25, 26, 27 to bypass conftest.py issues.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import subprocess
import time

def run_tests():
    """Run pytest tests with proper PYTHONPATH"""
    test_files = [
        "tests/test_command_suggestions.py",
        "tests/test_regex_validator.py",
        "tests/test_adaptive_pattern_recognition.py"
    ]

    components = [
        "component_26_command_suggestions",
        "component_27_regex_validator",
        "component_25_adaptive_thresholds"
    ]

    print("="*80)
    print("KAI Component Test Suite - Components 25, 26, 27")
    print("="*80)
    print()

    # Set PYTHONPATH environment variable
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent)

    # Run each test file individually
    for i, test_file in enumerate(test_files):
        print(f"\n[{i+1}/{len(test_files)}] Running {test_file}...")
        print("-"*80)

        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            f"--cov={components[i]}",
            "--cov-report=term-missing",
            "--tb=short"
        ]

        start = time.time()
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        duration = time.time() - start

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        print(f"\nTest execution time: {duration:.2f}s")
        print(f"Exit code: {result.returncode}")

        if result.returncode != 0:
            print(f"[FEHLER] Tests fehlgeschlagen in {test_file}")
        else:
            print(f"[OK] Tests bestanden in {test_file}")

    # Run all tests together for combined coverage
    print("\n" + "="*80)
    print("COMBINED COVERAGE ANALYSIS")
    print("="*80)

    cmd = [
        sys.executable, "-m", "pytest",
        *test_files,
        "-v",
        "--cov=component_25_adaptive_thresholds",
        "--cov=component_26_command_suggestions",
        "--cov=component_27_regex_validator",
        "--cov-report=term-missing",
        "--tb=short"
    ]

    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    duration = time.time() - start

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    print(f"\nTotal execution time: {duration:.2f}s")
    print(f"Exit code: {result.returncode}")

    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
