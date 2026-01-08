#!/usr/bin/env python3
"""
Main Entry Point for MemoryMover DUT Tests

This is the standard entry point for running all tests for the MemoryMover DUT.
Follows the chip_verification skill format.

Usage:
    python main.py                    # Run all default test cases
    python main.py --test_case=test_copy_basic        # Run single test
    python main.py --test_case=test_copy_large --test_case=test_copy_overlap  # Run multiple tests
"""
import argparse
import sys
from pathlib import Path
from traceback import format_exc

# Add testbench root to path
# TB_ROOT is the examples/05_buffer directory (parent of tb/)
TB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TB_ROOT / "tb"))

from torchbit.runner import Runner, FileConfig, BuildConfig, DEFAULT_VERILATOR_BUILD_CONFIG


def get_file_config():
    """Create and return the FileConfig for this DUT."""
    rtl_dir = TB_ROOT / "src" / "rtl"
    return FileConfig(
        name="memory_mover",
        sources=[
            rtl_dir / "memory_mover.v",
        ],
        top_design="memory_mover",
        includes=[]
    )


def get_build_config():
    """Create and return the BuildConfig."""
    return DEFAULT_VERILATOR_BUILD_CONFIG


def run_tests(test_cases=None):
    """Run the specified test cases.

    Args:
        test_cases: List of test case names to run. If None, runs default tests.

    Returns:
        0 if all tests pass, 1 otherwise.
    """
    file_config = get_file_config()
    build_config = get_build_config()

    # Create runner
    runner = Runner(file_config, build_config, current_dir=TB_ROOT)

    # Default test cases if none specified
    if test_cases is None or len(test_cases) == 0:
        test_cases = [
            "test_copy_basic",
            "test_copy_large",
            "test_copy_overlap",
            "test_buffer_roundtrip",
        ]

    # Add the 'tb/' prefix to test module path
    test_modules = [f"tb.dut_buffer.tests.{tc}" for tc in test_cases]

    # Run each test case
    failed = []
    for test_module in test_modules:
        test_name = test_module.split(".")[-1]
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)

        try:
            runner.test(test_module)
            print(f"\n[PASS] {test_name}")
        except Exception as e:
            print(f"\n[FAIL] {test_name}: {e}")
            print(format_exc())
            failed.append(test_name)

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print('='*60)
    print(f"Total: {len(test_cases)}, Passed: {len(test_cases) - len(failed)}, Failed: {len(failed)}")

    if failed:
        print(f"\nFailed tests:")
        for name in failed:
            print(f"  - {name}")
        return 1

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run MemoryMover DUT tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                                    # Run all tests
    python main.py --test_case=test_copy_basic        # Run single test
    python main.py --test_case=test_copy_basic --test_case=test_copy_large  # Run multiple
        """
    )

    parser.add_argument(
        "--test_case",
        action="append",
        help="Test case to run (can be specified multiple times)"
    )

    args = parser.parse_args()

    # Run tests
    return run_tests(args.test_case)


if __name__ == "__main__":
    sys.exit(main())
