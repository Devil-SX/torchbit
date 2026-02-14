#!/usr/bin/env python3
"""
Main Entry Point for Pipe DUT Tests (UVM style)

Usage:
    python main.py                                  # Run all default tests
    python main.py --test_case=test_basic           # Run single test
    python main.py --test_case=test_backpressure    # Run backpressure test
"""
import argparse
import sys
from pathlib import Path
from traceback import format_exc

TB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TB_ROOT / "tb"))

from torchbit.runner import Runner, FileConfig, DEFAULT_VERILATOR_BUILD_CONFIG


def get_file_config():
    """Create FileConfig for the Pipe DUT."""
    rtl_dir = TB_ROOT / "src" / "rtl"
    return FileConfig(
        name="pipe",
        sources=[
            rtl_dir / "pipe.sv",
        ],
        top_design="Pipe",
        includes=[],
    )


def run_tests(test_cases=None):
    """Run the specified test cases.

    Args:
        test_cases: List of test case names. If None, runs all defaults.

    Returns:
        0 if all pass, 1 otherwise.
    """
    file_config = get_file_config()
    build_config = DEFAULT_VERILATOR_BUILD_CONFIG
    runner = Runner(file_config, build_config, current_dir=TB_ROOT)

    if test_cases is None or len(test_cases) == 0:
        test_cases = [
            "test_basic",
            "test_backpressure",
        ]

    test_modules = [f"tb.dut_pipe.tests.{tc}" for tc in test_cases]

    failed = []
    for test_module in test_modules:
        test_name = test_module.split(".")[-1]
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print("=" * 60)

        try:
            runner.test(test_module)
            print(f"\n[PASS] {test_name}")
        except Exception as e:
            print(f"\n[FAIL] {test_name}: {e}")
            print(format_exc())
            failed.append(test_name)

    print(f"\n{'=' * 60}")
    print("Test Summary")
    print("=" * 60)
    print(f"Total: {len(test_cases)}, Passed: {len(test_cases) - len(failed)}, Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for name in failed:
            print(f"  - {name}")
        return 1

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Pipe DUT tests (UVM style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                                  # Run all tests
    python main.py --test_case=test_basic           # Run single test
    python main.py --test_case=test_backpressure    # Backpressure test
        """,
    )
    parser.add_argument(
        "--test_case",
        action="append",
        help="Test case to run (can be specified multiple times)",
    )
    args = parser.parse_args()
    return run_tests(args.test_case)


if __name__ == "__main__":
    sys.exit(main())
