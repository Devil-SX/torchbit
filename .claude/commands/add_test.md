---
name: add_test
description: Add tests for specified file/module with optional performance benchmarks
argument-hint: <file-path> [performance|functional|both]
---

# Claude Command: add_test

Generate and add tests for a specified file or module. Supports functional tests, performance benchmarks, or both.

## Behavior (based on argument)

| Argument | Action |
|----------|--------|
| `<file-path>` | Add functional tests for the specified file/module |
| `<file-path> functional` | Add functional tests only |
| `<file-path> performance` | Add performance benchmarks with size scaling |
| `<file-path> both` | Add both functional tests and performance benchmarks |

## Steps

### 1. Analyze Target File

1. Parse the argument to get the file path and test type
2. Read the target file to understand:
   - Module/class structure
   - Public functions and methods
   - Function signatures and return types
   - Dependencies
3. Determine the corresponding test file path:
   - If target is `torchbit/<module>.py`, test file is `tests/test_<module>.py`
   - If target is `torchbit/<subdir>/<module>.py`, test file is `tests/test_<module>.py`
   - If target is `examples/<name>/`, test file is `examples/<name>/tb/tests/`

### 2. Generate Functional Tests

For each public function/method:

1. **Test Discovery:**
   - Read function signature and docstring
   - Identify parameter types and constraints
   - Determine return type

2. **Test Generation:**
   - `test_<function>_basic` - Basic smoke test with valid inputs
   - `test_<function>_edge_cases` - Edge cases (empty, single element, max values)
   - `test_<function>_error_handling` - Invalid inputs raise appropriate errors
   - `test_<function>_type_validation` - Type checking for parameters

3. **Test Structure:**
   ```python
   class Test<ClassName>:
       def setup_method(self):
           # Common setup

       def test_<function>_basic(self):
           # Basic functionality test

       def test_<function>_edge_cases(self):
           # Edge case testing
   ```

### 3. Generate Performance Benchmarks

For performance test type:

1. **Size Scaling:**
   - Define size scales: small (n=10), medium (n=100), large (n=1000), xlarge (n=10000)
   - For each scale, measure:
     - Execution time (ms)
     - Memory usage (MB)
     - Throughput (ops/sec)

2. **Profiling:**
   - Use Python's `cProfile` to identify bottlenecks
   - Break down function execution time by step
   - Calculate percentage of time spent in each step

3. **Benchmark Structure with Profiling:**
   ```python
   import cProfile
   import pstats
   import io

   class Test<ClassName>Performance:
       @pytest.mark.parametrize("size", [10, 100, 1000, 10000])
       def test_<function>_performance(self, size, benchmark):
           # Performance test
           result = benchmark(self.<function>, generate_test_data(size))

       def test_<function>_profile(self, size):
           # Profiling test
           profiler = cProfile.Profile()
           profiler.enable()
           result = self.<function>(generate_test_data(size))
           profiler.disable()

           # Parse and print profiling results
           s = io.StringIO()
           stats = pstats.Stats(profiler, stream=s)
           stats.strip_dirs()
           stats.sort_stats('cumulative')
           stats.print_stats(20)  # Top 20 functions
   ```

4. **Report with Profiling:**
   - Generate `tests/reports/performance_<module>_<date>.md`
   - Include profiling breakdown showing:
     - Function name and step
     - Cumulative time (ms)
     - Percentage of total time
     - Number of calls
   - Format:
     ```markdown
     ## <function_name> Profiling Breakdown

     | Step | Calls | Total Time (ms) | Time per Call (ms) | % |
     |------|-------|-----------------|-------------------|---|
     | Step 1: Description | 1000 | 50.0 | 0.05 | 50% |
     | Step 2: Description | 100 | 30.0 | 0.30 | 30% |
     | Step 3: Description | 1 | 20.0 | 20.00 | 20% |

     **Total:** 100.0 ms
     ```

5. **Profiling Guidelines:**
   - Profile the **largest** size (xlarge) for most representative results
   - Group related operations into logical steps:
     - Data preparation/generation
     - Main computation/operation
     - Result processing/validation
     - Memory allocation/copy
   - Calculate percentage: `(step_time / total_time) * 100`
   - Highlight bottlenecks (>30% of total time)

### 4. Write Test File

1. Check if test file exists:
   - If exists: append new tests to existing file
   - If not: create new file with imports and structure

2. Add imports:
   ```python
   import pytest
   import torch
   from torchbit.<module> import <ClassName>
   ```

3. Add fixtures if needed:
   ```python
   @pytest.fixture
   def sample_data():
       return ...
   ```

### 5. Run Tests

1. Run newly created/updated tests:
   ```bash
   pytest tests/test_<module>.py -v
   ```

2. For performance tests, run with pytest-benchmark:
   ```bash
   pytest tests/test_<module>_performance.py -v --benchmark-only
   ```

### 6. Generate Report (for performance tests)

1. Parse benchmark output
2. Create markdown report with:
   - Summary table of all benchmarks
   - Detailed per-function tables
   - **Profiling breakdown** for each operation showing:
     - Step-by-step time distribution
     - Percentage of total time per step
     - Number of calls per step
     - Bottleneck identification (>30%)
   - Performance trends (scale linearly/O(n)/O(n²))
   - Recommendations for optimization if needed

3. Save report to `tests/reports/performance_<module>_<date>.md`

4. Report Format:
   ```markdown
   # Performance Report: <module>

   ## Summary

   | Operation | Min Throughput | Max Throughput | Bottleneck Step |
   |-----------|----------------|----------------|-----------------|
   | op1 | ... | ... | Step: Data conversion (45%) |
   | op2 | ... | ... | Step: Memory copy (60%) |

   ## <operation_name> Performance

   ### Benchmark Results

   | Size | Time (ms) | Memory (MB) | Throughput |
   |------|-----------|-------------|------------|
   | ... | ... | ... | ... |

   ### Profiling Breakdown (Size: 10000)

   | Step | Calls | Total (ms) | Per Call (ms) | % | Status |
   |------|-------|------------|---------------|---|--------|
   | Validation | 1 | 0.5 | 0.5 | 0.5% | |
   | Data prep | 10000 | 20.0 | 0.002 | 20% | |
   | Main op | 1 | 75.0 | 75.0 | **75%** | ⚠️ Bottleneck |
   | Cleanup | 1 | 4.5 | 4.5 | 4.5% | |

   **Total:** 100.0 ms

   ### Recommendations

   - Consider optimizing the Main op step (75% of time)
   - ...
   ```

## Notes

- Uses pytest as the test framework
- Uses cProfile for performance profiling
- Test fixtures follow `tests/conftest.py` conventions
- Performance reports use markdown tables for readability
- All functional tests should be independent and order-independent
- **Profiling requirements:**
  - Profile the largest size (xlarge) for representative results
  - Break down operations into logical steps with clear descriptions
  - Highlight bottlenecks (>30% of total time) with ⚠️ emoji
  - Show percentage breakdown: `(step_time / total_time) * 100`
