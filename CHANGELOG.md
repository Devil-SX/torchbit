# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Docs:** Added `doc/en/verification_api.md` — full API comparison document covering Native torchbit vs PyUVM-compatible verification APIs with comparison table, component reference, and workflow examples.
- **Docs:** Added `doc/zh-CN/verification_api.md` — Chinese translation of the verification API reference.
- **Docs:** Added Table of Contents, Verification API section, and Documentation section to both `README.md` and `README.zh-CN.md`.
- **Examples:** Added `09_uvm_pipeline` example demonstrating UVM-style verification with real RTL simulation using TorchbitBFM, TorchbitScoreboard, VectorItem, and TransferStrategy against a parameterized pipeline DUT.
- **Skills:** Enhanced `report_testbench` skill with report skeleton template, testbench topology diagram requirement, and conditional TileMapping/AddressMapping reporting.

## [3.2.0] - 2026-02-11

### Added
- **Examples:** Added `07_uvm_basics` example demonstrating BFM, VectorItem, LogicSequenceItem, Scoreboard, Driver/Monitor, Agent, Env/Test (pure Python, no sim) (#16).
- **Examples:** Added `08_uvm_advanced` example demonstrating ComponentRegistry, CoveragePoint/CoverageGroup, RegisterModel/RegisterBlock (pure Python, no sim) (#17).

### Changed
- **Examples:** Fixed broken API calls in `02_data_convert` for v3.0 canonical names: `from_array`/`to_logic`/`from_logic`/`to_array`/`from_matrix`/`array_to_logic`/`logic_to_array` (#14).
- **Examples:** Fixed broken API calls in `06_tile_mapping` wrapper and tests: `backdoor_load_tensor`/`from_logic`/`to_array` (#14).
- **Examples:** Updated stale comments in `05_buffer` to use canonical method names (#14).
- **Examples:** Simplified `06_tile_mapping` from nested testbench structure to flat structure matching other examples (#15).
- **Examples:** Updated `examples/README.md` with all 8 examples and canonical class names (#14).

## [3.1.0] - 2026-02-11

### Added
- **UVM:** Added `ComponentRegistry` for factory-style component type overrides (#11).
- **UVM:** Added `CoveragePoint` and `CoverageGroup` for functional coverage tracking with bin-based sampling and reporting (#11).
- **UVM:** Added `RegisterModel` and `RegisterBlock` for Register Abstraction Layer (RAL) integration with BitStruct (#11).
  - Field-level read/write, packed value roundtrip, and Buffer backdoor access.
- **Tests:** Added `test_uvm_advanced.py` with 16 tests for factory, coverage, and RAL components.

## [3.0.0] - 2026-02-11

### Added
- **UVM:** Added `TorchbitScoreboard` for expected vs actual comparison with match/mismatch tracking and reporting (#10).
- **UVM:** Added `TorchbitEnv` and `create_uvm_env()` factory for assembling agents + scoreboard (#10).
- **UVM:** Added `TorchbitTest` and `create_uvm_test()` factory for UVM test orchestration (#10).
- **Tests:** Added `test_uvm_env.py` with 11 tests for scoreboard, env, and test components.

### Removed
- **BREAKING:** Removed all deprecated backward-compatibility aliases (#3):
  - `Vector`: `from_cocotb()`, `from_tensor()`, `to_cocotb()`, `to_int()`, `to_tensor()`
  - `VectorSequence`: `from_tensor()`, `to_tensor()`, `from_int_sequence()`, `to_int_sequence()`
  - `Matrix` class (use `VectorSequence`)
  - `IntSequence` class (use `LogicSequence`)
  - `int_sequence.py` module (use `logic_sequence.py`)
  - `TileMapping.to_int_sequence()` (use `to_logic_sequence()`)
  - `tensor_to_cocotb()` / `cocotb_to_tensor()` (use `array_to_logic()` / `logic_to_array()`)
  - `tensor_to_cocotb_seq()` / `cocotb_seq_to_tensor()` (use `array_to_logic_seq()` / `logic_seq_to_array()`)
  - `Buffer.init_from_matrix()` / `dump_to_matrix()` / `init_from_tensor()` / `dump_to_tensor()`

## [2.6.0] - 2026-02-11

### Added
- **UVM:** Added `torchbit.uvm` package with pyuvm integration layer (#8, #9):
  - `TorchbitBFM`: Bridge between pyuvm components and cocotb DUT signals via InputPort/OutputPort.
  - `VectorItem` / `LogicSequenceItem`: Sequence item wrappers for torchbit data types.
  - `TorchbitDriver` / `TorchbitMonitor` / `TorchbitAgent`: Lightweight wrappers (work without pyuvm).
  - `create_uvm_driver()` / `create_uvm_monitor()` / `create_uvm_agent()`: Factory functions that lazily import pyuvm and return UVM-compatible subclasses.
- **Tools:** Added `ComponentDB` class for UVM-style hierarchical signal path registry (#6):
  - `set()` / `get()`: Register and retrieve signal mappings by dot-separated paths.
  - `find_by_tag()` / `find_by_path()`: Lookup components by tag or glob pattern.
  - `register_component()`: Register components for tag-based discovery.
- **Tools:** Added `from_path()` classmethod to `FIFODriver` and `FIFOReceiver` for ComponentDB-based signal resolution.
- **Tools:** Added `tag` attribute to `FIFODriver` and `FIFOReceiver` for component identification.
- **Package:** Added `uvm` optional dependency group: `pip install torchbit[uvm]` for pyuvm>=4.0.0.
- **Tests:** Added `test_component_db.py` with tests for ComponentDB path registry and component lookup.
- **Tests:** Added `test_uvm.py` with tests for BFM, sequence items, driver/monitor/agent (no pyuvm needed).

### Changed
- **Deprecation:** Added `DeprecationWarning` to all legacy aliases (#3), scheduled for removal in v3.0.0:
  - `Vector`: `from_cocotb()`, `from_tensor()`, `to_cocotb()`, `to_int()`, `to_tensor()`
  - `VectorSequence`: `from_tensor()`, `to_tensor()`, `from_int_sequence()`, `to_int_sequence()`
  - `Matrix` class (use `VectorSequence`)
  - `IntSequence` class (use `LogicSequence`)
  - `TileMapping.to_int_sequence()` (use `to_logic_sequence()`)
  - `tensor_to_cocotb()` / `cocotb_to_tensor()` (use `array_to_logic()` / `logic_to_array()`)
  - `tensor_to_cocotb_seq()` / `cocotb_seq_to_tensor()` (use `array_to_logic_seq()` / `logic_seq_to_array()`)
  - `Buffer`: `init_from_matrix()`, `dump_to_matrix()`, `init_from_tensor()`, `dump_to_tensor()`

### Fixed
- **Examples:** Fixed invalid `BitStruct` constructor calls in `03_bit_struct` example (#2).

## [2.5.0] - 2026-02-11

### Added
- **Tools:** Added `TransferStrategy` abstract base class and four concrete strategies for pluggable transfer timing control:
  - `GreedyStrategy`: Transfer whenever channel is ready (default).
  - `RandomBackpressure`: Randomly stall with configurable probability and optional seed for reproducibility.
  - `BurstStrategy`: Send N items, pause M cycles, repeat.
  - `ThrottledStrategy`: Transfer at most once every N cycles.
- **Tools:** Added `FIFODriver` class for pushing data into DUT input FIFOs (TB → DUT) with explicit `active_high` polarity parameter and pluggable `TransferStrategy`.
- **Tools:** Added `FIFOReceiver` class for capturing data from DUT output FIFOs (DUT → TB) with strategy-controlled ready assertion.
- **Tests:** Added `test_strategy.py` with comprehensive tests for all four transfer strategies (pure Python, no cocotb).
- **Tests:** Added `test_fifo.py` with construction and import tests for FIFODriver/FIFOReceiver.
- **Tests:** Added `test_no_cocotb.py` with 11 subprocess-based tests verifying all core imports and operations work without cocotb.

### Changed
- **Package:** Made cocotb an optional dependency (#13). Pure-Python features (Vector, TileMapping, BitStruct, Buffer, LogicSequence) now work without cocotb installed.
- **Package:** Moved `cocotb>=2.0.0` from required `dependencies` to `[project.optional-dependencies]` in `pyproject.toml`. Added `packaging` to core dependencies.
- **Core:** Replaced top-level `import cocotb` with lazy imports in `vector.py`, `vector_sequence.py`.
- **Tools:** Replaced top-level cocotb imports with lazy imports in `buffer.py`, `driver.py`, `monitor.py`.
- **Runner:** Replaced top-level `cocotb_tools` import with lazy import in `runner.py`.
- **Core:** Updated Vector class docstring to use canonical method names (from_array/to_logic).
- **Core:** Improved `LogicSequence` class docstring with usage patterns and examples.
- **Tiling:** Fixed TileMapping docstring: `IntSequence` → `LogicSequence`.
- **Core:** Replaced bare `print()` with `logging.getLogger(__name__)` in `vector.py` (#5).
- **Debug:** Replaced bare `print()` with `logging.getLogger(__name__)` in `temporal_event.py` (#5).
- **Tools:** Replaced bare `print()` with `logging.getLogger(__name__)` in `monitor.py`.

## [2.4.2] - 2026-02-10

### Added
- **Plugin:** Added Claude Code plugin marketplace configuration (`.claude-plugin/marketplace.json`) for plugin distribution.
- **Skills:** Added `report_testbench` skill for generating testbench architecture reports with ASCII art diagrams, stimulus framework descriptions, and test data documentation.
- **Docs:** Added version and cloc language badges (Python, Markdown, SystemVerilog) to both `README.md` and `README.zh-CN.md`.
- **Docs:** Centered Chinese/English language switch links using HTML `<p align="center">`.

## [2.4.1] - 2026-02-09

### Added
- **Package:** Added `__version__` attribute to `torchbit` package, read from `pyproject.toml` via `importlib.metadata`.
- **Runner:** Added `-f` / `-F` nested filelist support to `read_filelist()`, following EDA command-file conventions (VCS, Xcelium, Verilator, Questa).
- **Runner:** Added `//` comment syntax support in filelist parsing (in addition to existing `#` comments).
- **Runner:** Added silent skipping of known EDA directives (`+incdir+`, `+define+`, `+libext+`, `-v`, `-y`, `-sv`, `-timescale`) in filelist parsing.

## [2.4.0] - 2026-02-08

### Added
- **Core:** Added `LogicSequence` class (`torchbit/core/logic_sequence.py`) as the canonical typed sequence for packed integer values. `IntSequence` preserved as alias.
- **Core:** Added canonical `to_logic()`/`from_logic()` methods to `BitStruct` for packing/unpacking field values. `to_int()`/`from_int()`/`to_cocotb()`/`from_cocotb()` preserved as aliases.
- **Core:** Added canonical `to_logic()`/`from_logic()` (packed integer) and `to_array()`/`from_array()` (1D Tensor) methods to `Vector`. `to_cocotb()`/`from_cocotb()`/`to_int()`/`to_tensor()`/`from_tensor()` preserved as aliases.
- **Core:** Added `array_to_logic()`/`logic_to_array()` module-level shortcuts. `tensor_to_cocotb()`/`cocotb_to_tensor()` preserved as aliases.
- **Core:** Added canonical `to_matrix()`/`from_matrix()` (2D Tensor) and `to_logic_sequence()`/`from_logic_sequence()` methods to `VectorSequence`. `to_tensor()`/`from_tensor()`/`to_int_sequence()`/`from_int_sequence()` preserved as aliases.
- **Tiling:** Added `matrix_to_logic_seq()`/`logic_seq_to_matrix()` low-level shortcuts for direct 2D Matrix ↔ LogicSequence conversion without TileMapping.
- **Tiling:** Added `array_to_logic_seq()`/`logic_seq_to_array()` high-level shortcuts for Tensor ↔ LogicSequence conversion via TileMapping. `tensor_to_cocotb_seq()`/`cocotb_seq_to_tensor()` preserved as aliases.
- **Tiling:** Added `to_logic_sequence()` canonical method on `TileMapping`. `to_int_sequence()` preserved as alias.
- **Docs:** Added `doc/pic/04_sequence.typ` diagram showing Matrix / LogicSequence / VectorSequence conversion relationships.
- **Docs:** Added "Basic Datatypes" section to README with terminology table, conversion diagrams, and TileMapping pipeline explanation.
- **Tests:** Added `TestVectorCanonicalNames` test class with 11 tests verifying canonical method equivalence with aliases.

### Changed
- **Core:** `int_sequence.py` now re-exports from `logic_sequence.py` (canonical implementation moved).
- **Core:** `__init__.py` imports from `logic_sequence` instead of `int_sequence`.
- **Tiling:** Internal imports updated from `IntSequence` to `LogicSequence` across `tile_mapping.py`, `address_mapping.py`.
- **Tools:** Internal imports updated from `IntSequence` to `LogicSequence` across `buffer.py`, `driver.py`, `monitor.py`.
- **Docs:** Moved all diagram files from `doc/` to `doc/pic/` directory.
- **Docs:** Updated all 4 diagrams with canonical terminology (Logic/Array/Matrix/LogicSequence).
- **Docs:** Updated Sphinx API docs (`core.rst`, `tiling.rst`, `tools.rst`) to reference new module structure.
- **Docs:** Updated `docs/conf.py` version to 2.4.0.

## [2.3.0] - 2026-02-08

### Added
- **Core:** Added `IntSequence` class (`class IntSequence(list)`) as typed integer sequence for all hardware verification interfaces.
- **Core:** Added `BitStruct`/`BitField` to `torchbit.core` module (moved from `torchbit.tools`), re-exported from tools for backward compatibility.
- **Tiling:** Added `ContiguousAddressMapping` subclass of `AddressMapping` with auto-computed row-major contiguous strides.
- **Tiling:** Added `hw_temp_einops` parameter to `AddressMapping` for explicit dimension ordering (earlier = higher address bits). Dicts can now be passed in any key order.
- **Tests:** Added tests for `ContiguousAddressMapping`, `AddressMapping` key validation, and unordered dict support.
- **Tests:** Added backward compatibility tests for `Matrix` alias to `VectorSequence`.

### Changed
- **Core:** Renamed `Matrix` class to `VectorSequence` (file `matrix.py` → `vector_sequence.py`). `Matrix` alias kept for backward compatibility.
- **Tiling:** Separated concerns: `TileMapping` handles only tensor↔IntSequence value conversion (no addresses), `AddressMapping` handles only address generation.
- **Tiling:** Split `mapping.py` into `tile_mapping.py` and `address_mapping.py`.
- **Tiling:** `AddressMapping.__init__` now requires `hw_temp_einops` parameter and asserts key consistency with `hw_temp_dim`/`hw_temp_stride`.
- **Tiling:** `TileMapping` no longer has `base_addr` or `strides` fields. `to_hw()` returns only `IntSequence` values; `to_sw()` accepts only `IntSequence` values.
- **Tools:** `Buffer.backdoor_load_tensor()` and `backdoor_dump_tensor()` now require both `TileMapping` and `AddressMapping` as mandatory parameters.
- **Tools:** `Buffer.backdoor_read()` now returns `IntSequence` instead of `list[int]`.
- **Tools:** `Driver` queue and `load()` method now use `IntSequence` type.
- **Tools:** `PoolMonitor` and `FIFOMonitor` `data` and `dump()` now use `IntSequence` type.
- **Documentation:** Renamed "Hardware Matrix" terminology to "Hardware Vector Sequence" across all docs (English and Chinese). "Hardware Matrix" kept as alias.
- **Documentation:** Added emphasis that the temporal dimension is ordered: lower index = earlier time step.
- **Commands:** Added CHANGELOG version order validation step to `commit` command.

### Removed
- **Core:** Removed `torchbit/core/matrix.py` (replaced by `torchbit/core/vector_sequence.py`).
- **Tiling:** Removed `torchbit/tiling/mapping.py` (split into `address_mapping.py` and `tile_mapping.py`).
- **Tools:** Removed `torchbit/tools/bit_struct.py` (moved to `torchbit/core/bit_struct.py`).
- **Tools:** Removed `import einops` from `buffer.py` (no longer needed, delegated to TileMapping).

## [2.2.0] - 2026-02-07

### Added
- **Documentation:** Added English translation of `golden_model_recommend.md` with AI accelerator Golden Model best practices.
- **Commands:** Enhanced `commit` command to create git tags for version releases (e.g., `v2.1.0`).

### Changed
- **Tools:** Renamed `Sender` to `Driver` (UVM-aligned naming for stimulus driver).
- **Tools:** Renamed `PoolCollector` to `PoolMonitor` (UVM-aligned naming for passive output monitor).
- **Tools:** Renamed `FIFOCollector` to `FIFOMonitor` (UVM-aligned naming for FIFO handshake monitor).
- **Tools:** Split `sender_collector.py` into separate `driver.py` and `monitor.py` modules.
- **Documentation:** Polished `golden_model_recomd.md` with better structure, fixed markdown links, and improved language flow.
- **Tests:** Added `test_buffer_backdoor_performance.py` with performance benchmarks for Buffer backdoor operations.
- **Tests:** Added profiling tests using cProfile to identify bottlenecks in backdoor operations.
- **Tests:** Added `tests/reports/` directory with markdown performance reports including profiling breakdown.
- **Commands:** Added `add_test` command for generating tests with optional performance profiling.
- **Examples:** Reorganized examples into 5 structured directories:
  - `01_basic_runner` - Basic Runner setup with simple counter DUT
  - `02_data_convert` - Vector/Matrix conversion demonstrations (no DUT)
  - `03_bit_struct` - BitStruct usage examples
  - `04_driver_monitor` - Driver/Monitor pattern with pipeline DUT
  - `05_buffer` - TwoPortBuffer usage with memory mover DUT
- **Examples:** Added `05_buffer` example demonstrating TwoPortBuffer as shared memory with:
  - Memory mover DUT that copies data between memory regions
  - TwoPortBuffer for backdoor memory access (init/read operations)
  - Full chip_verification skill format (wrapper, golden model, test cases, main.py)
- **Examples:** Added main `examples/README.md` with overview of all examples
- **Tests:** Added comprehensive test suite in `tests/` directory:
  - `test_vector.py` - Vector class tests (38 tests)
  - `test_matrix.py` - Matrix class tests (57 tests)
  - `test_ports.py` - InputPort/OutputPort tests (24 tests)
  - `test_driver_monitor.py` - Driver/Monitor tests (27 tests)
  - `test_runner.py` - Runner and configuration tests (18 tests)
- **Debug:** Added new `waveform_tool.py` module with waveform analysis functions: `posedge_dump_to_csv()`, `list_signals()`, `check_signals()`, and `count_edges()`.
- **Debug:** Added `convert_fsdb_to_fst()` function to `file_converter.py` for FSDB to FST format conversion.
- **Debug:** Added support for flexible output in waveform tools - functions can output to stdout or file.

### Removed
- **Tools:** Removed `sender_collector.py` (split into `driver.py` and `monitor.py`).
- **Examples:** Removed old flat example structure:
  - `test_bit_ops.py`, `test_bit_struct.py`, `test_bit_struct_comparator.py`
  - `test_matrix.py`, `test_matrix_with_verilog.py`
  - `test_temporal_event.py`, `test_vector.py`, `test_wavetool.py`
  - `test_sender_collector/` directory (moved to `04_driver_monitor/`)
- **Debug:** Removed `main.py` from wave_converter module. Functions moved to `file_converter.py` and `waveform_tool.py`.

### Changed
- **Debug:** Reorganized wave_converter module structure for better separation of concerns:
  - `file_converter.py`: Format conversion and high-level workflow
  - `wal_parser.py`: Basic WAL parsing
  - `waveform_tool.py`: Waveform analysis tools
- **Debug:** `posedge_dump_to_csv()` now supports optional `signal_list` parameter (defaults to all signals) and stdout output.

## [2.1.0] - 2026-01-30

### Added
- **Tiling:** Added `TileMapping` class for tensor-to-memory mapping with spatial/temporal dimension support.
- **Tiling:** Added `AddressMapping` class for multi-dimensional to flat memory address translation.
- **Tiling:** Added `pad_utils` module with einops-style padding functions: `pad()`, `depad()`, `depad_like()`, `get_padlen()`.
- **Tools:** Added `backdoor_read()` and `backdoor_write()` to `Buffer` class for parallel bulk operations.
- **Tools:** Added UVM-style backdoor operations to `Buffer` class: `backdoor_load_tensor()`, `backdoor_dump_tensor()`, `backdoor_load_matrix()`, `backdoor_dump_matrix()`.
- **Documentation:** Added bilingual documentation structure with `doc/en/` (English) and `doc/zh-CN/` (Chinese) directories.
- **Documentation:** Added `doc/en/tilling_schedule.md` with tiling operation specifications.
- **Examples:** Added `06_tile_mapping` example demonstrating TileMapping with spatial/temporal dimensions.
- **Commands:** Added unified `commit` command combining changelog update and git operations.

### Changed
- **Tiling:** Refactored `TileMapping` API - removed `->` from einops patterns, now uses `sw_einops` and `hw_einops` as separate properties.
- **Tiling:** `TileMapping.hw_einops` must be 2D format `(temporal) (spatial)` for hardware vector sequence layout.
- **Tiling:** Renamed `shape_process.py` to `pad_utils.py` with einops-style interface.
- **Tools:** Refactored `backdoor_load_tensor()` and `backdoor_dump_tensor()` to use `mapping.to_hw()`/`to_sw()` and `backdoor_read()`/`backdoor_write()`.
- **Tools:** `Buffer.init_from_tensor()` now uses `TileMapping.sw_to_hw_formula` for tensor-to-memory conversion.
- **Documentation:** Separated Chinese and English documentation into distinct directories.
- **Documentation:** Updated README.md to English-only, added README.zh-CN.md for Chinese.

### Removed
- **Tools:** Removed `backdoor_read_write()` from `Buffer` class (functionality split into separate read/write).
- **Commands:** Removed `update_changelog` and `update_git` commands (merged into `commit` command).
- **Tools:** Removed deprecated `torchbit/tools/mapping.py` (functionality moved to `torchbit/tiling/mapping.py`).
- **Tools:** Removed deprecated `torchbit/tools/shape_process.py` (replaced by `torchbit/tiling/pad_utils.py`).
- **Documentation:** Removed `doc/tensor.md` and `doc/value.md` (content reorganized).

## [2.0.0] - 2025-12-22
### Changed
- **Dependencies:** Bumped `cocotb` version requirement to `>=2.0.0`.
- **Core:** Replaced `cocotb.binary.BinaryValue` with `cocotb.types.LogicArray` in `Vector` and `Matrix` classes to support Cocotb 2.0.
- **Runner:** Switched to `cocotb_tools.runner` for `get_runner`.
- **Core:** Enhanced `Vector.from_cocotb()` to handle both `cocotb.types.LogicArray` and `cocotb.types.Logic` by converting them to `int` uniformly.
- **Tools:** Renamed `Collector` to `PoolCollector` in `sender_collector.py` to better reflect its always-ready behavior.
- **Tools:** Enhanced `Sender.run()` to support optional `stop_event` parameter for controlled stopping.
- **Tools:** Updated `InputPort.get()` to use `int(self.wrapper.value)` instead of `value.integer` for better compatibility with Cocotb 2.0.
- **Runner:** Refactored `BuildConfig` to use modular simulator-specific build args classes (`VerilatorBuildArgs` and `VCSBuildArgs`).
- **Runner:** Renamed `DEFAULT_BUILD_CONFIG` to `DEFAULT_VERILATOR_BUILD_CONFIG` for clarity.
- **Core:** Improved `BitStruct` to support deep copying and pickling by implementing `__getstate__` and `__setstate__`.
- **Tools:** Modified `FIFOCollector` to use `cocotb.handle.Immediate` for setting `ready` port, ensuring immediate value updates in simulation.

### Added
- **Core:** Added version check for `cocotb >= 2.0.0` at package initialization.
- **Tools:** Added new `FIFOCollector` class in `sender_collector.py` for FIFO interfaces with empty/ready signals.
- **Runner:** Added VCS backend support with `VCSBuildArgs` class and automatic FST wave dump generation.
- **Runner:** Added `DEFAULT_VCS_BUILD_CONFIG` for VCS simulator configuration.
- **Runner:** Added `generate_vcs_dump_wave()` helper function to automatically create wave dump module for VCS.
- **Debug:** Added `compare_bit_struct` function in `torchbit.debug` to compare two `BitStruct` objects and report field-level mismatches.
- **Utils:** Added `signed` and `unsigned` conversion functions to `bit_ops.py`.
- **Examples:** Added `test_bit_struct_comparator.py` to test the `compare_bit_struct` function.

## [1.2.0] - 2025-12-11
### Added
- **Tools:** Added `clear()` method to `Buffer`.
- **Debug:** Exposed `Signal`, `SignalGroup`, and `generate_gtkwave_tcl` in `torchbit.debug` for better wave dump control.

### Changed
- **Struct API:** Major refactor of `BitStruct`. It now acts as a factory returning a class, allowing cleaner instantiation (e.g., `MyStruct = BitStruct(...)` then `s = MyStruct()`).
- **Struct API:** Added support for setting field values via attributes (e.g., `struct.field = 0x5`).
- **Tools:** Simplified `Collector` class in `sender_collector`. Removed `ready` backpressure simulation (now always ready) and `valid_is_empty` configuration (now strictly active-high valid).

## [1.1.0] - 2025-12-04
### Added
- **Tools:** Added `sender_collector` module with `Sender` and `Collector` classes for verifying FIFO/Pipeline interfaces using cocotb.
- **Tools:** Added `inspect()` method to `BitStruct` for visualizing layout and values.
- **Debug:** Added `temporal_event` module for visualizing simulation events as timing diagrams.
- **Examples:** Added `test_sender_collector` including a Verilog Pipe implementation and cocotb testbench.
- **Examples:** Added `test_temporal_event.py` and updated `test_bit_ops.py`.
- **Runner:** Integrated `runner` module into the main `torchbit` package.

### Changed
- **Tools:** Refactored `Buffer` tool to use concurrent processes for read/write logic, better mimicking hardware behavior.
- **Struct API:** Renamed `Struct` to `BitStruct` to avoid name collision. `BitStruct` constructor now requires a mandatory `lsb_first` (bool) argument to specify field bit order.

## [1.0.0] - 2025-12-02
### Added
- **Tools:** Added `struct.py` with `BitField` and `Struct` classes. `Struct` now supports field access via attributes (e.g., `my_struct.field_name.set_value()`).
- **Examples:** Added `test_struct.py` for testing `BitField` and `Struct`.
- **Utils:** Added `bit_ops.py` (containing `replicate_bits`, `get_bit`, `get_bit_slice`) and `tensor_ops.py`.

### Changed
- **Package Structure:** Restructured the entire package into subpackages:
    - `torchbit.core`: Contains `dtype`, `vector` (formerly `hensor`), `matrix` (formerly `hlist`).
    - `torchbit.debug`: Contains `judge`.
    - `torchbit.tools`: Contains `buffer`, `port`, `shape_process`, `struct`, `transform`.
    - `torchbit.utils`: Contains `bit_ops`, `tensor_ops`.
- **Renaming:**
    - Renamed `Hensor` to `Vector` (`torchbit.core.vector`).
    - Renamed `Hlist` to `Matrix` (`torchbit.core.matrix`).
    - Renamed parameters in `get_bit_slice` to `high_close` and `low_close` for clarity.
- **Moved:**
    - `InputPort` and `OutputPort` moved from `buffer.py` to `tools/port.py`.
    - `replicate_bits` moved from `buffer.py` to `utils/bit_ops.py`.
    - `utils.py` content split into `utils/bit_ops.py` and `utils/tensor_ops.py`.

### Removed
- Removed flat `torchbit.*` structure for most modules. Now requires explicit subpackage imports (e.g., `from torchbit.core import Vector`).

## [0.3.1] - 2025-10-18
### Added
- Shape Transform Support.
- Add `Buffer` test component.

## [0.2.0] - 2025-06-10
### Changed
- **Verilog Interface:** Changed default endianness from big-endian to little-endian.

## [0.1.0] - 2025-03-24
### Added
- **Verilog Interface:** Added handling for Byte Order conversion between Numpy/Torch and Verilog.
### Fixed
- **X/Z Value Support:** Fixed issue where IO values treated as `int` caused issues with `BinaryValue` containing X/Z states.
    - Previously, signals with X/Z were implicitly treated as 0 when accessed via `integer`.
    - `Vector.from_cocotb()` (formerly `Hensor.from_cocotb()`) now explicitly handles `integer` conversion and issues a warning (non-blocking) if X/Z values are detected.
    - Note: Verilator backend defaults X to 0/1. Z values are internal. X propagation is more visible with VCS backend.
    - References:
        - [Cocotb Writing Testbenches](https://docs.cocotb.org/en/stable/writing_testbenches.html)
        - [Cocotb Binary Module](https://docs.cocotb.org/en/stable/_modules/cocotb/binary.html)