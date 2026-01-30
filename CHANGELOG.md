# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Tiling:** `TileMapping.hw_einops` must be 2D format `(temporal) (spatial)` for hardware matrix layout.
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

## [Unreleased]

### Added
- **Examples:** Reorganized examples into 5 structured directories:
  - `01_basic_runner` - Basic Runner setup with simple counter DUT
  - `02_data_convert` - Vector/Matrix conversion demonstrations (no DUT)
  - `03_bit_struct` - BitStruct usage examples
  - `04_sender_collector` - Sender/Collector pattern with pipeline DUT
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
  - `test_sender_collector.py` - Sender/Collector tests (27 tests)
  - `test_runner.py` - Runner and configuration tests (18 tests)
- **Debug:** Added new `waveform_tool.py` module with waveform analysis functions: `posedge_dump_to_csv()`, `list_signals()`, `check_signals()`, and `count_edges()`.
- **Debug:** Added `convert_fsdb_to_fst()` function to `file_converter.py` for FSDB to FST format conversion.
- **Debug:** Added support for flexible output in waveform tools - functions can output to stdout or file.

### Removed
- **Examples:** Removed old flat example structure:
  - `test_bit_ops.py`, `test_bit_struct.py`, `test_bit_struct_comparator.py`
  - `test_matrix.py`, `test_matrix_with_verilog.py`
  - `test_temporal_event.py`, `test_vector.py`, `test_wavetool.py`
  - `test_sender_collector/` directory (moved to `04_sender_collector/`)
- **Debug:** Removed `main.py` from wave_converter module. Functions moved to `file_converter.py` and `waveform_tool.py`.

### Changed
- **Debug:** Reorganized wave_converter module structure for better separation of concerns:
  - `file_converter.py`: Format conversion and high-level workflow
  - `wal_parser.py`: Basic WAL parsing
  - `waveform_tool.py`: Waveform analysis tools
- **Debug:** `posedge_dump_to_csv()` now supports optional `signal_list` parameter (defaults to all signals) and stdout output.

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