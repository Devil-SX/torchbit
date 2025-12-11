# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Unreleased
### Changed
- **Dependencies:** Bumped `cocotb` version requirement to `>=2.0.0`.
- **Core:** Replaced `cocotb.binary.BinaryValue` with `cocotb.types.LogicArray` in `Vector` and `Matrix` classes to support Cocotb 2.0.
- **Runner:** Switched to `cocotb_tools.runner` for `get_runner`.

### Added
- **Core:** Added version check for `cocotb >= 2.0.0` at package initialization.

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