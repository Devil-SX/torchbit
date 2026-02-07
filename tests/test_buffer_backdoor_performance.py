"""Performance tests for Buffer backdoor operations.

Measures performance of parallel read/write operations and tensor operations
across different data sizes. Includes profiling to identify bottlenecks.
"""

import pytest
import torch
import time
import tracemalloc
import cProfile
import pstats
import io
from torchbit.tools.buffer import Buffer
from torchbit.tiling import TileMapping, AddressMapping


class TestBufferBackdoorPerformance:
    """Performance tests for Buffer backdoor operations."""

    @pytest.mark.parametrize("size", [10, 100, 1000, 10000])
    def test_backdoor_read_performance(self, size):
        """Test backdoor_read performance across different sizes."""
        buf = Buffer(width=64, depth=16384)

        # Pre-populate buffer
        for i in range(size):
            buf.write(i, i * 0x1000 + 0xDEAD)

        addr_list = list(range(size))

        # Measure time
        start = time.perf_counter()
        values = buf.backdoor_read(addr_list)
        elapsed = time.perf_counter() - start

        # Measure memory
        tracemalloc.start()
        _ = buf.backdoor_read(addr_list)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Verify correctness
        assert len(values) == size
        assert values[0] == 0xDEAD
        if size > 1:
            assert values[size - 1] == (size - 1) * 0x1000 + 0xDEAD

        # Print performance metrics
        print(f"\nbackdoor_read({size}):")
        print(f"  Time: {elapsed * 1000:.4f} ms")
        print(f"  Throughput: {size / elapsed:.0f} ops/sec")
        print(f"  Memory: {peak / 1024:.2f} KB")

    @pytest.mark.parametrize("size", [10, 100, 1000, 10000])
    def test_backdoor_write_performance(self, size):
        """Test backdoor_write performance across different sizes."""
        buf = Buffer(width=64, depth=16384)

        addr_list = list(range(size))
        data_list = [i * 0x1000 + 0xBEEF for i in range(size)]

        # Measure time
        start = time.perf_counter()
        buf.backdoor_write(addr_list, data_list)
        elapsed = time.perf_counter() - start

        # Measure memory
        buf2 = Buffer(width=64, depth=16384)
        tracemalloc.start()
        buf2.backdoor_write(addr_list, data_list)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Verify correctness
        for i in range(min(10, size)):  # Sample verification
            assert buf.read(i) == data_list[i]

        print(f"\nbackdoor_write({size}):")
        print(f"  Time: {elapsed * 1000:.4f} ms")
        print(f"  Throughput: {size / elapsed:.0f} ops/sec")
        print(f"  Memory: {peak / 1024:.2f} KB")

    @pytest.mark.parametrize("size", [16, 64, 256, 1024])
    def test_backdoor_load_matrix_performance(self, size):
        """Test backdoor_load_matrix performance across different sizes."""
        buf = Buffer(width=128, depth=16384)

        # Create test matrix (128 bits = 4 float32 per row)
        torch.manual_seed(42)
        matrix = torch.randn(size, 4, dtype=torch.float32)

        # Measure time
        start = time.perf_counter()
        buf.backdoor_load_matrix(0, size, matrix)
        elapsed = time.perf_counter() - start

        # Measure memory
        buf2 = Buffer(width=128, depth=16384)
        tracemalloc.start()
        buf2.backdoor_load_matrix(0, size, matrix)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nbackdoor_load_matrix({size}x4):")
        print(f"  Time: {elapsed * 1000:.4f} ms")
        print(f"  Throughput: {size * 4 / elapsed:.0f} elements/sec")
        print(f"  Memory: {peak / 1024:.2f} KB")

    @pytest.mark.parametrize("size", [16, 64, 256, 1024])
    def test_backdoor_dump_matrix_performance(self, size):
        """Test backdoor_dump_matrix performance across different sizes."""
        buf = Buffer(width=128, depth=16384)

        # Pre-populate buffer
        torch.manual_seed(42)
        matrix = torch.randn(size, 4, dtype=torch.float32)
        buf.backdoor_load_matrix(0, size, matrix)

        # Measure time
        start = time.perf_counter()
        result = buf.backdoor_dump_matrix(0, size, torch.float32)
        elapsed = time.perf_counter() - start

        # Measure memory
        tracemalloc.start()
        _ = buf.backdoor_dump_matrix(0, size, torch.float32)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Verify correctness
        assert torch.allclose(result, matrix)

        print(f"\nbackdoor_dump_matrix({size}x4):")
        print(f"  Time: {elapsed * 1000:.4f} ms")
        print(f"  Throughput: {size * 4 / elapsed:.0f} elements/sec")
        print(f"  Memory: {peak / 1024:.2f} KB")

    @pytest.mark.parametrize("channels,height,width", [
        (3, 8, 8),     # Small: 192 elements
        (3, 16, 16),   # Medium: 768 elements
        (3, 32, 32),   # Large: 3072 elements
        (3, 64, 64),   # XLarge: 12288 elements
    ])
    def test_backdoor_load_tensor_performance(self, channels, height, width):
        """Test backdoor_load_tensor performance across different tensor sizes."""
        buf = Buffer(width=128, depth=16384)

        torch.manual_seed(42)
        tensor = torch.randn(channels, height, width, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",
            hw_temp_dim={"c": channels},
            hw_spat_dim={"h": height, "w": width},
        )

        addr_mapping = AddressMapping(
            base=0,
            hw_temp_einops="c",
            hw_temp_dim={"c": channels},
            hw_temp_stride={"c": 1},
        )

        # Measure time
        start = time.perf_counter()
        buf.backdoor_load_tensor(tensor, mapping, addr_mapping)
        elapsed = time.perf_counter() - start

        # Measure memory
        buf2 = Buffer(width=128, depth=16384)
        tracemalloc.start()
        buf2.backdoor_load_tensor(tensor, mapping, addr_mapping)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_elements = channels * height * width
        print(f"\nbackdoor_load_tensor({channels}x{height}x{width}):")
        print(f"  Time: {elapsed * 1000:.4f} ms")
        print(f"  Throughput: {total_elements / elapsed:.0f} elements/sec")
        print(f"  Memory: {peak / 1024:.2f} KB")

    @pytest.mark.parametrize("channels,height,width", [
        (3, 8, 8),     # Small: 192 elements
        (3, 16, 16),   # Medium: 768 elements
        (3, 32, 32),   # Large: 3072 elements
        (3, 64, 64),   # XLarge: 12288 elements
    ])
    def test_backdoor_dump_tensor_performance(self, channels, height, width):
        """Test backdoor_dump_tensor performance across different tensor sizes."""
        buf = Buffer(width=128, depth=16384)

        torch.manual_seed(42)
        original = torch.randn(channels, height, width, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",
            hw_temp_dim={"c": channels},
            hw_spat_dim={"h": height, "w": width},
        )

        addr_mapping = AddressMapping(
            base=0,
            hw_temp_einops="c",
            hw_temp_dim={"c": channels},
            hw_temp_stride={"c": 1},
        )

        # Load first
        buf.backdoor_load_tensor(original, mapping, addr_mapping)

        # Measure dump time
        start = time.perf_counter()
        result = buf.backdoor_dump_tensor(mapping, addr_mapping)
        elapsed = time.perf_counter() - start

        # Measure memory
        tracemalloc.start()
        _ = buf.backdoor_dump_tensor(mapping, addr_mapping)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Verify correctness
        assert torch.allclose(result, original)

        total_elements = channels * height * width
        print(f"\nbackdoor_dump_tensor({channels}x{height}x{width}):")
        print(f"  Time: {elapsed * 1000:.4f} ms")
        print(f"  Throughput: {total_elements / elapsed:.0f} elements/sec")
        print(f"  Memory: {peak / 1024:.2f} KB")


class TestBufferBackdoorRoundtripPerformance:
    """Performance tests for complete roundtrip operations."""

    @pytest.mark.parametrize("channels,height,width", [
        (3, 8, 8),     # Small
        (3, 16, 16),   # Medium
        (3, 32, 32),   # Large
    ])
    def test_tensor_roundtrip_performance(self, channels, height, width):
        """Test complete load -> dump roundtrip performance."""
        buf = Buffer(width=128, depth=16384)

        torch.manual_seed(42)
        original = torch.randn(channels, height, width, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",
            hw_temp_dim={"c": channels},
            hw_spat_dim={"h": height, "w": width},
        )

        addr_mapping = AddressMapping(
            base=0,
            hw_temp_einops="c",
            hw_temp_dim={"c": channels},
            hw_temp_stride={"c": 1},
        )

        # Measure roundtrip time
        start = time.perf_counter()
        buf.backdoor_load_tensor(original, mapping, addr_mapping)
        recovered = buf.backdoor_dump_tensor(mapping, addr_mapping)
        elapsed = time.perf_counter() - start

        # Verify correctness
        assert torch.allclose(recovered, original)

        total_elements = channels * height * width
        print(f"\nroundtrip({channels}x{height}x{width}):")
        print(f"  Time: {elapsed * 1000:.4f} ms")
        print(f"  Throughput: {total_elements / elapsed:.0f} elements/sec")

    @pytest.mark.parametrize("size", [16, 64, 256, 1024])
    def test_matrix_roundtrip_performance(self, size):
        """Test complete matrix load -> dump roundtrip performance."""
        buf = Buffer(width=128, depth=16384)

        torch.manual_seed(42)
        original = torch.randn(size, 4, dtype=torch.float32)

        # Measure roundtrip time
        start = time.perf_counter()
        buf.backdoor_load_matrix(0, size, original)
        recovered = buf.backdoor_dump_matrix(0, size, torch.float32)
        elapsed = time.perf_counter() - start

        # Verify correctness
        assert torch.allclose(recovered, original)

        total_elements = size * 4
        print(f"\nmatrix_roundtrip({size}x4):")
        print(f"  Time: {elapsed * 1000:.4f} ms")
        print(f"  Throughput: {total_elements / elapsed:.0f} elements/sec")


class TestBufferBackdoorProfiling:
    """Profiling tests for Buffer backdoor operations to identify bottlenecks."""

    def _profile_operation(self, operation, *args, **kwargs):
        """Run an operation under cProfile and return parsed results."""
        profiler = cProfile.Profile()
        profiler.enable()
        result = operation(*args, **kwargs)
        profiler.disable()

        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(30)

        output = s.getvalue()

        lines = output.split('\n')
        steps = []
        total_time = 0

        for line in lines[5:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                ncalls = parts[0]
                cumtime = float(parts[3])
                percall = float(parts[4])
                filename = parts[5]

                if any(x in filename for x in ['pytest', 'site-packages', 'lib/python', '__pycache__']):
                    continue

                total_time += cumtime

                if '{method}' in filename:
                    func_name = filename.split('{method} ')[1].split('}')[0]
                elif '(' in filename:
                    func_name = filename.split('(')[0].split('/')[-1]
                else:
                    func_name = filename.split('/')[-1]

                steps.append({
                    'name': func_name,
                    'calls': ncalls,
                    'total_ms': cumtime * 1000,
                    'per_call_ms': percall * 1000,
                })
            except (ValueError, IndexError):
                continue

        for step in steps:
            step['percent'] = (step['total_ms'] / (total_time * 1000)) * 100 if total_time > 0 else 0

        return {'steps': steps, 'total_ms': total_time * 1000, 'result': result}

    def _print_profile_table(self, operation_name: str, profile_data: dict):
        """Print profiling results as a formatted table."""
        print(f"\n{'='*70}")
        print(f"Profiling: {operation_name}")
        print(f"{'='*70}")

        steps = profile_data['steps']
        total = profile_data['total_ms']

        if not steps:
            print("No profiling data available (operation too fast or filtered out)")
            return

        print(f"{'Step':<40} {'Calls':<10} {'Total(ms)':<12} {'PerCall(ms)':<12} {'%':<8}")
        print('-' * 90)

        for step in steps:
            status = 'Bottleneck' if step['percent'] > 30 else ''
            print(f"{step['name']:<40} {step['calls']:<10} "
                  f"{step['total_ms']:<12.4f} {step['per_call_ms']:<12.6f} "
                  f"{step['percent']:<8.1f}% {status}")

        print('-' * 90)
        print(f"{'Total':<53} {total:<12.4f} 100.0%")
        print(f"{'='*70}\n")

    def test_profile_backdoor_read_xlarge(self):
        """Profile backdoor_read with xlarge size (10000 elements)."""
        buf = Buffer(width=64, depth=16384)

        size = 10000
        for i in range(size):
            buf.write(i, i * 0x1000 + 0xDEAD)

        addr_list = list(range(size))

        def read_operation():
            return buf.backdoor_read(addr_list)

        profile_data = self._profile_operation(read_operation)
        self._print_profile_table("backdoor_read(10000)", profile_data)

        values = profile_data['result']
        assert len(values) == size

    def test_profile_backdoor_write_xlarge(self):
        """Profile backdoor_write with xlarge size (10000 elements)."""
        buf = Buffer(width=64, depth=16384)

        size = 10000
        addr_list = list(range(size))
        data_list = [i * 0x1000 + 0xBEEF for i in range(size)]

        def write_operation():
            buf.backdoor_write(addr_list, data_list)

        profile_data = self._profile_operation(write_operation)
        self._print_profile_table("backdoor_write(10000)", profile_data)

    def test_profile_backdoor_load_matrix_xlarge(self):
        """Profile backdoor_load_matrix with xlarge size (1024x4)."""
        buf = Buffer(width=128, depth=16384)

        size = 1024
        torch.manual_seed(42)
        matrix = torch.randn(size, 4, dtype=torch.float32)

        def load_matrix_operation():
            buf.backdoor_load_matrix(0, size, matrix)

        profile_data = self._profile_operation(load_matrix_operation)
        self._print_profile_table("backdoor_load_matrix(1024x4)", profile_data)

    def test_profile_backdoor_dump_matrix_xlarge(self):
        """Profile backdoor_dump_matrix with xlarge size (1024x4)."""
        buf = Buffer(width=128, depth=16384)

        size = 1024
        torch.manual_seed(42)
        matrix = torch.randn(size, 4, dtype=torch.float32)
        buf.backdoor_load_matrix(0, size, matrix)

        def dump_matrix_operation():
            return buf.backdoor_dump_matrix(0, size, torch.float32)

        profile_data = self._profile_operation(dump_matrix_operation)
        self._print_profile_table("backdoor_dump_matrix(1024x4)", profile_data)

    def test_profile_backdoor_load_tensor_xlarge(self):
        """Profile backdoor_load_tensor with xlarge tensor (3x64x64)."""
        buf = Buffer(width=128, depth=16384)

        torch.manual_seed(42)
        tensor = torch.randn(3, 64, 64, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 64, "w": 64},
        )

        addr_mapping = AddressMapping(
            base=0,
            hw_temp_einops="c",
            hw_temp_dim={"c": 3},
            hw_temp_stride={"c": 1},
        )

        def load_tensor_operation():
            buf.backdoor_load_tensor(tensor, mapping, addr_mapping)

        profile_data = self._profile_operation(load_tensor_operation)
        self._print_profile_table("backdoor_load_tensor(3x64x64)", profile_data)

    def test_profile_backdoor_dump_tensor_xlarge(self):
        """Profile backdoor_dump_tensor with xlarge tensor (3x64x64)."""
        buf = Buffer(width=128, depth=16384)

        torch.manual_seed(42)
        original = torch.randn(3, 64, 64, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 64, "w": 64},
        )

        addr_mapping = AddressMapping(
            base=0,
            hw_temp_einops="c",
            hw_temp_dim={"c": 3},
            hw_temp_stride={"c": 1},
        )

        buf.backdoor_load_tensor(original, mapping, addr_mapping)

        def dump_tensor_operation():
            return buf.backdoor_dump_tensor(mapping, addr_mapping)

        profile_data = self._profile_operation(dump_tensor_operation)
        self._print_profile_table("backdoor_dump_tensor(3x64x64)", profile_data)
