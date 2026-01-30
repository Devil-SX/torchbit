# Performance Report: Buffer Backdoor Operations

**Date:** 2026-01-30

**Test Environment:**
- Python 3.12.12
- pytest 9.0.2
- torch 2.x

---

## Summary

| Operation | Min Size | Max Size | Min Throughput | Max Throughput | Bottleneck |
|-----------|----------|----------|----------------|----------------|------------|
| `backdoor_read` | 10 | 10000 | 2.5M ops/s | 24.5M ops/s | - (fast) |
| `backdoor_write` | 10 | 10000 | 1.6M ops/s | 21.9M ops/s | - (fast) |
| `backdoor_load_matrix` | 16×4 | 1024×4 | 8.7K elem/s | 971K elem/s | Vector conversion (36%) |
| `backdoor_dump_matrix` | 16×4 | 1024×4 | 12.4K elem/s | 1.4M elem/s | Memory copy (30%) |
| `backdoor_load_tensor` | 3×8×8 | 3×64×64 | 121 elem/s | 1.9M elem/s | einops rearrange (55%) |
| `backdoor_dump_tensor` | 3×8×8 | 3×64×64 | 1.0M elem/s | 3.6M elem/s | Vector conversion (24%) |

---

## 1. backdoor_read Performance

Parallel read operation from multiple addresses.

| Size | Time (ms) | Throughput (ops/s) | Memory (KB) |
|------|-----------|-------------------|-------------|
| 10 | 0.0040 | 2,514,459 | 0.17 |
| 100 | 0.0067 | 14,838,991 | 0.89 |
| 1000 | 0.0408 | 24,515,812 | 8.64 |
| 10000 | 0.4101 | 24,381,443 | 83.17 |

**Analysis:**
- Linear scaling with size
- Throughput increases with batch size (better amortization)
- Memory usage scales linearly: ~8 bytes per address

---

## 2. backdoor_write Performance

Parallel write operation to multiple addresses.

| Size | Time (ms) | Throughput (ops/s) | Memory (KB) |
|------|-----------|-------------------|-------------|
| 10 | 0.0063 | 1,595,405 | 0.16 |
| 100 | 0.0062 | 16,123,829 | 0.16 |
| 1000 | 0.0503 | 19,897,330 | 0.16 |
| 10000 | 0.4561 | 21,926,891 | 0.16 |

**Analysis:**
- Consistent ~O(n) scaling
- Constant memory overhead (write doesn't allocate)
- Better throughput at larger batch sizes

---

## 3. backdoor_load_matrix Performance

Load 2D matrix data into buffer.

| Size (rows×cols) | Time (ms) | Throughput (elem/s) | Memory (KB) |
|------------------|-----------|---------------------|-------------|
| 16×4 | 7.37 | 8,688 | 1.57 |
| 64×4 | 0.30 | 856,416 | 3.63 |
| 256×4 | 1.13 | 904,091 | 11.88 |
| 1024×4 | 4.22 | 971,645 | 44.91 |

**Analysis:**
- First iteration (16×4) shows cold start overhead
- Consistent ~900K-1M elements/sec for larger sizes
- Memory: ~0.04 KB per element loaded

---

## 4. backdoor_dump_matrix Performance

Dump buffer contents to 2D matrix.

| Size (rows×cols) | Time (ms) | Throughput (elem/s) | Memory (KB) |
|------------------|-----------|---------------------|-------------|
| 16×4 | 5.18 | 12,362 | 3.68 |
| 64×4 | 0.25 | 1,012,474 | 14.18 |
| 256×4 | 1.02 | 1,006,630 | 56.37 |
| 1024×4 | 2.87 | 1,426,284 | 225.38 |

**Analysis:**
- Throughput increases with size (better at scale)
- Memory: ~0.22 KB per element dumped
- More memory intensive than load (tensor allocation)

---

## 5. backdoor_load_tensor Performance

Load tensor using TileMapping.

| Size (C×H×W) | Elements | Time (ms) | Throughput (elem/s) | Memory (KB) |
|--------------|----------|-----------|---------------------|-------------|
| 3×8×8 | 192 | 1586.95* | 121 | 5.07 |
| 3×16×16 | 768 | 0.40 | 1,938,841 | 16.62 |
| 3×32×32 | 3072 | 2.04 | 1,508,975 | 62.69 |
| 3×64×64 | 12288 | 23.04 | 533,406 | 246.68 |

*First iteration includes cold start overhead.

**Analysis:**
- Consistent ~500K-2M elements/sec (excluding cold start)
- Memory: ~0.02 KB per element
- Quadratic scaling with tensor dimensions

---

## 6. backdoor_dump_tensor Performance

Dump tensor using TileMapping.

| Size (C×H×W) | Elements | Time (ms) | Throughput (elem/s) | Memory (KB) |
|--------------|----------|-----------|---------------------|-------------|
| 3×8×8 | 192 | 0.18 | 1,046,487 | 4.50 |
| 3×16×16 | 768 | 0.21 | 3,593,302 | 15.89 |
| 3×32×32 | 3072 | 1.46 | 2,107,301 | 61.43 |
| 3×64×64 | 12288 | 15.01 | 818,766 | 241.05 |

**Analysis:**
- Peak throughput at medium sizes (3×16×16)
- Consistent ~1-2M elements/sec at scale
- Memory: ~0.02 KB per element

---

## 7. Roundtrip Performance

Complete load → dump cycle.

### Tensor Roundtrip

| Size (C×H×W) | Elements | Time (ms) | Throughput (elem/s) |
|--------------|----------|-----------|---------------------|
| 3×8×8 | 192 | 0.25 | 753,884 |
| 3×16×16 | 768 | 0.41 | 1,886,408 |
| 3×32×32 | 3072 | 3.07 | 999,277 |

### Matrix Roundtrip

| Size (rows×cols) | Elements | Time (ms) | Throughput (elem/s) |
|------------------|----------|-----------|---------------------|
| 16×4 | 64 | 0.18 | 348,305 |
| 64×4 | 256 | 0.69 | 373,566 |
| 256×4 | 1024 | 2.46 | 416,892 |
| 1024×4 | 4096 | 7.57 | 540,988 |

**Analysis:**
- Matrix roundtrip shows consistent ~350-550K elements/sec
- Tensor roundtrip varies: 750K-1.9M elements/sec
- Better performance at medium sizes for tensors

---

## Recommendations

1. **Batch Size:** Use larger batches (100+) for better throughput amortization
2. **Memory:** Dump operations allocate more memory; reuse buffers when possible
3. **Cold Start:** First operation has overhead; warm up buffer in critical paths
4. **Tensor vs Matrix:** Matrix operations are more predictable; tensor operations vary more with size

---

## Profiling Breakdown (XLarge Size)

### backdoor_read(10000)

| Step | Calls | Total (ms) | Per Call (ms) | % | Status |
|------|-------|------------|---------------|---|--------|
| List comprehension read | 1 | <0.01 | <0.01 | <1% | ✓ Fast |

**Total:** ~0.4 ms

**Analysis:** Read operation is very fast with no significant bottlenecks.

---

### backdoor_write(10000)

| Step | Calls | Total (ms) | Per Call (ms) | % | Status |
|------|-------|------------|---------------|---|--------|
| Loop and write | 1 | <0.01 | <0.01 | <1% | ✓ Fast |

**Total:** ~0.5 ms

**Analysis:** Write operation is very fast with no significant bottlenecks.

---

### backdoor_load_matrix(1024×4)

| Step | Calls | Total (ms) | Per Call (ms) | % | Status |
|------|-------|------------|---------------|---|--------|
| Main loop (buffer.py:161) | 1 | 5.0 | 5.0 | 36% | ⚠️ Bottleneck |
| Vector.from_tensor (vector.py:182) | 1024 | 3.0 | 0.003 | 21% | |
| Vector packing | 1024 | 1.0 | 0.001 | 7% | |

**Total:** ~14 ms

**Analysis:**
- Main loop takes 36% of time (address iteration and assignment)
- Vector conversion takes 21% (tensor row → packed integer)
- Recommendation: Consider batch processing or vectorization

---

### backdoor_dump_matrix(1024×4)

| Step | Calls | Total (ms) | Per Call (ms) | % | Status |
|------|-------|------------|---------------|---|--------|
| Main loop (buffer.py:190) | 1 | 6.0 | 6.0 | 30% | ⚠️ Bottleneck |
| Vector.from_int (vector.py:99) | 1024 | 5.0 | 0.005 | 25% | |
| Memory copy (copy.py) | 1025 | 1.0 | 0.001 | 5% | |

**Total:** ~20 ms

**Analysis:**
- Main loop takes 30% (deepcopy overhead)
- Vector unpacking takes 25% (integer → tensor row)
- Recommendation: Avoid deepcopy, use direct slicing

---

### backdoor_load_tensor(3×64×64)

| Step | Calls | Total (ms) | Per Call (ms) | % | Status |
|------|-------|------------|---------------|---|--------|
| einops rearrange (_ops.py:370) | 38 | 15473 | 407 | **55%** | ⚠️ Bottleneck |
| einops setup/validation | 98 | 2551 | 26 | 9% | |
| Address generation | 22 | 2158 | 98 | 8% | |
| Vector conversion | 3 | 1428 | 476 | 5% | |

**Total:** ~28411 ms (first run with torch compilation)

**Analysis:**
- **First run includes torch JIT compilation overhead (~28s)**
- einops rearrange dominates (55%) - this is the tensor transformation step
- Subsequent runs are much faster (~2ms per profiling output)
- Recommendation: Pre-warm torch kernels for performance-critical paths

---

### backdoor_dump_tensor(3×64×64)

| Step | Calls | Total (ms) | Per Call (ms) | % | Status |
|------|-------|------------|---------------|---|--------|
| Main loop & address gen (buffer.py:269) | 1 | 17.0 | 17.0 | 24% | |
| Vector.from_cocotb (vector.py:143) | 3 | 17.0 | 5.7 | 24% | |
| Vector unpacking (vector.py:99) | 3 | 17.0 | 5.7 | 24% | |
| einops rearrange (einops.py:545) | 1 | <0.01 | <0.01 | <1% | ✓ Fast |

**Total:** ~70 ms

**Analysis:**
- Vector conversion and unpacking takes ~48% combined
- Address generation and reading takes 24%
- einops rearrange is fast (already cached from load)
- Recommendation: Cache parsed addresses for repeated dumps

---

## Optimization Priorities

Based on profiling, here are the optimization priorities:

| Priority | Operation | Issue | Potential Gain |
|----------|-----------|-------|----------------|
| 🔴 High | `backdoor_load_tensor` | First-run compilation | Pre-warm: 28s → 2ms |
| 🟡 Medium | `backdoor_load_matrix` | Vector conversion loop | Batch processing: ~30% |
| 🟡 Medium | `backdoor_dump_matrix` | Deepcopy overhead | Direct slicing: ~30% |
| 🟢 Low | `backdoor_read/write` | Already optimal | - |

---

## Test Details

**Test File:** `tests/test_buffer_backdoor_performance.py`

**Classes:**
- `TestBufferBackdoorPerformance` - Individual operation benchmarks
- `TestBufferBackdoorRoundtripPerformance` - Complete cycle benchmarks
- `TestBufferBackdoorProfiling` - Step-by-step profiling breakdown with cProfile

**Size Scales:**
- Small: 10-100 elements / 3×8×8 tensors
- Medium: 100-1000 elements / 3×16×16 tensors
- Large: 1000-10000 elements / 3×32×32 tensors
- XLarge: 10000+ elements / 3×64×64 tensors

**Profiling Method:**
- Uses Python's `cProfile` for detailed timing breakdown
- Profiles the largest size (xlarge) for representative results
- Shows cumulative time, calls, and percentage per step
- Bottlenecks (>30%) marked with ⚠️
