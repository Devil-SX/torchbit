# Einstein-Centered Symbolic System

AI computing centers on Tensors, which should be represented and understood through dimensions. The most suitable tool for this is Einstein notation. Let's compare two programming paradigms using the transpose operation as an example.

```python
# Native PyTorch syntax
a = torch.transpose(x, 0, 2, 3, 1)  # Requires comments to indicate meaning of each dimension, 0,2,3,1 is unclear

# einops syntax
a = einops.rearrange(x, "b c h w -> b h w c")  # Self-documenting through character notation
```

**Advantages**:
- Semantic dimension naming: `b`(batch), `c`(channel), `h`(height), `w`(width)
- No additional comments needed for understanding
- Type safety: Dimension matching can be checked at compile time
- Easy refactoring: Changing dimension order only requires modifying the pattern string


# Logic and Scheduler

An algorithm consists of two levels: **Logic** and **Scheduler**:

- **Logic**: What the computation is, describing the mathematical input-output relationship
- **Scheduler**: How the computation is done, describing the specific execution strategy

## Logic Layer

In AI program development, the PyTorch-level description typically focuses only on the logic part, using operations like einops:

```python
# Example 1: torch.einsum - matrix multiplication
# C[i,k] = sum(A[i,j] * B[j,k])
C = torch.einsum("ij,jk->ik", A, B)

# Example 2: einops.rearrange - dimension rearrangement
x = einops.rearrange(x, "b c h w -> b (c h w)")

# Example 3: Complex Attention computation
attn = torch.einsum("bqh,bkh->bqk", Q, K) / sqrt(d)
attn = torch.softmax(attn, dim=-1)
out = torch.einsum("bqk,bkh->bqh", attn, V)
```

This describes the computation process, i.e., the dimension relationships before and after variable transformation, without concern for specific hardware execution.


## Scheduler Layer

### Hardware Vector Sequence Lowering

The scheduler layer describes how a computation is executed. In the PyTorch ecosystem, this part is hidden in backend compilers and CUDA operator libraries, where the compilation process automatically generates the required scheduling strategies. However, if you need to implement hardware or compilers yourself, you must explicitly define the scheduler layer.

Here we adopt a concise philosophical strategy to define the scheduler layer: **convert multidimensional Tensors to padded two-dimensional Tensors**.

#### Why Two Dimensions?

Hardware storage can be understood as SIMD-based parallel architecture, where each time step has a certain degree of parallelism. In a 2D tensor, the first dimension represents the time dimension, and the second dimension represents the spatial dimension. For example, a `(10, 16)` tensor means processing 16 data points in parallel per time step, requiring 10 time steps to complete traversal. Here we define **tile** as the spatial dimension unit of processing per unit time, so 16 data points constitute a tile.

#### What is Padded?

Hardware dimensions are typically static fixed constants. To match hardware requirements, dimensions need to be padded to specified sizes. For example, if the hardware processes 16 data points per time step and there are 100 data points total, padding to 112 data points (a multiple of 16) is required.

#### Three-Step Conversion Process

The conversion process consists of three steps: **Pad - Tiling - Sort**

- **pad**: Pad each dimension to a multiple of a specific size
- **tiling**: Split each dimension into spatial dimension and time dimension
- **sort**: Reorder time dimension and spatial dimension in a specific order

#### Example: Scheduling a Four-Dimensional Tensor

Take a `B,C,H,W` four-dimensional Tensor as an example. Assume the hardware's basic Tile size is `C=16, W=16`:

**Step 1: Pad**

Pad the four dimensions to specific sizes, `B→B_pad, C→C_pad, H→H_pad, W→W_pad`. This process can be formally expressed as:

```python
# pseudo-function
x_padded = pad(
    x,
    "b c h w",
    tiling_descript={
        "b": 1,
        "c": 16,
        "h": 1,
        "w": 16
    }
)
```

**Step 2 & 3: Tiling and Sort**

Tiling splits the four dimensions into spatial dimension and time dimension, Sort reorders the time dimension and spatial dimension in a specific order. These two steps can be completed with one `einops.rearrange` function:

```python
x_scheduled = einops.rearrange(
    x_padded,
    "(bt bs) (ct cs) (ht hs) (wt ws) -> (bt ht wt ct) (bs hs ws cs)",
    tiling_descript={
        "bs": 1,
        "cs": 16,
        "hs": 1,
        "ws": 16
    }
)
```

#### Two-Step Understanding

The above process can be understood in two steps:

1. **Tiling**: Split the original `[B][C][H][W]` into `[Bt][Bs][Ct][Cs][Ht][Hs][Wt][Ws]`
2. **Sort**: Reorder dimensions into two parts: `[Bt][Ht][Wt][Ct]` and `[Bs][Hs][Ws][Cs]`

The final 2D matrix has the first dimension as the time dimension and the second dimension as the spatial dimension.

#### Software Tensor vs Hardware Vector Sequence

We call this 2D matrix a **Hardware Vector Sequence** (also known as **Hardware Matrix**) because it is hardware-related and carries hardware information:
- First dimension is the temporal dimension, **ordered by time: lower indices represent earlier time steps** (i.e., index 0 corresponds to the first time step, index 1 to the second, and so on)
- Second dimension is the spatial dimension

The spatial dimension of the hardware vector sequence corresponds to the **Hardware Vector**, reflecting hardware parallelism and tile size granularity. A hardware vector sequence is essentially an ordered collection of hardware vectors arranged in temporal order.

Correspondingly, the multidimensional Tensor before transformation is called a **Software Tensor**.

The transformation process from software tensor to hardware vector sequence defines a **Scheduling**, also called **Lowering**.

Conversely, the inverse transformation process from hardware vector sequence to software tensor is called **De-scheduling** or **Raising**, following the reverse order of desort-detilling-depad.


### Index Calculation

For specific hardware implementation, the spatial dimension of the hardware vector sequence represents hardware computational parallelism, and the time dimension reflects control flow. Control flow is essentially a mapping relationship describing how to map multidimensional indices to a one-dimensional scalar. Here we define two mappings: **Serialization** and **Addressing**.

#### Serialization

For the split dimensions `[Bt][Bs][Ct][Cs][Ht][Hs][Wt][Ws]`, extracting the time dimension `[Bt][Ct][Ht][Wt]` can be viewed as a four-dimensional vector. Serialization maps this four-dimensional vector to a consecutive integer sequence starting from 0:

```
idx = bt * bt_stride + ct * ct_stride + ht * ht_stride + wt * wt_stride
```

The stride values are determined according to the sorted order. After sorting, the order is `(bt ht wt ct)`, therefore:
- `ct_stride = 1`
- `wt_stride = ct_size`
- `ht_stride = wt_size * ct_size`
- `bt_stride = ht_size * wt_size * ct_size`

#### Addressing

Addressing is similar to serialization, but the formula adds a base address, and stride values can be freely specified:

```
addr = addr_base + bt * bt_stride + ct * ct_stride + ht * ht_stride + wt * wt_stride
```


### Data Structures

For such hardware vector sequences, the actual runtime hardware storage structures include:
- **Serialization structure**: Can be implemented through FIFO (First-In-First-Out) queues
- **Addressing structure**: Implemented through Buffer read/write operations
