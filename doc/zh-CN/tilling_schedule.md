# Einstein-Centered Symbolic System

AI 计算以 Tensor 为中心，Tensor 应当通过维度来表征和理解，最合适的工具是艾因斯坦符号表达。这里以 transpose 操作为例，对比两种编程范式。

```python
# PyTorch 原生写法
a = torch.transpose(x, 0, 2, 3, 1)  # 需要注释提示每个维度的含义，0,2,3,1 指代不明

# einops 写法
a = einops.rearrange(x, "b c h w -> b h w c")  # 通过字符表示，self-documenting
```

**优势说明**：
- 维度命名语义化：`b`(batch)、`c`(channel)、`h`(height)、`w`(width)
- 无需额外注释即可理解
- 类型安全：编译期可检查维度匹配
- 易于重构：改变维度顺序只需修改 pattern 字符串


# Logic and Scheduler

一个算法分为**逻辑**和**调度**两个层次：

- **逻辑 (Logic)**：计算是什么，描述数学上的输入输出关系
- **调度 (Scheduler)**：计算怎么做，描述具体的执行策略

## 逻辑层 (Logic Layer)

在 AI 程序开发中，PyTorch 层面的描述通常只关注逻辑部分，使用 einops 等操作：

```python
# Example 1: torch.einsum - 矩阵乘法
# C[i,k] = sum(A[i,j] * B[j,k])
C = torch.einsum("ij,jk->ik", A, B)

# Example 2: einops.rearrange - 维度重排
x = einops.rearrange(x, "b c h w -> b (c h w)")

# Example 3: 复杂的 Attention 计算
attn = torch.einsum("bqh,bkh->bqk", Q, K) / sqrt(d)
attn = torch.softmax(attn, dim=-1)
out = torch.einsum("bqk,bkh->bqh", attn, V)
```

这里描述了计算的过程，即变量变换前后的维度关系，但不关心具体在硬件上如何执行。


## 调度层 (Scheduler)

### 硬件向量序列降低 (Hardware Vector Sequence Lowering)

调度层描述了一个计算如何执行。在 PyTorch 生态中，这部分隐藏在后端编译器和 CUDA 算子库中，编译过程会自动生成所需的调度策略。但如果需要自行实现硬件或编译器，就必须显式地定义调度层。

这里采用一种简洁的哲学策略来定义调度层：**将多维 Tensor 转换为填充后的二维 Tensor**。

#### 为什么是二维？

可以将硬件存储理解为基于 SIMD 的并行架构，每个时间步具有一定的并行度。二维 tensor 中，第一维表示时间维度，第二维表示空间维度。例如一个 `(10, 16)` 的 tensor，表示每个时间步并行处理 16 个数据，需要 10 个时间步才能遍历完毕。这里定义 **tile** 为空间维单位时间处理的粒度，因此 16 个数据构成一个 tile。

#### 什么是 Padded？

硬件维度通常是静态的固定常量。为了匹配硬件要求，需要将维度填充到指定尺寸。例如硬件每个时间步处理 16 个数据，若共有 100 个数据，则需要 pad 到 112 个数据（16 的倍数）。

#### 三步转换流程

转换过程分为三步：**Pad（填充） - Tiling（分块） - Sort（排序）**

- **pad**：将每个维度填充到特定尺寸的倍数
- **tiling**：将每个维度拆分为空间维度和时间维度
- **sort**：将时间维度和空间维度按特定顺序重新排序

#### 示例：四维 Tensor 的调度

以 `B,C,H,W` 四维 Tensor 为例，假设硬件的基础 Tile 尺寸为 `C=16, W=16`：

**步骤 1：Pad**

将四个维度填充到特定尺寸，`B→B_pad, C→C_pad, H→H_pad, W→W_pad`。这个过程可以用形式化方式表述：

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

**步骤 2 & 3：Tiling 和 Sort**

Tiling 将四个维度拆分为空间维度和时间维度，Sort 将时间维度和空间维度按特定顺序排序。这两步可以用一个 `einops.rearrange` 函数完成：

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

#### 两步理解

可以将上述过程理解为两步：

1. **Tiling**：将原始的 `[B][C][H][W]` 拆分为 `[Bt][Bs][Ct][Cs][Ht][Hs][Wt][Ws]`
2. **Sort**：将维度重新排序为 `[Bt][Ht][Wt][Ct]` 和 `[Bs][Hs][Ws][Cs]` 两部分

最终得到的二维矩阵中，第一维是时间维度，第二维是空间维度。

#### 软件 Tensor 与硬件向量序列

我们将这种二维矩阵称为**硬件向量序列 (Hardware Vector Sequence)**，也可以称为**硬件矩阵 (Hardware Matrix)**，因为它与具体硬件相关，携带了硬件信息：
- 第一维是处理时间维度，**索引有序，索引越低代表时间越早**（即 index 0 对应第一个时间步，index 1 对应第二个时间步，依此类推）
- 第二维是空间维度

硬件向量序列的空间维度对应**硬件向量 (Hardware Vector)**，反映了硬件并行度和 tile size 的粒度。硬件向量序列本质上是一个按时间顺序排列的硬件向量的有序集合。

相应地，变换之前的多维 Tensor 称为**软件张量 (Software Tensor)**。

软件张量到硬件向量序列的变换过程定义了一种**调度 (Scheduling)**，也称为**降低 (Lowering)**。

反之，硬件向量序列到软件张量的逆变换过程称为**逆调度 (De-scheduling)** 或**提升 (Raising)**，遵循 desort-detilling-depad 的逆序过程。


### 索引计算

具体到硬件实现，硬件向量序列的空间维度代表硬件计算并行度，时间维度反映控制流。控制流本质上是一种映射关系，描述了将多维索引映射到一维标量的方法。这里定义两种映射：**序列化**和**地址化**。

#### 序列化 (Serialization)

对于拆分后的维度 `[Bt][Bs][Ct][Cs][Ht][Hs][Wt][Ws]`，取出时间维度 `[Bt][Ct][Ht][Wt]`，这可以看作一个四维向量。序列化就是将这个四维向量映射到一组从 0 开始的连续整数序列上：

```
idx = bt * bt_stride + ct * ct_stride + ht * ht_stride + wt * wt_stride
```

其中 stride 的值根据 sort 之后的顺序确定。sort 后的顺序是 `(bt ht wt ct)`，因此：
- `ct_stride = 1`
- `wt_stride = ct_size`
- `ht_stride = wt_size * ct_size`
- `bt_stride = ht_size * wt_size * ct_size`

#### 地址化 (Addressing)

地址化与序列化类似，但公式中添加了基地址，且 stride 的值可以自由指定：

```
addr = addr_base + bt * bt_stride + ct * ct_stride + ht * ht_stride + wt * wt_stride
```


### 数据结构

对于这种硬件向量序列，实际运行时对应的硬件存储结构：
- **序列化结构**：可通过 FIFO（先进先出）队列实现
- **地址化结构**：通过 Buffer 读写实现
