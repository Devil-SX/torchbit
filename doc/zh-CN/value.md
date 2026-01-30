# 字节序

| 类型 | 字节序 |
| --- | --- |
| x86 | 小端序 |
| NVIDIA GPU | 小端序 |
| 默认 numpy | `=` 与 CPU 相同 |
| 默认 torch (device = "cpu") | `=` 与 CPU 相同 |
| Verilog | 大端序 |


x86 的默认字节序是小端序，可以通过 `lscpu | grep "Byte Order"` 检查。但 Verilog 的 `$readmemh` 和 `$fread` 函数使用大端序，这与 x86 不同。

torchbit 默认将 binfile / hexfile 视为**小端序**。在 `sv` 文件夹中有一个小端序的系统 Verilog 文件接口。

例如一个 bf16 类型的 tensor `[1., 2., 3.]`

| 十进制 | 大端序 | 小端序 |
|--- | --- | --- |
| 1 | 0x3F80 | 0x803F |
| 2 | 0x4000 | 0x0040 |
| 3 | 0x4040 | 0x4040 |

转储到 hexfile / binfile（小端序，默认）

```
80 3F 00 40 40 40
```


转储到 hexfile / binfile（大端序）

```
40 40 40 00 3F 80
```

转储到 cocotb（Verilog 中的 Value）

```
value = 0x3F80 | 0x4000 << 16 | 0x4040 << 32
```



**实现细节**

需要注意三个概念：字节、十进制值、数据类型

`-` 表示保持不变，`?` 表示未知，`x` 表示改变

| 方法 | 字节 | 十进制值 | 数据类型 |
| --- | --- | --- | --- |
| torch / numpy `.view()` | - | ? | x |
| torch `.to()` / numpy `.astype()` | ? | - | x |
