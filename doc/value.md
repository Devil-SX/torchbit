
# Endianness

| Type | Endianness | 
| --- | --- |
| x86 | little endian | 
| NVIDIA GPU | little endian |
| default numpy | `=` same as CPU |
| default torch (device = "cpu") | `=` same as CPU |
| Verilog | big endian |


The default endianness of x86 is little endian, which can be checked by `lscpu | grep "Byte Order"`. But Verilog `$readmemh` `$fread` function use big-endian, which is different.

torchbit treats binfile / hexfile as **LITTLE-ENDIAN** defaultly. There is a little-endian system verilog file interface in `sv` folder.

Such a bf16 tensor `[1., 2., 3.]`

| Decimal |  Big Endian | Little Endian|
|---  | --- | --- |
| 1 | 0x3F80 | 0x803F |
| 2 | 0x4000 | 0x0040 |
| 3 | 0x4040 | 0x4040 |

Dump to hexfile / binfile (little-endian, default)

```
80 3F 00 40 40 40
```


Dump to hexfile / binfile (big-endian)

```
40 40 40 00 3F 80
```

Dump to cocotb (Value in Verilog)

```
value = 0x3F80 | 0x4000 << 16 | 0x4040 << 32
```



**Implementation Details**

There are three concept need to be concerned: bytes,decimal value, data type

`-` means keep same, `?` means unknown, `x` means change

| Method | Bytes | Decimal Value| Data Type |
| --- | --- | --- | --- |
| torch / numpy `.view()` | - | ? | x |
| torch `.to()` / numpy `.astype()` | ? | - | x |
