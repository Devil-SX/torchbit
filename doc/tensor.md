# Python Driver/Monitor

Vector: Hardware 1D Tensor

- pytorch 1D-tensor -> cocotb input, `Vector.from_tensor(tensor).to_cocotb()`
- cocotb output -> pytorch 1D-tensor, `Vector.from_cocotb(dut.io_xx.value, num, dtype).to_tensor()`

# Verilog Driver/Monitor

Matrix: Hardware 2D Tensor, use pointer to read

- pytorch 2D-tensor -> memhex / bin file
- memhex / bin file -> reader -> 1D logic
- 1D logic ->  collector -> memhex / bin file
- memhex / bin file -> pytorch 2D-tensor