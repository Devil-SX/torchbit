import torchbit as torchbit
import os
import torch
from pathlib import Path

if __name__ == "__main__":
    dir_path = Path("./temp")
    x = torch.arange(0, 224 * 3).reshape(224, 3).to(torch.bfloat16)

    matrix = torchbit.core.Matrix.from_tensor(x)
    matrix.to_binfile(dir_path / "dump.bin")
    matrix.to_memhexfile(dir_path / "dump.mem")

    y = torchbit.core.Matrix.from_binfile(dir_path / "dump.bin", 3, torch.bfloat16).to_tensor()
    print(y)
    torchbit.debug.compare(x, y)