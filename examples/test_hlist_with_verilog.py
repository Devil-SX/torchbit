import torchbit as torchbit
import os
import torch
from pathlib import Path
import subprocess

if __name__ == "__main__":
    dir_path = Path("./temp")
    bin_file_path = dir_path / "dump.bin"
    mem_file_path = dir_path / "dump.mem"
    output_bin_file_path = dir_path / "output.bin"

    rtl_dir = Path("../sv")


    x = torch.randn(3, 10, dtype=torch.bfloat16)
    hlist = torchbit.core.HwMatrix.from_tensor(x)
    bin_file_path = dir_path / "dump.bin"
    hlist.to_binfile(bin_file_path)

    y = torchbit.core.HwMatrix.from_binfile(str(bin_file_path), 3, torch.bfloat16).to_tensor()
    
    torchbit.debug.compare(x, y, color=True)

    # run verilog
    output_bin_file_path = dir_path / "dump_out.bin"
    run_verilog(bin_file_path, output_bin_file_path)

    y = torchbit.core.HwMatrix.from_binfile(str(output_bin_file_path), 3, torch.bfloat16).to_tensor()

    torchbit.debug.compare(x, y, color=True)