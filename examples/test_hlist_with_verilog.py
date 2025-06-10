import torchbit
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


    x = torch.arange(0, 224 * 3).reshape(224, 3).to(torch.bfloat16)

    hlist = torchbit.Hlist.from_tensor(x)
    hlist.to_binfile(str(bin_file_path))
    hlist.to_memhexfile(str(mem_file_path))

    y = torchbit.Hlist.from_binfile(str(bin_file_path), 3, torch.bfloat16).to_tensor()
    # dump.bin is equal to tensor x
    torchbit.compare(x, y, color=True)

    subprocess.run(["verilator", "--cc", "--exe", "-Wno-fatal", "--build", str((rtl_dir / "tb.sv").resolve()), f"-I{rtl_dir.resolve()}", f"-DBIT_WIDTH={3*16}","--main", "--trace"], check=True)
    subprocess.run([Path("obj_dir/Vtb"), f"+INPUT={bin_file_path.resolve()}", f"+OUTPUT={output_bin_file_path.resolve()}"], check=True)

    y = torchbit.Hlist.from_binfile(str(output_bin_file_path), 3, torch.bfloat16).to_tensor()
    # output.bin is equal to tensor x
    torchbit.compare(x, y, color=True)