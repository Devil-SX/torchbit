import torch
from pathlib import Path


def save_tensor(tensor_dict, dir_path):
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for name, tensor in tensor_dict.items():
        dtype = tensor.dtype
        dtype_str = str(dtype).replace('torch.', '')
        print(f"Saving {name} tensor with dtype {dtype} to {dir_path}/{name}_{dtype_str}.pth")
        torch.save(tensor, f"{dir_path}/{name}_{dtype_str}.pth")


