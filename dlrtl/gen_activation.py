import torch
from pathlib import Path

def random_mvm(B, O, I):
    tensor_if = torch.randint(-128, 127, (B, I), dtype=torch.int8) # [B, I]
    tensor_w = torch.randint(-128, 127, (O, I), dtype=torch.int8) # [O, I]
    tensor_of = tensor_if @ tensor_w.T
    return {f"if_b{B}_i{I}": tensor_if, f"w_o{O}_i{I}": tensor_w, f"of_b{B}_o{O}": tensor_of}

def random_ssm(L, D, N):
    pass

def save_tensor(tensor_dict, dir_path):
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for name, tensor in tensor_dict.items():
        dtype = tensor.dtype
        dtype_str = str(dtype).replace('torch.', '')
        print(f"Saving {name} tensor with dtype {dtype} to {dir_path}/{name}_{dtype_str}.pth")
        torch.save(tensor, f"{dir_path}/{name}_{dtype_str}.pth")


