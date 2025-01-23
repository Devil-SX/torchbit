import dlrtl
import os
import torch

if __name__ == "__main__":
    dir_path = "./temp"
    B = 16
    O = 16
    I = 16
    tensor_dict = dlrtl.random_mvm(B, O, I)
    dlrtl.save_tensor(tensor_dict, dir_path)

    for file_name in os.listdir(dir_path):
        if file_name.endswith(".pth"):
            file_path = os.path.join(dir_path, file_name)
            tensor = torch.load(file_path)
            print(f"Loaded tensor from {file_path}")
            dlrtl.tensor2memhex(tensor, file_path.replace(".pth", ".hex"))