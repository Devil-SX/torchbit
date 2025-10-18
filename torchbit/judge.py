import torch
import numpy as np
from pathlib import Path

def compare(input, other, rtol=1e-3, atol=1e-1, visualize=False, save_path=None, color=None):
    all_is_numpy = isinstance(input, np.ndarray) and isinstance(other, np.ndarray)
    all_is_torch = isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor)  
    assert all_is_numpy or all_is_torch, f"Input and other must be either both numpy arrays or both torch tensors, while got {type(input)} and {type(other)}"    
    assert input.shape == other.shape, f"Shape mismatch: {input.shape} != {other.shape}"
    shape = input.shape

    if all_is_numpy:
        input = torch.from_numpy(input)
        other = torch.from_numpy(other)
    input = input.to(torch.float32)
    other = other.to(torch.float32)
    input = input.cpu()
    other = other.cpu()
    is_equal = torch.allclose(input, other, rtol=rtol, atol=atol)

    abs_diff = torch.abs(input - other)
    max_abs_diff = torch.max(abs_diff).item()
    max_abs_diff_idx = torch.argmax(abs_diff)
    max_abs_diff_input = input.flatten()[max_abs_diff_idx].item()
    max_abs_diff_other = other.flatten()[max_abs_diff_idx].item()

    rel_diff = torch.where(input != 0, abs_diff / torch.abs(input), abs_diff)
    max_rel_diff = torch.max(rel_diff).item()
    max_rel_diff_idx = torch.argmax(rel_diff)
    max_rel_diff_input = input.flatten()[max_rel_diff_idx].item()
    max_rel_diff_other = other.flatten()[max_rel_diff_idx].item()

    mean_ref_diff = torch.mean(rel_diff).item()

    mse = abs_diff.pow(2).mean().item()

    max_abs_diff_idx_tensor = torch.unravel_index(torch.tensor(max_abs_diff_idx), shape)
    max_abs_diff_idx_tuple = tuple(idx.item() for idx in max_abs_diff_idx_tensor)
    max_rel_diff_idx_tensor = torch.unravel_index(torch.tensor(max_rel_diff_idx), shape)
    max_rel_diff_idx_tuple = tuple(idx.item() for idx in max_rel_diff_idx_tensor)
    print(f"Shape: {shape}")
    print(f"Max abs diff:\t {max_abs_diff} at {max_abs_diff_idx_tuple} \t {max_abs_diff_input}/{max_abs_diff_other}")
    print(f"Max rel diff:\t {max_rel_diff} at {max_rel_diff_idx_tuple} \t {max_rel_diff_input}/{max_rel_diff_other}")
    print(f"Mean rel diff:\t {mean_ref_diff}")
    print(f"MSE:\t {mse}")
    print(f"rtol: {rtol}, atol: {atol}")
    is_equal_str = str(is_equal)
    absolute_equal_str = "Absolutely " if (max_abs_diff == 0) else ""
    is_equal_str = absolute_equal_str + is_equal_str
    if color:
        prefix = "\033[1;32m" if is_equal else "\033[1;31m"
        suffix = "\033[0m"
        is_equal_str = prefix  + is_equal_str + suffix
    print(f"Is equal: {is_equal_str}")

    if visualize:
        import matplotlib.pyplot as plt  
        input_np = input.numpy().flatten()
        other_np = other.numpy().flatten()

        plt.figure(figsize=(12, 6))
        plt.hist(input_np, bins=100, alpha=0.5, label='Test', color='red')
        plt.hist(other_np, bins=100, alpha=0.5, label='Ref', color='green')
        plt.title('Histogram of Test and Reference Tensors')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()
    
    return is_equal

if __name__ == "__main__":
    import torch

    size = (100, 100)
    input = torch.randn(size)
    other = torch.rand(size) * 10

    compare(input, other, visualize=True)