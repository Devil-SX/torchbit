import torch
import numpy as np

def compare(input, other, rtol=1e-3, atol=1e-1, visualize=False, save_path=None):
    assert input.shape == other.shape, f"Shape mismatch: {input.shape} != {other.shape}"
    input = input.to(torch.float32)
    other = other.to(torch.float32)
    is_equal = torch.allclose(input, other, rtol=rtol, atol=atol)

    abs_diff = torch.abs(input - other)
    max_abs_diff = torch.max(abs_diff).item()

    rel_diff = torch.where(input != 0, abs_diff / torch.abs(input), abs_diff)
    max_rel_diff = torch.max(rel_diff).item()
    mean_ref_diff = torch.mean(rel_diff).item()

    mse = abs_diff.pow(2).mean().item()
    print(f"Max abs diff:\t {max_abs_diff}")
    print(f"Max rel diff:\t {max_rel_diff}")
    print(f"Mean rel diff:\t {mean_ref_diff}")
    print(f"MSE:\t {mse}")
    print(f"rtol: {rtol}, atol: {atol}")
    print(f"Is equal: {is_equal}")

    if visualize:
        import matplotlib.pyplot as plt  
        input_np = input.numpy().flatten()
        other_np = other.numpy().flatten()

        # 绘制直方图
        plt.figure(figsize=(12, 6))
        plt.hist(input_np, bins=100, alpha=0.5, label='Input', color='blue')
        plt.hist(other_np, bins=100, alpha=0.5, label='Other', color='red')
        plt.title('Histogram of Input and Other Tensors')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

if __name__ == "__main__":
    import torch

    size = (100, 100)
    input = torch.randn(size)
    other = torch.rand(size) * 10

    compare(input, other, visualize=True)