import torch

def compare(input, other, rtol=1e-3, atol=1e-1):
    assert input.shape == other.shape, f"Shape mismatch: {input.shape} != {other.shape}"
    input = input.to(torch.float32)
    other = other.to(torch.float32)
    is_equal = torch.allclose(input, other, rtol=rtol, atol=atol)

    abs_diff = torch.abs(input-other)
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