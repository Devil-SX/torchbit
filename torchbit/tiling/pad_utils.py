"""
Tensor padding utilities using einops-style dimension notation.

Provides functions for padding tensors to meet hardware alignment requirements
using an intuitive einops-based interface. Essential for tilling operations
where tensor dimensions need to be aligned to specific tile sizes.

The interface follows the einops convention where dimensions are named
(e.g., "b c h w" for batch, channel, height, width) and alignment
is specified per dimension using a dictionary.
"""
import torch
import torch.nn.functional as F


def get_padlen(dim_size: int, align: int) -> int:
    """Calculate padding length needed to align a dimension.

    Computes how many elements need to be added to reach the next
    aligned boundary.

    Args:
        dim_size: Current dimension size.
        align: Alignment boundary (must be positive).

    Returns:
        Number of elements to add. Returns 0 if already aligned.

    Example:
        >>> get_padlen(232, 16)
        8  # Need 8 elements to reach 240 (next multiple of 16)
        >>> get_padlen(240, 16)
        0  # Already aligned
    """
    return (align - dim_size % align) % align


def pad(
    tensor: torch.Tensor,
    pattern: str,
    tiling_descript: dict,
    pad_pos: str = "last",
    pad_value: float = 0
) -> torch.Tensor:
    """Pad a tensor to achieve alignment per dimension using einops notation.

    Pads each dimension to a multiple of its specified alignment value.
    Uses einops-style dimension naming for intuitive hardware alignment.

    Args:
        tensor: Input PyTorch tensor.
        pattern: Einops-style dimension pattern (e.g., "b c h w").
        tiling_descript: Dict mapping dimension names to alignment values.
            Dimensions with alignment of 1 are not padded.
            Example: {"b": 1, "c": 16, "h": 1, "w": 16}
        pad_pos: Padding position, "last" (default) or "begin".
            "last" adds padding at the end of each dimension.
            "begin" adds padding at the start of each dimension.
        pad_value: Value to use for padding elements. Defaults to 0.

    Returns:
        A new tensor with padded dimensions. The pattern is preserved
        but each dimension size is a multiple of its alignment.

    Raises:
        AssertionError: If pattern doesn't match tensor shape or
            pad_pos is not "last" or "begin".

    Example:
        >>> x = torch.randn(2, 3, 224, 224)  # b=2, c=3, h=224, w=224
        >>> # Pad to align c to 16 and h,w to 32 (default: pad at end)
        >>> x_padded = pad(
        ...     x,
        ...     "b c h w",
        ...     tiling_descript={"b": 1, "c": 16, "h": 32, "w": 32}
        ... )
        >>> x_padded.shape
        torch.Size([2, 16, 224, 256])
        >>>
        >>> # Pad at beginning instead
        >>> x_padded_begin = pad(
        ...     x,
        ...     "b c h w",
        ...     tiling_descript={"b": 1, "c": 16, "h": 32, "w": 32},
        ...     pad_pos="begin"
        ... )
        >>> x_padded_begin.shape
        torch.Size([2, 16, 224, 256])
        >>>
        >>> # For B,C,H,W with tile sizes C=16, W=16
        >>> x = torch.randn(4, 13, 32, 15)
        >>> x_padded = pad(
        ...     x,
        ...     "b c h w",
        ...     tiling_descript={"b": 1, "c": 16, "h": 1, "w": 16}
        ... )
        >>> x_padded.shape
        torch.Size([4, 16, 32, 16])
    """
    assert pad_pos in ("last", "begin"), \
        f"pad_pos must be 'last' or 'begin', got '{pad_pos}'"

    # Parse pattern and validate
    dims = pattern.split()
    assert len(dims) == len(tensor.shape), \
        f"Pattern '{pattern}' has {len(dims)} dims but tensor has {len(tensor.shape)} dims"

    # Calculate padding for each dimension
    current_shape = tensor.shape
    padded_shape = list(current_shape)
    pad_sizes = []

    for i, dim in enumerate(dims):
        align = tiling_descript.get(dim, 1)
        if align > 1:
            pad_len = get_padlen(current_shape[i], align)
            padded_shape[i] += pad_len
            pad_sizes.append(pad_len)
        else:
            pad_sizes.append(0)

    # Use F.pad for efficient padding
    # F.pad expects padding from last dimension first, in (left, right) pairs
    pad_config = []
    for i in range(len(dims) - 1, -1, -1):
        if pad_sizes[i] > 0:
            if pad_pos == "last":
                pad_config.extend([0, pad_sizes[i]])  # pad at end (right)
            else:  # pad_pos == "begin"
                pad_config.extend([pad_sizes[i], 0])  # pad at start (left)
        else:
            pad_config.extend([0, 0])

    if any(pad_config):
        padded = F.pad(tensor, pad_config, value=pad_value)
        return padded
    return tensor.clone()


def depad(
    tensor: torch.Tensor,
    pattern: str,
    original_shape: tuple[int, ...],
    pad_pos: str = "last"
) -> torch.Tensor:
    """Remove padding to restore original tensor dimensions.

    Extracts the original tensor shape from a padded tensor by slicing
    each dimension to its original size.

    Args:
        tensor: Padded PyTorch tensor.
        pattern: Einops-style dimension pattern (e.g., "b c h w").
        original_shape: Target shape after removing padding.
        pad_pos: Padding position used during padding, "last" (default) or "begin".
            Must match the pad_pos value used when padding.

    Returns:
        A new tensor with the original shape.

    Raises:
        AssertionError: If pattern doesn't match tensor shape,
            original_shape is larger than current shape, or
            pad_pos is not "last" or "begin".

    Example:
        >>> x = torch.randn(4, 13, 32, 15)
        >>> x_padded = pad(x, "b c h w", tiling_descript={"b": 1, "c": 16, "h": 1, "w": 16})
        >>> x_restored = depad(x_padded, "b c h w", x.shape)
        >>> x_restored.shape
        torch.Size([4, 13, 32, 15])
        >>>
        >>> # Padding at beginning
        >>> x_padded_begin = pad(x, "b c h w", tiling_descript={"b": 1, "c": 16, "h": 1, "w": 16}, pad_pos="begin")
        >>> x_restored = depad(x_padded_begin, "b c h w", x.shape, pad_pos="begin")
        >>> x_restored.shape
        torch.Size([4, 13, 32, 15])
    """
    assert pad_pos in ("last", "begin"), \
        f"pad_pos must be 'last' or 'begin', got '{pad_pos}'"

    dims = pattern.split()
    assert len(dims) == len(tensor.shape), \
        f"Pattern '{pattern}' has {len(dims)} dims but tensor has {len(tensor.shape)} dims"
    assert len(original_shape) == len(tensor.shape), \
        f"Original shape has {len(original_shape)} dims but tensor has {len(tensor.shape)} dims"

    # Verify each original dimension is <= current dimension
    for i, (orig, curr) in enumerate(zip(original_shape, tensor.shape)):
        assert orig <= curr, \
            f"Original dim {i} ({orig}) is larger than current dim ({curr})"

    # Create slicing specification
    slices = []
    for i, (orig, curr) in enumerate(zip(original_shape, tensor.shape)):
        if orig == curr:
            slices.append(slice(None))  # No padding, take all
        elif pad_pos == "last":
            slices.append(slice(0, orig))  # Take from start
        else:  # pad_pos == "begin"
            pad_len = curr - orig
            slices.append(slice(pad_len, None))  # Take from end

    return tensor[tuple(slices)].clone()


def depad_like(
    tensor: torch.Tensor,
    pattern: str,
    reference_tensor: torch.Tensor,
    pad_pos: str = "last"
) -> torch.Tensor:
    """Remove padding to match the shape of a reference tensor.

    Convenience function that uses another tensor's shape as the target
    for depadding, useful in pair operations.

    Args:
        tensor: Padded PyTorch tensor.
        pattern: Einops-style dimension pattern (e.g., "b c h w").
        reference_tensor: Tensor whose shape will be used as target.
        pad_pos: Padding position used during padding, "last" (default) or "begin".
            Must match the pad_pos value used when padding.

    Returns:
        A new tensor with the same shape as reference_tensor.

    Example:
        >>> x = torch.randn(4, 13, 32, 15)
        >>> x_padded = pad(x, "b c h w", tiling_descript={"b": 1, "c": 16, "h": 1, "w": 16})
        >>> x_restored = depad_like(x_padded, "b c h w", x)
        >>> x_restored.shape == x.shape
        True
    """
    return depad(tensor, pattern, reference_tensor.shape, pad_pos=pad_pos)


if __name__ == '__main__':
    # Test basic padding at end (default)
    print("=== Basic pad/depad test (pad_pos='last') ===")
    x = torch.randn(2, 232)
    print(f"original shape: {x.shape}")

    x_pad = pad(x, "b c", tiling_descript={"b": 1, "c": 16})
    print(f"padded shape: {x_pad.shape}")

    x_depad = depad(x_pad, "b c", x.shape)
    print(f"depadded shape: {x_depad.shape}")

    # Test padding at beginning
    print("\n=== Basic pad/depad test (pad_pos='begin') ===")
    x_pad_begin = pad(x, "b c", tiling_descript={"b": 1, "c": 16}, pad_pos="begin")
    print(f"padded shape (begin): {x_pad_begin.shape}")

    x_depad_begin = depad(x_pad_begin, "b c", x.shape, pad_pos="begin")
    print(f"depadded shape (begin): {x_depad_begin.shape}")
    assert torch.allclose(x, x_depad_begin), "Begin roundtrip failed!"
    print("Begin roundtrip successful!")

    # Test multi-dimensional padding (like tilling_schedule.md example)
    print("\n=== Multi-dimensional pad test ===")
    x = torch.randn(4, 13, 32, 15)  # b=4, c=13, h=32, w=15
    print(f"original shape: {x.shape}")

    x_padded = pad(
        x,
        "b c h w",
        tiling_descript={"b": 1, "c": 16, "h": 1, "w": 16}
    )
    print(f"padded shape: {x_padded.shape}")

    x_restored = depad(x_padded, "b c h w", x.shape)
    print(f"restored shape: {x_restored.shape}")
    assert torch.allclose(x, x_restored), "Roundtrip failed!"
    print("Roundtrip successful!")

    # Test with begin position
    print("\n=== Multi-dimensional pad test (pad_pos='begin') ===")
    x_padded_begin = pad(
        x,
        "b c h w",
        tiling_descript={"b": 1, "c": 16, "h": 1, "w": 16},
        pad_pos="begin"
    )
    print(f"padded shape (begin): {x_padded_begin.shape}")

    x_restored_begin = depad(x_padded_begin, "b c h w", x.shape, pad_pos="begin")
    print(f"restored shape (begin): {x_restored_begin.shape}")
    assert torch.allclose(x, x_restored_begin), "Begin roundtrip failed!"
    print("Begin roundtrip successful!")

    print("\n=== All tests passed! ===")
