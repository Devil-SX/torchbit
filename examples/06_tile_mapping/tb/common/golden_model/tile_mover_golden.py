"""
Golden Model for TileMover DUT

The golden model provides the reference implementation for tile transpose.
It's a pure data transformation model using einops rearrange.

This model handles the b,w,c -> w,b,c transpose operation.
The c dimension has both temporal (ct) and spatial (cs) components,
but the golden model only sees the software view (b, w, c).
"""
import torch
import einops


class TileMoverGolden:
    """Golden model for TileMover DUT.

    This is a pure functional model - no memory, no addresses, no timing.
    Just computes the correct transpose using einops rearrange.

    The tile mover transposes data from (b, w, c) layout to (w, b, c) layout:
    - Source: (batch, width, channel) order
    - Destination: (width, batch, channel) order (b and w swapped)
    """

    def transpose(self, tensor: torch.Tensor, b: int, w: int, c: int) -> torch.Tensor:
        """Perform the tile transpose operation using einops rearrange.

        Transposes data from (b, w, c) layout to (w, b, c) layout.

        Args:
            tensor: Input tensor with shape (b, w, c)
            b: Batch dimension
            w: Width dimension
            c: Channel dimension (ct * cs, where cs is spatial)

        Returns:
            Transposed tensor with shape (w, b, c)
        """
        assert tensor.shape == (b, w, c), \
            f"Input tensor shape {tensor.shape} doesn't match (b={b}, w={w}, c={c})"

        # Use einops rearrange for transpose: (b, w, c) -> (w, b, c)
        result = einops.rearrange(tensor, "b w c -> w b c")

        return result
