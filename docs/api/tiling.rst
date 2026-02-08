# Tiling Module

Tensor-to-memory mapping, address generation, and padding utilities.

## torchbit.tiling.tile_mapping

TileMapping for Tensor ↔ LogicSequence conversion via einops rearrangement.
Includes shortcuts: ``array_to_logic_seq``, ``logic_seq_to_array``,
``matrix_to_logic_seq``, ``logic_seq_to_matrix``.

.. automodule:: torchbit.tiling.tile_mapping
   :members:
   :undoc-members:

## torchbit.tiling.address_mapping

AddressMapping and ContiguousAddressMapping for hardware address generation.

.. automodule:: torchbit.tiling.address_mapping
   :members:
   :undoc-members:

## torchbit.tiling.pad_utils

Einops-style tensor padding utilities.

.. automodule:: torchbit.tiling.pad_utils
   :members:
   :undoc-members:
