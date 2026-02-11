"""Typed logic sequence for hardware verification data.

Provides LogicSequence, a list subclass for representing sequences of
packed integer values used in hardware verification interfaces.
"""


class LogicSequence(list):
    """A typed sequence of packed integers for hardware verification data.

    LogicSequence is a thin ``list`` subclass used throughout torchbit to
    represent ordered sequences of packed-integer values.  Each element is
    a Python ``int`` that encodes one hardware word (the bit-width is
    determined by context, e.g. the Buffer width or TileMapping dtype).

    Typical sources:
    - ``TileMapping.to_int_sequence(tensor)`` — tile a tensor into a
      LogicSequence for loading into a Buffer.
    - ``Buffer.backdoor_read(addr_list)`` — read multiple addresses.
    - ``Monitor.dump()`` — collected data from a Cocotb monitor.

    Typical consumers:
    - ``TileMapping.to_tensor(logic_seq)`` — reconstruct a tensor.
    - ``Buffer.backdoor_write(addr_list, logic_seq)`` — bulk write.
    - ``Driver.load(logic_seq)`` — enqueue data for a Cocotb driver.

    Why not a plain list?  Using a dedicated type makes intent explicit
    and enables future type-checking or validation without breaking the
    existing ``list`` interface.

    Example:
        >>> from torchbit.core import LogicSequence
        >>> seq = LogicSequence([0xDEAD, 0xBEEF, 0xCAFE])
        >>> seq.append(0xF00D)
        >>> len(seq)
        4
    """
    pass


# Backward compatibility alias
IntSequence = LogicSequence
