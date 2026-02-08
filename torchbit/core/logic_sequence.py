"""Typed logic sequence for hardware verification data.

Provides LogicSequence, a list subclass for representing sequences of
packed integer values used in hardware verification interfaces.
"""


class LogicSequence(list):
    """A typed sequence of integers for hardware verification data."""
    pass


# Backward compatibility alias
IntSequence = LogicSequence
