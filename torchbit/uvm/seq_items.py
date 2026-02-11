"""UVM sequence items wrapping torchbit data types."""
import torch


class VectorItem:
    """Sequence item wrapping a torchbit Vector.

    When pyuvm is available, this also inherits from uvm_sequence_item.
    When pyuvm is not installed, it works as a standalone data container.
    """

    def __init__(self, name: str, vector=None):
        self.name = name
        self.vector = vector

    def __eq__(self, other):
        if not isinstance(other, VectorItem):
            return False
        if self.vector is None or other.vector is None:
            return self.vector is other.vector
        return torch.equal(self.vector.tensor, other.vector.tensor)

    def __repr__(self):
        if self.vector is not None:
            return f"VectorItem({self.name}, shape={self.vector.tensor.shape}, dtype={self.vector.tensor.dtype})"
        return f"VectorItem({self.name}, None)"


class LogicSequenceItem:
    """Sequence item wrapping a torchbit LogicSequence.

    Represents a batch of packed integer values for bulk transfer.
    """

    def __init__(self, name: str, logic_seq=None):
        self.name = name
        self.logic_seq = logic_seq

    def __eq__(self, other):
        if not isinstance(other, LogicSequenceItem):
            return False
        return list(self.logic_seq or []) == list(other.logic_seq or [])

    def __repr__(self):
        length = len(self.logic_seq) if self.logic_seq else 0
        return f"LogicSequenceItem({self.name}, len={length})"
