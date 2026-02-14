"""
Golden Model for Pipeline DUT

The pipeline delays input by DELAY cycles without modifying data.
The golden model simply predicts that output values equal input values
in the same order.
"""
import torch


class PipeGolden:
    """Golden model for the Pipe DUT.

    The pipe is a pure delay element: output[i] == input[i] for all i,
    shifted by DELAY clock cycles. The golden model ignores timing and
    only predicts the value sequence.
    """

    def __init__(self, delay: int = 4):
        """Initialize the golden model.

        Args:
            delay: Pipeline depth (number of delay stages).
        """
        self.delay = delay

    def predict(self, input_sequence: list) -> list:
        """Predict expected output sequence.

        Args:
            input_sequence: List of integer input values.

        Returns:
            List of expected output values (same values, same order).
        """
        return list(input_sequence)
