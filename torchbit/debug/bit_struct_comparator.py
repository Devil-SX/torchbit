"""
BitStruct comparison utilities.

Provides the compare_bit_struct function for field-by-field comparison
of BitStruct instances, useful for verifying protocol headers and
instruction fields.
"""
from torchbit.tools.bit_struct import BitField


def compare_bit_struct(expect, real) -> list[dict]:
    """Compare two BitStruct objects field by field.

    Compares each field of two BitStruct instances and returns a list
    of mismatches. Useful for verifying protocol headers, instruction
    fields, or any structured bit-level data.

    Args:
        expect: The expected BitStruct object (golden reference).
        real: The actual BitStruct object (testbench output).

    Returns:
        A list of dictionaries, where each dictionary represents a mismatch
        and contains the following keys:
        - name: Field name (str)
        - width: Field width in bits (int)
        - expect: Expected value (int)
        - real: Actual value (int)

        Returns an empty list if there are no mismatches (all fields match).

    Raises:
        TypeError: If the two BitStruct objects are not from the same factory
                   (i.e., were created by different BitStruct() calls with
                   different field definitions).

    Example:
        >>> from torchbit.tools.bit_struct import BitField, BitStruct
        >>> from torchbit.debug import compare_bit_struct
        >>>
        >>> # Define fields
        >>> fields = [
        ...     BitField(name="opcode", width=7),
        ...     BitField(name="rd", width=5),
        ...     BitField(name="rs1", width=5),
        ...     BitField(name="funct3", width=3),
        ...     BitField(name="rs2", width=5),
        ...     BitField(name="funct7", width=7),
        ... ]
        >>> RType = BitStruct(fields)
        >>>
        >>> # Create expected and actual structs
        >>> expected = RType()
        >>> expected.from_int(0x33)  # ADD instruction
        >>>
        >>> actual = RType()
        >>> actual.from_int(0x33)    # Same
        >>> mismatches = compare_bit_struct(expected, actual)
        >>> # mismatches == [] (no differences)
        >>>
        >>> # Now with a difference
        >>> actual2 = RType()
        >>> actual2.from_int(0x23)   # Different instruction
        >>> mismatches = compare_bit_struct(expected, actual2)
        >>> # Returns list of fields that differ
    """
    if type(expect) is not type(real):
        raise TypeError("Input BitStructs are not instances of the same class (created by different factories).")

    mismatches = []

    # The `fields` attribute is a list of BitField objects that defines the struct.
    # We can iterate through this list to check each field's value on the instances.
    for field_definition in expect.fields:
        field_name = field_definition.name

        # Access the actual BitField instance on each struct to get its value
        expected_field = getattr(expect, field_name)
        real_field = getattr(real, field_name)

        if expected_field.value != real_field.value:
            mismatches.append({
                "name": field_name,
                "width": expected_field.width,
                "expect": expected_field.value,
                "real": real_field.value,
            })

    return mismatches
