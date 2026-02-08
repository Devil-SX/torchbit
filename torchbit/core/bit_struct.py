"""
Bit-level struct manipulation utilities.

Provides the BitStruct factory function for creating structured views
of integer values. Fields can be accessed as attributes for convenient
reading and writing of individual bit fields.
"""
import copy
from dataclasses import dataclass, field
from typing import List


@dataclass
class BitField:
    """Represents a named field within a BitStruct.

    Attributes:
        name (str): Unique identifier for the field.
        width (int): Width in bits.
        _value (int): Current value stored in the field (internal).
    """
    name: str
    width: int
    _value: int = field(default=0, init=False)

    def set_value(self, value: int) -> None:
        """Set the field value with proper bit masking.

        Args:
            value: Integer value to store. Will be masked to field width.
        """
        self._value = value & ((1 << self.width) - 1)

    @property
    def value(self) -> int:
        """Get the current field value."""
        return self._value


def BitStruct(fields: List[BitField], lsb_first: bool = True):
    """Factory function to create a bit-level struct class.

    Creates a dynamically generated class for manipulating structured
    bit fields within an integer value. Fields can be accessed as
    attributes for convenient reading and writing.

    The struct layout is determined by the order of fields:
    - lsb_first=True: First field in list is at bits [0:width-1], next at [width:2*width-1], etc.
    - lsb_first=False: First field in list is at most significant bits.

    Args:
        fields: List of BitField objects defining the struct layout.
            Each field must have a unique name.
        lsb_first: If True, fields are packed from LSB to MSB.
                   If False, fields are packed from MSB to LSB.
                   Defaults to True.

    Returns:
        A new BitStruct class (not an instance). Call it to create instances.

    Raises:
        ValueError: If duplicate field names are provided.

    Example:
        >>> from torchbit.core.bit_struct import BitField, BitStruct
        >>>
        >>> # Define fields for RISC-V R-type instruction
        >>> fields = [
        ...     BitField(name="funct7", width=7),
        ...     BitField(name="rs2", width=5),
        ...     BitField(name="rs1", width=5),
        ...     BitField(name="funct3", width=3),
        ...     BitField(name="rd", width=5),
        ...     BitField(name="opcode", width=7),
        ... ]
        >>>
        >>> # Create struct class
        >>> RType = BitStruct(fields)
        >>>
        >>> # Use struct
        >>> inst = RType()
        >>> inst.opcode = 0x33  # R-type opcode
        >>> inst.rs1 = 0x10     # source register 1
        >>> inst.rs2 = 0x05     # source register 2
        >>> inst.rd = 0x07      # destination register
        >>> inst.funct3 = 0x0
        >>> inst.funct7 = 0x00
        >>>
        >>> # Get packed integer
        >>> value = inst.to_int()
        >>> hex(value)
        '0x37010'
        >>>
        >>> # Parse from integer
        >>> inst2 = RType()
        >>> inst2.from_int(0x37010)
        >>> hex(inst2.rs1)
        '0x10'

    Common use cases:
        - Parsing instruction sets (RISC-V, ARM, x86)
        - Protocol header field manipulation
        - Register map bit field access
        - Communication protocol framing
    """
    # Factory Mode
    class _BitStruct:
        def __init__(self):
            self.fields = [copy.deepcopy(f) for f in fields]
            self.lsb_first = lsb_first
            self.total_width = sum(f.width for f in self.fields)
            self._field_map = {f.name: f for f in self.fields}

            if len(self._field_map) != len(self.fields):
                raise ValueError("Duplicate field names are not allowed in BitStruct")

        def __getattr__(self, name: str):
            # Avoid infinite recursion during deepcopy by checking __dict__ directly
            if name.startswith('__') and name.endswith('__'):
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            if name in self._field_map:
                return self._field_map[name]
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def __setattr__(self, name: str, value) -> None:
            if '_field_map' in self.__dict__ and name in self._field_map:
                self._field_map[name].set_value(value)
            else:
                object.__setattr__(self, name, value)

        def __getstate__(self):
            # Return the state for pickling/deepcopy
            # We need to return all instance attributes
            return {
                'fields': self.fields,
                'lsb_first': self.lsb_first,
                'total_width': self.total_width,
                '_field_map': self._field_map
            }

        def __setstate__(self, state):
            # Restore the state during unpickling/deepcopy
            # Directly set __dict__ to avoid triggering __setattr__
            self.__dict__.update(state)

        def from_logic(self, value: int) -> None:
            """Parse an integer value into the struct fields.

            Args:
                value: Integer value to parse into field components.
            """
            shift = 0
            fields_to_process = self.fields if self.lsb_first else reversed(self.fields)

            for field in fields_to_process:
                mask = (1 << field.width) - 1
                field_value = (value >> shift) & mask
                field.set_value(field_value)
                shift += field.width

        def from_int(self, value: int) -> None:
            """Alias for from_logic(). Parse an integer into the struct fields."""
            self.from_logic(value)

        def from_cocotb(self, value: int) -> None:
            """Alias for from_logic(). Parse an integer into the struct fields."""
            self.from_logic(value)

        def to_logic(self) -> int:
            """Pack all field values into a single integer.

            Returns:
                Integer with all fields packed according to the layout.
            """
            result = 0
            shift = 0
            fields_to_process = self.fields if self.lsb_first else reversed(self.fields)

            for field in fields_to_process:
                result |= (field.value & ((1 << field.width) - 1)) << shift
                shift += field.width
            return result

        def to_int(self) -> int:
            """Alias for to_logic(). Pack all fields into a single integer."""
            return self.to_logic()

        def to_cocotb(self) -> int:
            """Alias for to_logic(). Pack all fields into a single integer."""
            return self.to_logic()

        def inspect(self) -> None:
            """Print a formatted table of all fields and their values.

            Useful for debugging and verification. Shows:
            - Field name
            - Field width in bits
            - Bit range [high:low]
            - Current value in hex
            """
            print(f"BitStruct Layout (Total Width: {self.total_width} bits, LSB First: {self.lsb_first})")
            print("-" * 60)
            print(f"{'Name':<15} {'Width':<8} {'Range':<15} {'Value':<10}")
            print("-" * 60)

            shift = 0
            fields_to_process = self.fields if self.lsb_first else reversed(self.fields)

            field_ranges = {}
            current_shift = 0
            for field in fields_to_process:
                low = current_shift
                high = current_shift + field.width - 1
                field_ranges[field.name] = (high, low)
                current_shift += field.width

            for field in self.fields:
                high, low = field_ranges[field.name]
                range_str = f"[{high}:{low}]"
                value_str = hex(field.value)
                print(f"{field.name:<15} {field.width:<8} {range_str:<15} {value_str:<10}")
            print("-" * 60)

    return _BitStruct
