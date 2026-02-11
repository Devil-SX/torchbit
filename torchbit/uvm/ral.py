"""Register Abstraction Layer (RAL) integration with BitStruct."""
from ..core.bit_struct import BitStruct, BitField


class RegisterModel:
    """Register model generated from BitStruct definitions.

    Maps BitStruct field definitions to addressable registers,
    providing a clean API for register access in verification.

    Example:
        >>> fields = [BitField("opcode", 8), BitField("addr", 16), BitField("data", 32)]
        >>> struct_cls = BitStruct(fields, lsb_first=True)
        >>> model = RegisterModel("ctrl_reg", struct_cls)
        >>> model.write_field("opcode", 0x42)
        >>> assert model.read_field("opcode") == 0x42
        >>> packed = model.get_packed()
    """

    def __init__(self, name: str, bit_struct_cls, base_addr: int = 0):
        """
        Args:
            name: Register name.
            bit_struct_cls: BitStruct class (from BitStruct([fields], ...)).
            base_addr: Base address for this register.
        """
        self.name = name
        self.bit_struct_cls = bit_struct_cls
        self.base_addr = base_addr
        self._instance = bit_struct_cls()

    def write_field(self, field_name: str, value: int) -> None:
        """Write a value to a named field."""
        setattr(self._instance, field_name, value)

    def read_field(self, field_name: str) -> int:
        """Read the current value of a named field."""
        return getattr(self._instance, field_name).value

    def get_packed(self) -> int:
        """Get the fully packed register value."""
        return self._instance.to_logic()

    def set_packed(self, value: int) -> None:
        """Set register from a packed integer value."""
        self._instance.from_logic(value)

    def reset(self) -> None:
        """Reset all fields to zero."""
        self._instance = self.bit_struct_cls()

    @property
    def fields(self) -> list:
        """List field names."""
        return [f.name for f in self._instance.fields]


class RegisterBlock:
    """A block of registers at consecutive addresses.

    Example:
        >>> block = RegisterBlock("ctrl", base_addr=0x1000)
        >>> block.add_register("status", status_struct, offset=0)
        >>> block.add_register("config", config_struct, offset=4)
        >>> block.write("config", "enable", 1)
        >>> val = block.read("status", "busy")
    """

    def __init__(self, name: str, base_addr: int = 0):
        self.name = name
        self.base_addr = base_addr
        self.registers: dict = {}  # name -> RegisterModel

    def add_register(self, name: str, bit_struct_cls, offset: int = 0) -> RegisterModel:
        """Add a register to the block.

        Args:
            name: Register name.
            bit_struct_cls: BitStruct class for this register.
            offset: Address offset from base_addr.

        Returns:
            The created RegisterModel.
        """
        reg = RegisterModel(name, bit_struct_cls, self.base_addr + offset)
        self.registers[name] = reg
        return reg

    def write(self, reg_name: str, field_name: str, value: int) -> None:
        """Write to a field in a named register."""
        self.registers[reg_name].write_field(field_name, value)

    def read(self, reg_name: str, field_name: str) -> int:
        """Read a field from a named register."""
        return self.registers[reg_name].read_field(field_name)

    def get_register(self, name: str) -> RegisterModel:
        """Get a register by name."""
        return self.registers[name]

    def backdoor_write(self, buffer, reg_name: str) -> None:
        """Write register value to a Buffer at the register's address.

        Args:
            buffer: torchbit Buffer instance.
            reg_name: Register name to write.
        """
        reg = self.registers[reg_name]
        buffer.write(reg.base_addr, reg.get_packed())

    def backdoor_read(self, buffer, reg_name: str) -> None:
        """Read register value from a Buffer at the register's address.

        Args:
            buffer: torchbit Buffer instance.
            reg_name: Register name to update.
        """
        reg = self.registers[reg_name]
        reg.set_packed(buffer.read(reg.base_addr))
