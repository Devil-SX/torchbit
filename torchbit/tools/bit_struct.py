from dataclasses import dataclass

@dataclass
class BitField:
    name: str
    width: int
    
    def set_value(self, value:int):
        self._value = value & ((1 << self.width) - 1)

    @property
    def value(self):
        return self._value
    

class BitStruct:
    def __init__(self, fields:list[BitField], lsb_first: bool):
        self.fields = fields
        self.lsb_first = lsb_first
        # lsb to msb
        self.total_width = sum(field.width for field in fields)
        
        # Create a mapping for easy access
        self._field_map = {field.name: field for field in fields}
        
        # Verify unique names
        if len(self._field_map) != len(fields):
            raise ValueError("Duplicate field names are not allowed in BitStruct")

    def __getattr__(self, name):
        if name in self._field_map:
            return self._field_map[name]
        raise AttributeError(f"'BitStruct' object has no attribute '{name}'")

    def from_int(self, value:int):
        shift = 0
        fields_to_process = self.fields if self.lsb_first else reversed(self.fields)
        
        for field in fields_to_process:
            mask = (1 << field.width) - 1
            field_value = (value >> shift) & mask
            field.set_value(field_value)
            shift += field.width

    def to_int(self) -> int:
        result = 0
        shift = 0
        fields_to_process = self.fields if self.lsb_first else reversed(self.fields)

        for field in fields_to_process:
            result |= (field.value & ((1 << field.width) - 1)) << shift
            shift += field.width
        return result