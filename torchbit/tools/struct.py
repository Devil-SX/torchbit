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
    

class Struct:
    def __init__(self, fields:list[BitField]):
        self.fields = fields
        # lsb to msb
        self.total_width = sum(field.width for field in fields)


    def from_int(self, value:int):
        shift = 0
        for field in self.fields:
            mask = (1 << field.width) - 1
            field_value = (value >> shift) & mask
            field.set_value(field_value)
            shift += field.width

    def to_int(self) -> int:
        result = 0
        shift = 0
        for field in self.fields:
            result |= (field.value & ((1 << field.width) - 1)) << shift
            shift += field.width
        return result