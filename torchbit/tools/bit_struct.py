import copy
from dataclasses import dataclass, field

@dataclass
class BitField:
    name: str
    width: int
    _value: int = field(default=0, init=False)
    
    def set_value(self, value:int):
        self._value = value & ((1 << self.width) - 1)

    @property
    def value(self):
        return self._value
    

def BitStruct(fields: list[BitField], lsb_first: bool = True):
    # Factory Mode
    class _BitStruct:
        def __init__(self):
            self.fields = [copy.deepcopy(f) for f in fields]
            self.lsb_first = lsb_first
            self.total_width = sum(f.width for f in self.fields)
            self._field_map = {f.name: f for f in self.fields}
            
            if len(self._field_map) != len(self.fields):
                raise ValueError("Duplicate field names are not allowed in BitStruct")

        def __getattr__(self, name):
            if name in self._field_map:
                return self._field_map[name]
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def __setattr__(self, name, value):
            if '_field_map' in self.__dict__ and name in self._field_map:
                self._field_map[name].set_value(value)
            else:
                object.__setattr__(self, name, value)

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

        def inspect(self):
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