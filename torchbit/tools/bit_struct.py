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

    def inspect(self):
        print(f"BitStruct Layout (Total Width: {self.total_width} bits, LSB First: {self.lsb_first})")
        print("-" * 60)
        print(f"{'Name':<15} {'Width':<8} {'Range':<15} {'Value':<10}")
        print("-" * 60)
        
        shift = 0
        fields_to_process = self.fields if self.lsb_first else list(reversed(self.fields))
        
        # Collect rows first to print in correct visual order (MSB at top usually makes sense for hardware, 
        # but list order or logical order is also fine. Let's strictly follow the processing order for range calculation)
        
        # For range calculation, we must iterate based on lsb_first logic
        rows = []
        for field in fields_to_process:
            low = shift
            high = shift + field.width - 1
            range_str = f"[{high}:{low}]"
            value_str = hex(field.value)
            rows.append((field.name, field.width, range_str, value_str))
            shift += field.width
            
        # If MSB first (big endian-ish visualization), usually we want to see MSB at top.
        # If LSB first, often we list index 0 first. 
        # Let's just print in the order fields are defined in 'fields_to_process' which effectively
        # lists LSB fields first if lsb_first=True, or MSB fields first if lsb_first=False.
        # Wait, if lsb_first=False, the first field in the list is the MSB.
        # So 'fields_to_process' (reversed(self.fields)) will start with the LSB field?
        # No.
        # If lsb_first=True: fields=[a(LSB), b, c(MSB)]. shift starts at 0 for 'a'. a is [width-1:0].
        # If lsb_first=False: fields=[a(MSB), b, c(LSB)]. 
        # In to_int logic: 
        #   fields_to_process = reversed(fields) -> [c, b, a]. 
        #   loop c: shift=0. c is at [width-1:0].
        #   loop b: shift=c.width.
        # So for lsb_first=False, 'c' is the LSB.
        
        # To print in a "natural" order (usually definition order), we need to calculate ranges first.
        # Let's calculate ranges for all fields first using the same logic as to_int/from_int.
        
        field_ranges = {}
        current_shift = 0
        processing_order = self.fields if self.lsb_first else reversed(self.fields)
        
        for field in processing_order:
            low = current_shift
            high = current_shift + field.width - 1
            field_ranges[field.name] = (high, low)
            current_shift += field.width
            
        # Now print in the order of self.fields (definition order)
        for field in self.fields:
            high, low = field_ranges[field.name]
            range_str = f"[{high}:{low}]"
            value_str = hex(field.value)
            print(f"{field.name:<15} {field.width:<8} {range_str:<15} {value_str:<10}")
        print("-" * 60)