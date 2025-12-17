from torchbit.tools.bit_struct import BitField

def compare_bit_struct(expect, real):
    """
    Compares two BitStruct objects field by field.

    Args:
        expect: The expected BitStruct object.
        real: The actual BitStruct object.

    Returns:
        A list of dictionaries, where each dictionary represents a mismatch
        and contains 'name', 'width', 'expect', and 'real' keys.
        Returns an empty list if there are no mismatches.
        
    Raises:
        TypeError: If the two BitStruct objects are not from the same factory.
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
