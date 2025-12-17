import torchbit.tools as struct_ops
import torchbit.debug as debug_ops

def create_sample_bit_struct_factory():
    """Helper to create a consistent BitStruct factory."""
    return struct_ops.BitStruct(fields=[
        struct_ops.BitField("field_a", 4),
        struct_ops.BitField("field_b", 8),
        struct_ops.BitField("field_c", 2)
    ], lsb_first=True)

def create_another_bit_struct_factory():
    """Helper to create a different BitStruct factory."""
    return struct_ops.BitStruct(fields=[
        struct_ops.BitField("id", 16),
        struct_ops.BitField("status", 1)
    ], lsb_first=False)

def test_compare_bit_struct_no_mismatch():
    print("\nTesting compare_bit_struct: No Mismatch")
    MyStruct = create_sample_bit_struct_factory()

    struct1 = MyStruct()
    struct1.field_a = 0b1011
    struct1.field_b = 0b10101010
    struct1.field_c = 0b11

    struct2 = MyStruct()
    struct2.field_a = 0b1011
    struct2.field_b = 0b10101010
    struct2.field_c = 0b11

    mismatches = debug_ops.compare_bit_struct(struct1, struct2)
    assert len(mismatches) == 0, f"Expected no mismatches, but got {len(mismatches)}: {mismatches}"
    print("  Passed: No mismatches detected.")

def test_compare_bit_struct_with_mismatch():
    print("\nTesting compare_bit_struct: With Mismatches")
    MyStruct = create_sample_bit_struct_factory()

    expect_struct = MyStruct()
    expect_struct.field_a = 0b1011
    expect_struct.field_b = 0b10101010
    expect_struct.field_c = 0b11

    real_struct = MyStruct()
    real_struct.field_a = 0b0011 # Mismatch
    real_struct.field_b = 0b10101010 # Match
    real_struct.field_c = 0b01 # Mismatch

    mismatches = debug_ops.compare_bit_struct(expect_struct, real_struct)
    assert len(mismatches) == 2, f"Expected 2 mismatches, but got {len(mismatches)}: {mismatches}"

    expected_mismatches = [
        {"name": "field_a", "width": 4, "expect": 0b1011, "real": 0b0011},
        {"name": "field_c", "width": 2, "expect": 0b11, "real": 0b01},
    ]

    # Check if all expected mismatches are present
    for expected_mismatch in expected_mismatches:
        assert expected_mismatch in mismatches, f"Expected mismatch {expected_mismatch} not found in {mismatches}"
    
    print("  Passed: Correct mismatches detected.")

def test_compare_bit_struct_different_types():
    print("\nTesting compare_bit_struct: Different Types (expecting TypeError)")
    MyStruct1 = create_sample_bit_struct_factory()
    MyStruct2 = create_another_bit_struct_factory()

    struct1 = MyStruct1()
    struct2 = MyStruct2()

    try:
        debug_ops.compare_bit_struct(struct1, struct2)
        assert False, "Expected TypeError for different BitStruct types, but no error was raised."
    except TypeError as e:
        assert "not instances of the same class" in str(e), f"Unexpected TypeError message: {e}"
        print("  Passed: TypeError correctly raised for different BitStruct types.")
    
    # Also test with one non-BitStruct object
    try:
        debug_ops.compare_bit_struct(struct1, "not a bit struct")
        assert False, "Expected TypeError for non-BitStruct object, but no error was raised."
    except TypeError as e:
        assert "not instances of the same class" in str(e), f"Unexpected TypeError message: {e}"
        print("  Passed: TypeError correctly raised for non-BitStruct object.")

def test_compare_bit_struct_all_fields_mismatch():
    print("\nTesting compare_bit_struct: All Fields Mismatch")
    MyStruct = create_sample_bit_struct_factory()

    expect_struct = MyStruct()
    expect_struct.field_a = 0b0001
    expect_struct.field_b = 0b00000001
    expect_struct.field_c = 0b01

    real_struct = MyStruct()
    real_struct.field_a = 0b0010
    real_struct.field_b = 0b00000010
    real_struct.field_c = 0b10

    mismatches = debug_ops.compare_bit_struct(expect_struct, real_struct)
    assert len(mismatches) == 3, f"Expected 3 mismatches, but got {len(mismatches)}: {mismatches}"

    expected_mismatches = [
        {"name": "field_a", "width": 4, "expect": 0b0001, "real": 0b0010},
        {"name": "field_b", "width": 8, "expect": 0b00000001, "real": 0b00000010},
        {"name": "field_c", "width": 2, "expect": 0b01, "real": 0b10},
    ]

    for expected_mismatch in expected_mismatches:
        assert expected_mismatch in mismatches, f"Expected mismatch {expected_mismatch} not found in {mismatches}"
    
    print("  Passed: All fields mismatch correctly detected.")

if __name__ == "__main__":
    test_compare_bit_struct_no_mismatch()
    test_compare_bit_struct_with_mismatch()
    test_compare_bit_struct_all_fields_mismatch()
    test_compare_bit_struct_different_types()
