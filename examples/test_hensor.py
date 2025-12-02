import torchbit as torchbit
from pathlib import Path
import torch


if __name__ == "__main__":

    def test_tensor2memhex_and_memhex2tensor(tmp_path):
        print("Testing tensor2memhex and memhex2tensor...")
        dtypes = torchbit.core.dtype_to_bits.keys()

        # Create the directory if it does not exist
        tmp_path.mkdir(parents=True, exist_ok=True)

        for dtype in dtypes:
            # Create a 2D tensor with the given dtype
            tensor = torch.randint(0, 10, (20, 4), dtype=dtype)

            # Define the output path for the memhex file
            out_path = tmp_path / f"tensor_{dtype}.hex"

            # Convert the tensor to a memhex file
            torchbit.core.Vector.from_tensor(tensor).to_memhex(out_path)

            # Convert the memhex file back to a tensor
            converted_tensor = torchbit.core.Vector.from_memhex(out_path, dtype).tensor

            # Check if the original tensor and the converted tensor are equal
            if torch.equal(tensor, converted_tensor):
                print(f"Test passed for dtype: {dtype}")
            else:

                print(f"Test failed for dtype: {dtype}")

    def test_tensor2cocotb_and_cocotb2tensor():
        print("Testing tensor2cocotb and cocotb2tensor...")
        dtypes = torchbit.core.dtype_to_bits.keys()

        for dtype in dtypes:
            # Create a tensor with the given dtype
            tensor = torch.randint(0, 10, (20,), dtype=dtype)

            # Convert the tensor to a cocotb value
            # value = torchbit.core.Vector.from_tensor(tensor).to_cocotb()
            value = torchbit.core.tensor_to_cocotb(tensor)

            # Convert the cocotb value back to a tensor
            # converted_tensor = torchbit.core.Vector.from_cocotb(value, 20, dtype)
            converted_tensor = torchbit.core.cocotb_to_tensor(value, 20, dtype)


            # Check if the original tensor and the converted tensor are equal
            if torch.equal(tensor, converted_tensor):
                print(f"Test passed for dtype: {dtype}")
            else:
                print(f"Test failed for dtype: {dtype}")

    # test_tensor2memhex_and_memhex2tensor(Path("test"))
    test_tensor2cocotb_and_cocotb2tensor()
