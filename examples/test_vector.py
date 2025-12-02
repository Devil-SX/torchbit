import torchbit as torchbit
from pathlib import Path
import torch


if __name__ == "__main__":

    def test_tensor2cocotb_and_cocotb2tensor_1D():
        print("Testing tensor2cocotb and cocotb2tensor for 1D tensor...")
        dtypes = torchbit.core.dtype_to_bits.keys()

        for dtype in dtypes:
            # Create a 1D tensor with the given dtype
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

    # The original test_tensor2memhex_and_memhex2tensor was for 2D tensors, which is now handled by Matrix.
    # test_tensor2memhex_and_memhex2tensor(Path("test"))
    test_tensor2cocotb_and_cocotb2tensor_1D()
