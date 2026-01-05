"""
Test example for torchbit wavetool.

This script:
1. Uses pyvcd to generate a test VCD file
2. Uses torchbit.debug.wave_converter to convert VCD to CSV text format
"""

import vcd
from pathlib import Path
import tempfile
import os

from torchbit.debug.wave_converter import convert_wave_to_text


def generate_test_vcd(output_path: str = "test_waveform.vcd"):
    """
    Generate a test VCD file using pyvcd.

    Creates a simple waveform with:
    - clk: Clock signal (toggle every 10ns)
    - counter: 8-bit counter incrementing at each posedge
    - valid: 1-bit valid signal (asserted every 3 cycles)
    - data: 16-bit data bus (data = counter * 2 when valid)

    Args:
        output_path: Path where VCD file will be written
    """
    print(f"Generating test VCD file: {output_path}")

    # Create VCD writer
    with open(output_path, 'w') as f:
        writer = vcd.VCDWriter(
            f,
            timescale="1ns",
            comment="Test waveform for torchbit wavetool",
            date="today"
        )

        # Register signals - returns Variable objects
        clk_sig = writer.register_var("top", "clk", "wire", size=1)
        counter_sig = writer.register_var("top", "counter", "wire", size=8)
        valid_sig = writer.register_var("top", "valid", "wire", size=1)
        data_sig = writer.register_var("top", "data", "wire", size=16)

        # Generate 20 clock cycles
        counter_value = 0

        for cycle in range(20):
            # Clock toggle (10ns period, 5ns high/low)
            # negedge at cycle*10, posedge at cycle*10 + 5
            writer.change(clk_sig, cycle * 10, 0)

            # Update counter and data at negedge (setup before posedge)
            if cycle > 0:
                counter_value = (counter_value + 1) % 256

            # Update valid and data
            valid_value = 1 if (counter_value % 3) == 0 else 0
            data_value = counter_value * 2

            writer.change(counter_sig, cycle * 10, counter_value)
            writer.change(valid_sig, cycle * 10, valid_value)
            writer.change(data_sig, cycle * 10, data_value)

            # posedge at cycle*10 + 5
            writer.change(clk_sig, cycle * 10 + 5, 1)

        print(f"  Generated 20 clock cycles (200ns total)")

    return output_path


def test_waveform_conversion(vcd_path: str, csv_path: str = "test_waveform.csv"):
    """
    Test torchbit's waveform conversion.

    Converts the VCD file to CSV format with posedge sampling.

    Args:
        vcd_path: Path to input VCD file
        csv_path: Path to output CSV file
    """
    print(f"\nConverting VCD to CSV: {vcd_path} -> {csv_path}")

    # Convert using torchbit
    num_samples = convert_wave_to_text(
        wavefile_path=vcd_path,
        output_path=csv_path,
        clk="top.clk",
        format="csv_with_header",
        delimiter=","
    )

    print(f"  Extracted {num_samples} samples at posedge")

    # Display first few lines of output
    print(f"\nFirst 5 lines of output CSV:")
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {line.rstrip()}")

    return num_samples


def main():
    """Main test function."""
    print("=" * 60)
    print("Torchbit Wavetool Test")
    print("=" * 60)

    # Create temp directory for test files
    test_dir = Path("examples/test_waveform_output")
    test_dir.mkdir(exist_ok=True)

    vcd_path = test_dir / "test_waveform.vcd"
    csv_path = test_dir / "test_waveform.csv"

    # Generate test VCD
    generate_test_vcd(str(vcd_path))

    # Convert to CSV
    num_samples = test_waveform_conversion(str(vcd_path), str(csv_path))

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print(f"  VCD file: {vcd_path}")
    print(f"  CSV file: {csv_path}")
    print(f"  Samples:  {num_samples}")
    print("=" * 60)

    # Clean up (comment out if you want to keep the files)
    # import shutil
    # shutil.rmtree(test_dir)
    # print(f"\nCleaned up test directory: {test_dir}")


if __name__ == "__main__":
    main()
