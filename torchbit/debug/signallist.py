"""
Waveform signal list utilities for GTKWave integration.

Provides Signal and SignalGroup dataclasses for organizing HDL signals,
and a function to generate GTKWave TCL scripts for waveform visualization.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Signal:
    """Represents a signal for waveform viewing.

    Attributes:
        name (str): Short signal name for display in the waveform viewer.
        full_path (str): Full hierarchical path in the design
            (e.g., "top.dut.module.signal_name").
    """
    name: str
    full_path: str


@dataclass
class SignalGroup:
    """Groups related signals for waveform display.

    Groups signals together in the waveform viewer with a common label.
    Each group can be assigned a distinct color.

    Attributes:
        name (str): Group name displayed as a comment in the waveform.
        signals (list[Signal]): List of signals in this group.
    """
    name: str
    signals: List[Signal]


def generate_gtkwave_tcl(waveform_path: str, signal_groups: List[SignalGroup], output_tcl_path: str) -> None:
    """Generate a GTKWave TCL script for waveform visualization.

    Creates a TCL script that loads a waveform file (VCD/FST) and
    arranges signals into groups with auto-assigned colors.

    Args:
        waveform_path: Path to the waveform file (vcd/fst format).
        signal_groups: List of SignalGroup objects defining which signals
            to display and how to group them.
        output_tcl_path: Path for the generated TCL script.

    Example:
        >>> from torchbit.debug import Signal, SignalGroup
        >>>
        >>> # Define signals for a DUT interface
        >>> ctrl_group = SignalGroup(name="Control", signals=[
        ...     Signal("clk", "top.dut.clk"),
        ...     Signal("rst", "top.dut.rst"),
        ...     Signal("valid", "top.dut.valid"),
        ... ])
        >>>
        >>> data_group = SignalGroup(name="Data", signals=[
        ...     Signal("data_in", "top.dut.data_in"),
        ...     Signal("data_out", "top.dut.data_out"),
        ... ])
        >>>
        >>> # Generate TCL script
        >>> generate_gtkwave_tcl(
        ...     waveform_path="dump.fst",
        ...     signal_groups=[ctrl_group, data_group],
        ...     output_tcl_path="view.tcl"
        ... )
        >>>
        >>> # Then run GTKWave:
        >>> # gtkwave -o view.tcl

    Notes:
        - Generated TCL script uses GTKWave's tcl command interface
        - Colors are auto-assigned from a predefined palette
        - Script includes zoom-to-full at the end for overview
        - Signal hierarchy paths must match the waveform file contents
    """
    # GTKWave Color IDs (from GTKWave documentation)
    # 0:Cycle (default), 1:Red, 2:Orange, 3:Yellow, 4:Green, 5:Blue, 6:Indigo, 7:Violet, 8:Grey, 9:Black
    # Cycle through a subset of distinct colors
    gtkwave_colors = ["Red", "Orange", "Green", "Blue", "Violet", "Yellow", "Indigo", "Grey"]  # Exclude Black for visibility

    lines = []

    lines.append(f'gtkwave::loadFile "{waveform_path}"')

    for i, group in enumerate(signal_groups):
        lines.append(f'gtkwave::/Edit/Insert_Comment "--- {group.name} ---"')

        # Auto-assign color from the cycle
        assigned_color_name = gtkwave_colors[i % len(gtkwave_colors)]

        for signal in group.signals:
            lines.append(f'gtkwave::addSignalsFromList "{signal.full_path}"')

            # Highlight/Coloring for the last added signal
            # This requires selecting the signal and then applying the color.
            # GTKWave TCL 'gtkwave::/Edit/Color_Format/<ColorName>' applies to selected traces.
            # To apply to a specific signal after adding it:
            lines.append(f'gtkwave::highlightSignalsFromList "{signal.full_path}"')
            lines.append(f'gtkwave::/Edit/Color_Format/{assigned_color_name}')
            lines.append(f'gtkwave::unhighlightSignalsFromList "{signal.full_path}"')  # Unhighlight to prepare for next signal or group

    lines.append('gtkwave::/Time/Zoom/Zoom_Full')

    with open(output_tcl_path, "w") as f:
        f.write("\n".join(lines))
