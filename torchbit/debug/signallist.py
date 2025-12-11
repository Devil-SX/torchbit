from dataclasses import dataclass

@dataclass
class Signal:
    name: str
    full_path: str

@dataclass
class SignalGroup:
    name: str
    signals: list[Signal]

def generate_gtkwave_tcl(waveform_path: str, signal_groups: list[SignalGroup], output_tcl_path: str):
    """
    Generates a GTKWave TCL script to load a waveform and add grouped signals with auto-assigned colors.

    Args:
        waveform_path (str): Path to the waveform file (vcd/fst).
        signal_groups (list[SignalGroup]): List of SignalGroup objects to add.
        output_tcl_path (str): Path to save the generated TCL script.
    """
    
    # GTKWave Color IDs (from GTKWave documentation)
    # 0:Cycle (default), 1:Red, 2:Orange, 3:Yellow, 4:Green, 5:Blue, 6:Indigo, 7:Violet, 8:Grey, 9:Black
    # Cycle through a subset of distinct colors
    gtkwave_colors = ["Red", "Orange", "Green", "Blue", "Violet", "Yellow", "Indigo", "Grey"] # Exclude Black for visibility

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
            lines.append(f'gtkwave::unhighlightSignalsFromList "{signal.full_path}"') # Unhighlight to prepare for next signal or group

    lines.append('gtkwave::/Time/Zoom/Zoom_Full')
    
    with open(output_tcl_path, "w") as f:
        f.write("\n".join(lines))