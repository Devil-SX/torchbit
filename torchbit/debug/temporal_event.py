"""
Temporal event visualization utilities.

Provides functions for drawing and analyzing temporal event sequences,
useful for visualizing the timing relationship between sender/collector
events in verification tests.
"""
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any


def load_from_json(json_path: str | Path) -> Dict[str, Any]:
    """Load temporal event data from a JSON file.

    Parses a JSON file containing event timestamps and returns a dictionary
    suitable for passing to draw_temporal_event_seqs.

    The JSON structure is expected to be:
    {
        "unit": "ns",
        "title": "My Graph",  # Optional
        "events": [
            {"name": "Event A", "timestamps": [10, 20, 30]},
            {"name": "Event B", "timestamps": [15, 25, 35]}
        ]
    }

    Args:
        json_path: Path to the JSON file containing event data.

    Returns:
        Dictionary with the following keys:
        - names: List of event names (str)
        - seqs: List of timestamp sequences (list of lists)
        - unit: Time unit string (str, default: "ticks")
        - title: Graph title (str, default: "Temporal Event Graph")

    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        KeyError: If required keys are missing (will use defaults).

    Example:
        >>> data = load_from_json("events.json")
        >>> data["names"]
        ['Sender', 'Collector']
        >>> data["seqs"]
        [[0, 100, 200], [150, 250, 350]]
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    names = []
    seqs = []

    events = data.get("events", [])
    if not events:
        print(f"Warning: No 'events' list found in {json_path}")

    for event in events:
        names.append(event.get("name", "Unnamed"))
        seqs.append(event.get("timestamps", []))

    return {
        "names": names,
        "seqs": seqs,
        "unit": data.get("unit", "ticks"),
        "title": data.get("title", "Temporal Event Graph")
    }


def draw_from_json(json_path: str | Path, output_path: str | Path) -> None:
    """Load data from a JSON file and draw the temporal event graph.

    Convenience function that combines load_from_json and draw_temporal_event_seqs.

    Args:
        json_path: Path to the input JSON file.
        output_path: Path to save the output image file.
    """
    args = load_from_json(json_path)
    draw_temporal_event_seqs(output_path, **args)


def draw_temporal_event_seqs(path: str, names: List[str], seqs: List[List[float | int]], unit: str, title: str = "Temporal Event Graph") -> None:
    """Draw a temporal event chart.

    Creates a visualization where each sequence is plotted as a row of events.
    Events are colored based on their index (order) within the sequence to
    visualize correspondence between events across different sequences.

    Args:
        path: Output path for the image file.
        names: List of names for each sequence (y-axis labels).
        seqs: List of sequences, where each sequence contains timestamps.
        unit: Time unit string for the x-axis label (e.g., "ns", "us", "cycles").
        title: Title of the plot.

    Raises:
        ValueError: If lengths of names and seqs don't match.
        ValueError: If names is empty.

    Example:
        >>> # Draw from Python data
        >>> draw_temporal_event_seqs(
        ...     path="events.png",
        ...     names=["Sender", "Collector"],
        ...     seqs=[[0, 100, 200], [50, 150, 250]],
        ...     unit="ns",
        ...     title="Data Transfer Timing"
        ... )
        >>>
        >>> # Draw from JSON file
        >>> draw_from_json("events.json", "events.png")

    Output:
        Saves a PNG image showing:
        - X-axis: Time in specified units
        - Y-axis: Event sequence names
        - Colored dots/lines: Events at their respective timestamps
        - Grid lines for easy time reading
    """
    if len(names) != len(seqs):
        raise ValueError(f"Length mismatch: names({len(names)}) vs seqs({len(seqs)})")

    if not names:
        print("Warning: No data to plot.")
        return

    # Determine figure height based on number of sequences
    fig_height = max(5, len(names) * 0.6 + 2)

    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Plot events
    # lineoffsets determines the y-position of each sequence.
    # We use range(len(names)) to map them to 0, 1, 2...
    # We will invert the y-axis later so index 0 appears at the top.
    lineoffsets = np.arange(len(names))

    # Assign a unique color to each sequence (cycling through default palette)
    colors = [f"C{i % 10}" for i in range(len(names))]

    # eventplot is ideal for this: draws vertical lines at specific positions
    # We pass colors as a list matching the number of sequences
    ax.eventplot(seqs, lineoffsets=lineoffsets, linelengths=0.8, colors=colors)

    # Formatting
    # Map integer Y-ticks to names
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Sequence 0 at the top

    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Event Sequences")
    ax.set_title(title)

    # Add grid for easier time reading
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    # Ensure layout is neat
    plt.tight_layout()

    # Save to file
    plt.savefig(path)
    plt.close(fig)
