import os
import torchbit.debug.temporal_event as temporal_event
from pathlib import Path
import random

def test_temporal_event_drawing():
    print("Testing temporal event drawing...")
    
    # Define output directory
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "temporal_event_test.png"

    # Sample data
    names = ["Producer Start", "FIFO Write", "FIFO Read", "Consumer End", "Error Flag"]
    seqs = []

    # Generate some random timestamps for demonstration
    max_time = 1000
    for i in range(len(names)):
        num_events = random.randint(5, 15)
        # Ensure timestamps are sorted
        current_seq = sorted([random.randint(0, max_time) for _ in range(num_events)])
        seqs.append(current_seq)

    # Add an empty sequence to test robustness
    names.append("Empty Sequence")
    seqs.append([])

    # Add a sequence with a single event
    names.append("Single Event")
    seqs.append([random.randint(0, max_time)])

    # Draw the plot
    temporal_event.draw_temporal_event_seqs(
        path=output_path,
        names=names,
        seqs=seqs,
        unit="ps",
        title="Sample Temporal Event Diagram"
    )

    # Verify if the file was created
    assert output_path.exists(), f"Test failed: Output file {output_path} was not created."
    print(f"Successfully created temporal event diagram at: {output_path}")
    print("Temporal event drawing test passed!")

if __name__ == "__main__":
    test_temporal_event_drawing()
