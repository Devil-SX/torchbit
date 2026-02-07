"""
Driver/Monitor Example with Pipeline DUT

Demonstrates the Driver and PoolMonitor classes for driving
and capturing data from a hardware pipeline.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Event
from torchbit.tools import Driver, PoolMonitor
from torchbit.runner import Runner, FileConfig, BuildConfig, DEFAULT_VERILATOR_BUILD_CONFIG
import random
from pathlib import Path
import torchbit.debug.temporal_event as temporal_event


def random_seq_gen(length=10, min_val=0, max_val=100):
    """Generate a random sequence of integers."""
    return [random.randint(min_val, max_val) for _ in range(length)]


class PipeWrapper:
    """Wrapper class connecting Driver and Monitor to the Pipe DUT."""

    def __init__(self, dut, debug=False):
        self.dut = dut
        self.driver = Driver(debug=debug)
        self.monitor = PoolMonitor(debug=debug)

    def connect(self):
        """Connect Driver to DUT input and Monitor to DUT output."""
        # Driver connects to Input of Pipe
        # Pipe has no backpressure/ready, so pass None for full
        self.driver.connect(
            self.dut,
            self.dut.clk,
            self.dut.din,
            self.dut.din_vld,
            full=None
        )

        # PoolMonitor connects to Output of Pipe
        # Pipe output is valid/data
        self.monitor.connect(
            self.dut,
            self.dut.clk,
            self.dut.dout,
            self.dut.dout_vld
        )


# File configuration for the Pipe DUT
file_config = FileConfig(
    name="pipe",
    sources=["dut/pipe.sv"],
    top_design="Pipe",
    includes=[]
)

# Build configuration
build_config = DEFAULT_VERILATOR_BUILD_CONFIG

# Create runner instance
PIPE_RUNNER = Runner(file_config, build_config, current_dir=Path(__file__).parent)


@cocotb.test()
async def test_pipe_basic(dut):
    """Test basic Driver/Monitor functionality with pipeline."""
    # Create clock
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # Reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Create wrapper and connect
    wrapper = PipeWrapper(dut, debug=True)
    wrapper.connect()

    # Generate random test data
    seq = random_seq_gen(length=20)
    dut._log.info(f"[Test] Generated test sequence: {seq}")

    wrapper.driver.load(seq)

    # Run monitor and driver
    stop_event = Event()
    monitor_task = cocotb.start_soon(wrapper.monitor.run(stop_event))

    await wrapper.driver.run()

    # Wait for pipeline delay (approx 4 cycles + margin)
    for _ in range(10):
        await RisingEdge(dut.clk)

    stop_event.set()
    await monitor_task

    # Verify results
    collected_data = wrapper.monitor.dump()
    dut._log.info(f"[Test] Sent: {seq}")
    dut._log.info(f"[Test] Received: {collected_data}")

    assert collected_data == seq, f"Mismatch! Sent: {seq}, Got: {collected_data}"
    dut._log.info("[Test] Data verification PASSED!")

    # Generate timing visualization
    driver_times = wrapper.driver.dump_time()
    monitor_times = wrapper.monitor.dump_time()

    graph_path = Path("sim_pipe_default_verilator_test_pipe") / "pipe_timing.png"

    temporal_event.draw_temporal_event_seqs(
        path=graph_path,
        names=["Driver Valid", "PoolMonitor Valid"],
        seqs=[driver_times, monitor_times],
        unit="sim_steps",
        title="Pipe Driver/PoolMonitor Timing"
    )

    dut._log.info(f"[Test] Timing graph saved to {graph_path}")


@cocotb.test()
async def test_pipe_single_value(dut):
    """Test sending a single value through the pipeline."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # Reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    wrapper = PipeWrapper(dut, debug=False)
    wrapper.connect()

    # Single value
    seq = [42]
    wrapper.driver.load(seq)

    stop_event = Event()
    monitor_task = cocotb.start_soon(wrapper.monitor.run(stop_event))

    await wrapper.driver.run()

    # Wait for pipeline delay
    for _ in range(10):
        await RisingEdge(dut.clk)

    stop_event.set()
    await monitor_task

    collected_data = wrapper.monitor.dump()
    assert collected_data == seq, f"Single value test failed! Expected {seq}, got {collected_data}"
    dut._log.info("[Test] Single value test PASSED!")


if __name__ == "__main__":
    PIPE_RUNNER.test("test_pipe")
