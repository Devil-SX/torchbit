import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Event
from torchbit.tools import Sender, Collector
from torchbit.runner import Runner, FileConfig, BuildConfig
import random
from pathlib import Path
import torchbit.debug.temporal_event as temporal_event

def random_seq_gen(length=10, min_val=0, max_val=100):
    return [random.randint(min_val, max_val) for _ in range(length)]

class PipeWrapper:
    def __init__(self, dut):
        self.dut = dut
        self.sender = Sender(debug=True)
        self.collector = Collector(debug=True)

    def connect(self):
        # Sender connects to Input of Pipe
        # Pipe has no backpressure/ready, so pass None for full
        self.sender.connect(self.dut, self.dut.clk, self.dut.din, self.dut.din_vld, full=None)
        
        # Collector connects to Output of Pipe
        # Pipe output is valid/data. No ready port is passed to Collector as its interface simplified
        self.collector.connect(self.dut, self.dut.clk, self.dut.dout, self.dut.dout_vld)

file_config = FileConfig(
    name="pipe_test",
    sources=["examples/test_sender_collector/Pipe.sv"],
    top_design="Pipe",
    includes=[]
)
build_config = BuildConfig(
    name="default",
    trace=True
)

PIPE_RUNNER = Runner(file_config, build_config, current_dir=Path(__file__).parent)

@cocotb.test()
async def run_test(dut):
    # Clock
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # Reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    wrapper = PipeWrapper(dut)
    wrapper.connect()

    # Data
    seq = random_seq_gen(length=20)
    wrapper.sender.load(seq)

    # Run
    stop_event = Event()
    collector_task = cocotb.start_soon(wrapper.collector.run(stop_event))
    
    await wrapper.sender.run()
    
    # Wait for pipeline delay (approx 4 cycles + margin)
    for _ in range(10):
        await RisingEdge(dut.clk)
        
    stop_event.set()
    await collector_task

    # Check
    collected_data = wrapper.collector.dump()
    assert collected_data == seq, f"Mismatch! Sent: {seq}, Got: {collected_data}"
    
    # Dump temporal event graph
    sender_times = wrapper.sender.dump_time()
    collector_times = wrapper.collector.dump_time()
    
    # Convert to int for plotting if they are floats, or keep as is
    # Assuming get_sim_time returns ps or ns ticks as numbers
    
    graph_path = Path("sim_pipe_test_default") / "pipe_timing.png"
    # Ensure directory exists (runner creates it, but just in case)
    if not graph_path.parent.exists():
         graph_path.parent.mkdir(parents=True, exist_ok=True)

    temporal_event.draw_temporal_event_seqs(
        path=graph_path,
        names=["Sender Valid", "Collector Valid"],
        seqs=[sender_times, collector_times],
        unit="sim_steps",
        title="Pipe Sender/Collector Timing"
    )
    
    dut._log.info(f"Test Passed! Timing graph saved to {graph_path}")


if __name__ == "__main__":
    PIPE_RUNNER.test("pipe_test")
