import cocotb
from cocotb.triggers import RisingEdge, Event, Timer
from cocotb.utils import get_sim_time
from .port import InputPort, OutputPort

class Sender:
    def __init__(self, debug=False):
        self.debug = debug
        self.queue = []
        self.timestamps = []

    def connect(self, dut, clk, data, valid, full=None):
        self.dut = dut
        self.clk = clk
        self.data_port = OutputPort(data)
        self.valid_port = OutputPort(valid)
        self.full_port = InputPort(full) if full is not None else None

    def load(self, sequence):
        self.queue = list(sequence)
        self.timestamps = []

    def dump_time(self):
        return self.timestamps

    async def run(self):
        idx = 0
        while idx < len(self.queue):
            # Drive signals
            self.data_port.set(self.queue[idx])
            self.valid_port.set(1)
            
            # Wait for clock edge
            await RisingEdge(self.clk)
            
            # Check flow control to decide if we advance
            can_send = True
            if self.full_port is not None:
                r_val = self.full_port.get()
                # If full signal: 1 means Stop
                # If normal READY: 0 means Stop
                can_send = (r_val == 0)
            
            if can_send:
                t = get_sim_time()
                self.timestamps.append(t)
                if self.debug:
                    self.dut._log.info(f"[Sender] Sent {self.queue[idx]} at {t}")
                idx += 1
        
        # Done, deassert valid
        self.valid_port.set(0)


class Collector:
    def __init__(self, debug=False):
        self.debug = debug
        self.data = []
        self.timestamps = []

    def connect(self, dut, clk, data, valid, ready=None, valid_is_empty=False):
        self.dut = dut
        self.clk = clk
        self.data_port = InputPort(data)
        self.valid_port = InputPort(valid)
        self.ready_port = OutputPort(ready) if ready is not None else None
        self.valid_is_empty = valid_is_empty

    async def run(self, stop_event: Event):
        # Assert ready if connected (Always ready/read logic)
        if self.ready_port is not None:
            self.ready_port.set(1)

        while not stop_event.is_set():
            await RisingEdge(self.clk)
            
            # Check valid/empty
            v_val = self.valid_port.get()
            is_valid = False
            
            if self.valid_is_empty:
                # If signal is EMPTY, 0 means Data Available (Valid)
                if v_val == 0: 
                    is_valid = True
            else:
                # Normal VALID, 1 means Data Available
                if v_val != 0: 
                    is_valid = True
            
            if is_valid:
                val = self.data_port.get()
                t = get_sim_time()
                self.data.append(val)
                self.timestamps.append(t)
                if self.debug:
                    self.dut._log.info(f"[Collector] Collected {val} at {t}")
        
        # Cleanup
        if self.ready_port is not None:
            self.ready_port.set(0)

    def dump(self):
        return self.data

    def dump_time(self):
        return self.timestamps
