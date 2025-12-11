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

    async def run(self, stop_event: Event=None):
        idx = 0
        while idx < len(self.queue):
            if stop_event is not None and isinstance(stop_event, Event):
                if stop_event.is_set():
                    break

            # Drive signals
            self.data_port.set(self.queue[idx])
            self.valid_port.set(1) # Always Ready logic
            
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


class PoolCollector:
    def __init__(self, debug=False):
        self.debug = debug
        self.data = []
        self.timestamps = []

    def connect(self, dut, clk, data, valid):
        self.dut = dut
        self.clk = clk
        self.data_port = InputPort(data)
        self.valid_port = InputPort(valid)


    async def run(self, stop_event: Event):
        # Collector now implicitly assumes it's always ready to receive, as 'ready' output was removed.

        while not stop_event.is_set():
            await RisingEdge(self.clk)
            
            # Check valid
            # Receive Logic
            v_val = self.valid_port.get()
            is_valid = (v_val == 1) # Hardcoded to check for active-high valid
            
            if is_valid:
                val = self.data_port.get()
                t = get_sim_time()
                self.data.append(val)
                self.timestamps.append(t)
                if self.debug:
                    self.dut._log.info(f"[Collector] Collected {val} at {t}")

    def dump(self):
        return self.data

    def dump_time(self):
        return self.timestamps


class FIFOCollector:
    def __init__(self, debug=False):
        self.debug = debug
        print(f"[FIFOCollector] Initialized {debug=}")
        self.data = []
        self.timestamps = []

    def connect(self, dut, clk, data, empty, ready, valid=None):
        self.dut = dut
        self.clk = clk
        self.data_port = InputPort(data)

        self.empty_port = InputPort(empty) 
        self.ready_port = OutputPort(ready) 
        self.valid_port = InputPort(valid) if valid is not None else None

    async def run(self, stop_event: Event):
        # Collector now implicitly assumes it's always ready to receive, as 'ready' output was removed.
        self.ready_port.set(0)

        last_ready = 0
        cur_ready = 0
        while not stop_event.is_set():
            await RisingEdge(self.clk)
            t = get_sim_time()

            empty = self.empty_port.get()
            if not empty:
                self.ready_port.set(1)
                cur_ready = 1
            else:
                self.ready_port.set(0)
                cur_ready = 0
            
            # Check valid
            # Receive Logic
            if self.valid_port is not None:
                v_val = self.valid_port.get()
            else:
                v_val = last_ready
            if self.debug:
                self.dut._log.info(f"[Collector] Valid {v_val} at {t}")
            

            is_valid = (v_val == 1) # Hardcoded to check for active-high valid
            
            if is_valid:
                val = self.data_port.get()
                self.data.append(val)
                self.timestamps.append(t)
                if self.debug:
                    self.dut._log.info(f"[Collector] Collected {val} at {t}")
            
            last_ready = cur_ready


        # Cleanup: No ready_port to set to 0.

    def dump(self):
        return self.data

    def dump_time(self):
        return self.timestamps
