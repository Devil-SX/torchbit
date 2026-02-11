"""Tests for ComponentDB and from_path integration (pure Python, no cocotb)."""
import pytest
from torchbit.tools.component_db import ComponentDB
from torchbit.tools.fifo import FIFODriver, FIFOReceiver


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockSignal:
    """Minimal signal mock supporting .value read/write."""

    def __init__(self, value=0):
        self.value = value


class MockDUT:
    """Minimal DUT mock with input and output FIFO signals."""

    def __init__(self):
        self.clk = MockSignal()
        self.din = MockSignal()
        self.din_valid = MockSignal()
        self.din_ready = MockSignal()
        self.dout = MockSignal()
        self.dout_valid = MockSignal()
        self.dout_ready = MockSignal()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_db():
    """Ensure ComponentDB state is clean before and after every test."""
    ComponentDB.clear()
    yield
    ComponentDB.clear()


# ---------------------------------------------------------------------------
# ComponentDB core tests
# ---------------------------------------------------------------------------

class TestComponentDB:

    def test_set_get(self):
        """Register and retrieve signals at a path."""
        signals = {"data": "sig_data", "valid": "sig_valid"}
        ComponentDB.set("top.encoder.input", signals)
        result = ComponentDB.get("top.encoder.input")
        assert result is signals
        assert result["data"] == "sig_data"
        assert result["valid"] == "sig_valid"

    def test_get_missing_path(self):
        """KeyError when querying a path that has not been registered."""
        with pytest.raises(KeyError, match="No signals registered at path 'nonexistent'"):
            ComponentDB.get("nonexistent")

    def test_clear(self):
        """clear() resets both registry and component list."""
        ComponentDB.set("a.b", {"x": 1})

        class Dummy:
            tag = "t"

        ComponentDB.register_component(Dummy())
        assert len(ComponentDB._registry) == 1
        assert len(ComponentDB._components) == 1

        ComponentDB.clear()
        assert len(ComponentDB._registry) == 0
        assert len(ComponentDB._components) == 0

    def test_find_by_tag(self):
        """register_component + find_by_tag returns correct components."""

        class C:
            def __init__(self, tag):
                self.tag = tag

        c1 = C("encoder")
        c2 = C("decoder")
        c3 = C("encoder")
        ComponentDB.register_component(c1)
        ComponentDB.register_component(c2)
        ComponentDB.register_component(c3)

        found = ComponentDB.find_by_tag("encoder")
        assert found == [c1, c3]

        found_dec = ComponentDB.find_by_tag("decoder")
        assert found_dec == [c2]

        found_none = ComponentDB.find_by_tag("missing")
        assert found_none == []

    def test_find_by_path_wildcard(self):
        """find_by_path supports glob-style * wildcards."""
        ComponentDB.set("top.encoder.input", {"a": 1})
        ComponentDB.set("top.encoder.output", {"b": 2})
        ComponentDB.set("top.decoder.input", {"c": 3})

        result = ComponentDB.find_by_path("top.encoder.*")
        assert set(result.keys()) == {"top.encoder.input", "top.encoder.output"}

        result_all = ComponentDB.find_by_path("top.*.*")
        assert len(result_all) == 3

        result_input = ComponentDB.find_by_path("*.input")
        assert set(result_input.keys()) == {"top.encoder.input", "top.decoder.input"}

    def test_overwrite_path(self):
        """Setting the same path twice overwrites the previous value."""
        ComponentDB.set("top.a", {"x": 1})
        ComponentDB.set("top.a", {"x": 2})
        assert ComponentDB.get("top.a") == {"x": 2}


# ---------------------------------------------------------------------------
# from_path integration tests
# ---------------------------------------------------------------------------

class TestFromPath:

    def test_from_path_fifo_driver(self):
        """FIFODriver.from_path creates a connected driver with correct tag."""
        dut = MockDUT()
        ComponentDB.set("top.encoder.input", {
            "data": dut.din,
            "valid": dut.din_valid,
            "ready": dut.din_ready,
        })

        driver = FIFODriver.from_path("top.encoder.input", dut, dut.clk)

        assert driver.tag == "input"
        assert driver.dut is dut
        assert driver.clk is dut.clk
        assert driver.data_port.wrapper is dut.din
        assert driver.valid_port.wrapper is dut.din_valid
        assert driver.ready_port.wrapper is dut.din_ready
        assert driver.active_high is True

        # Component should be registered in ComponentDB
        found = ComponentDB.find_by_tag("input")
        assert driver in found

    def test_from_path_fifo_driver_no_ready(self):
        """FIFODriver.from_path works when ready is not provided."""
        dut = MockDUT()
        ComponentDB.set("top.noc.tx", {
            "data": dut.din,
            "valid": dut.din_valid,
            # no "ready" key
        })

        driver = FIFODriver.from_path("top.noc.tx", dut, dut.clk)
        assert driver.ready_port is None
        assert driver.tag == "tx"

    def test_from_path_fifo_driver_custom_strategy(self):
        """FIFODriver.from_path passes strategy and debug through."""
        from torchbit.tools.strategy import RandomBackpressure

        dut = MockDUT()
        ComponentDB.set("top.x", {
            "data": dut.din,
            "valid": dut.din_valid,
        })

        strat = RandomBackpressure(stall_prob=0.5, seed=42)
        driver = FIFODriver.from_path("top.x", dut, dut.clk,
                                       strategy=strat, debug=True,
                                       active_high=False)
        assert driver.strategy is strat
        assert driver.debug is True
        assert driver.active_high is False

    def test_from_path_fifo_receiver(self):
        """FIFOReceiver.from_path creates a connected receiver with correct tag."""
        dut = MockDUT()
        ComponentDB.set("top.decoder.output", {
            "data": dut.dout,
            "valid": dut.dout_valid,
            "ready": dut.dout_ready,
        })

        receiver = FIFOReceiver.from_path("top.decoder.output", dut, dut.clk)

        assert receiver.tag == "output"
        assert receiver.dut is dut
        assert receiver.clk is dut.clk
        assert receiver.data_port.wrapper is dut.dout
        assert receiver.valid_port.wrapper is dut.dout_valid
        assert receiver.ready_port.wrapper is dut.dout_ready

        # Component should be registered in ComponentDB
        found = ComponentDB.find_by_tag("output")
        assert receiver in found

    def test_from_path_fifo_receiver_missing_path(self):
        """FIFOReceiver.from_path raises KeyError for unregistered path."""
        dut = MockDUT()
        with pytest.raises(KeyError, match="No signals registered"):
            FIFOReceiver.from_path("top.missing", dut, dut.clk)

    def test_from_path_fifo_driver_missing_path(self):
        """FIFODriver.from_path raises KeyError for unregistered path."""
        dut = MockDUT()
        with pytest.raises(KeyError, match="No signals registered"):
            FIFODriver.from_path("top.missing", dut, dut.clk)


# ---------------------------------------------------------------------------
# Tag attribute on regular construction
# ---------------------------------------------------------------------------

class TestTagAttribute:

    def test_fifo_driver_default_tag(self):
        """FIFODriver.__init__ sets tag to None."""
        d = FIFODriver()
        assert d.tag is None

    def test_fifo_receiver_default_tag(self):
        """FIFOReceiver.__init__ sets tag to None."""
        r = FIFOReceiver()
        assert r.tag is None


# ---------------------------------------------------------------------------
# Import from torchbit.tools
# ---------------------------------------------------------------------------

class TestComponentDBImport:

    def test_import_from_tools(self):
        """ComponentDB is importable from torchbit.tools."""
        from torchbit.tools import ComponentDB as CDB
        assert CDB is ComponentDB
