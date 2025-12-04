from cocotb.runner import get_runner
from .config import *
from pathlib import Path

class Runner:
    def __init__(self, file_config:FileConfig, build_config:BuildConfig, current_dir:Path=Path(".")):
        self.file_config = file_config
        self.build_config = build_config
        self.sources = file_config.sources
        self.top_design = file_config.top_design
        self.includes = file_config.includes
        self.build_args = build_config.build_args
        self.backend = build_config.backend
        self.current_dir = current_dir

    def test(self, test_module:str):
        build_dir = self.current_dir / f"sim_{self.file_config.name}_{self.build_config.name}_{test_module}"
        runner = get_runner(self.backend)
        # Cocotb > 1.8 dump wave problem https://github.com/cocotb/cocotb/issues/3894#issuecomment-2249133550
        # Best practice: Cocotb = 1.8.1, Verilator = 5.024
        runner.build(
            verilog_sources=self.sources,
            hdl_toplevel=self.top_design,
            includes=self.includes,
            verbose=True,
            waves=True,
            build_args=self.build_args,
            build_dir=build_dir,
        )
        runner.test(hdl_toplevel=self.top_design, test_module=test_module, waves=True, verbose=True)


