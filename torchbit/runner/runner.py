
from .config import *
from pathlib import Path
from cocotb_tools.runner import get_runner

def generate_vcs_dump_wave(path, top_module):
    with open(path, "w") as f:
        f.write(f"module dump_fsdb;\ninitial begin\n$fsdbDumpfile(\"dump.fsdb\");\n$fsdbDumpvars(0, {top_module});\n $fsdbDumpvars(\"+all\");\n end\nendmodule")


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

        if self.backend == "vcs":
            dump_wave_path = build_dir / "dump_wave.v"
            generate_vcs_dump_wave(dump_wave_path, self.top_design)
            self.sources.append(dump_wave_path)
            # self.top_design = "dump_fsdb"
            self.build_args += ["-top", "dump_fsdb"] # set multi-top modules

        
        runner = get_runner(self.backend)
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


