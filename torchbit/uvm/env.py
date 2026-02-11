"""pyuvm-compatible environment template."""


class TorchbitEnv:
    """Documentation-level environment (works without pyuvm).

    Assembles agents + scoreboard into a verification environment.
    """

    def __init__(self, name: str = "env"):
        self.name = name
        self.agents = {}
        self.scoreboard = None

    def add_agent(self, name: str, agent):
        """Register an agent."""
        self.agents[name] = agent

    def set_scoreboard(self, scoreboard):
        """Set the scoreboard."""
        self.scoreboard = scoreboard


def create_uvm_env(agent_configs=None, name: str = "TorchbitUvmEnv"):
    """Factory that creates a uvm_env subclass.

    Args:
        agent_configs: List of dicts with keys:
            - name: agent instance name
            - driver_cls: optional custom driver class
            - monitor_cls: optional custom monitor class
        name: Class name for the generated env.

    Returns:
        A class inheriting from pyuvm.uvm_env.
    """
    try:
        from pyuvm import uvm_env
    except ImportError:
        raise ImportError(
            "torchbit.uvm requires pyuvm. Install with: pip install pyuvm>=4.0.0"
        )
    from .agent import create_uvm_agent
    from .scoreboard import create_uvm_scoreboard

    configs = agent_configs or [{"name": "agent"}]

    class _TorchbitUvmEnv(uvm_env):
        def build_phase(self):
            super().build_phase()
            self.agents = {}
            for cfg in configs:
                agent_cls = create_uvm_agent(
                    driver_cls=cfg.get("driver_cls"),
                    monitor_cls=cfg.get("monitor_cls"),
                )
                self.agents[cfg["name"]] = agent_cls.create(cfg["name"], self)

            scoreboard_cls = create_uvm_scoreboard()
            self.scoreboard = scoreboard_cls("scoreboard", self)

        def connect_phase(self):
            # Connect first agent's monitor to scoreboard actual
            for agent in self.agents.values():
                if hasattr(agent, "mon") and hasattr(agent.mon, "ap"):
                    agent.mon.ap.connect(
                        self.scoreboard.actual_fifo.analysis_export
                    )
                    break  # only connect first by default

    _TorchbitUvmEnv.__name__ = name
    _TorchbitUvmEnv.__qualname__ = name
    return _TorchbitUvmEnv
