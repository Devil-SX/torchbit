"""UVM-style hierarchical path and tag management for component reuse.

Provides a singleton-like registry (ComponentDB) that maps hierarchical
signal paths to signal dictionaries, inspired by UVM's ``uvm_config_db``.

Typical usage:
    1. Register signal mappings at the top level::

        ComponentDB.set("top.encoder.input", {
            "data": dut.din,
            "valid": dut.din_valid,
            "ready": dut.din_ready,
        })

    2. Components resolve their own signals by path::

        driver = FIFODriver.from_path("top.encoder.input", dut, dut.clk)

This module is pure Python and does **not** depend on cocotb.
"""
import fnmatch


class ComponentDB:
    """Hierarchical signal path registry for component reuse.

    Inspired by UVM's uvm_config_db.  Register signal mappings at the
    top level, then components resolve their own signals by path.

    All methods are classmethods operating on shared class-level state,
    giving singleton-like semantics without requiring instantiation.
    """

    _registry: dict = {}     # path -> dict of signal mappings
    _components: list = []   # all registered components

    @classmethod
    def set(cls, path: str, signals: dict) -> None:
        """Register signals at a hierarchical path.

        Args:
            path: Dot-separated hierarchical path (e.g. ``"top.encoder.input"``).
            signals: Dictionary mapping signal names to signal objects.
        """
        cls._registry[path] = signals

    @classmethod
    def get(cls, path: str) -> dict:
        """Retrieve signals for a path.

        Args:
            path: Dot-separated hierarchical path.

        Returns:
            Dictionary of signal mappings registered at *path*.

        Raises:
            KeyError: If no signals are registered at *path*.
        """
        if path not in cls._registry:
            raise KeyError(f"No signals registered at path '{path}'")
        return cls._registry[path]

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (paths and components)."""
        cls._registry.clear()
        cls._components.clear()

    @classmethod
    def register_component(cls, component) -> None:
        """Register a component for tag-based lookup.

        Args:
            component: Any object; typically has a ``tag`` attribute.
        """
        cls._components.append(component)

    @classmethod
    def find_by_tag(cls, tag: str) -> list:
        """Find all components with a given tag.

        Args:
            tag: Tag string to search for.

        Returns:
            List of components whose ``tag`` attribute equals *tag*.
        """
        return [c for c in cls._components if getattr(c, 'tag', None) == tag]

    @classmethod
    def find_by_path(cls, pattern: str) -> dict:
        """Find all paths matching a pattern (supports ``*`` wildcard).

        Args:
            pattern: Glob-style pattern (e.g. ``"top.encoder.*"``).

        Returns:
            Dictionary of matching ``{path: signals}`` entries.
        """
        return {p: v for p, v in cls._registry.items()
                if fnmatch.fnmatch(p, pattern)}


__all__ = ["ComponentDB"]
