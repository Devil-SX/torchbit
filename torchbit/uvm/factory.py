"""Factory utilities for torchbit UVM components."""


class ComponentRegistry:
    """Registry for component type overrides (works without pyuvm).

    Allows registering alternate implementations that factory functions
    will use instead of defaults. Useful for test-specific customization.
    """

    _overrides: dict = {}

    @classmethod
    def set_override(cls, base_name: str, override_cls) -> None:
        """Register a type override.

        Args:
            base_name: Name of the base component (e.g., "driver", "monitor").
            override_cls: Replacement class to use.
        """
        cls._overrides[base_name] = override_cls

    @classmethod
    def get(cls, base_name: str, default=None):
        """Get the overridden class or the default.

        Args:
            base_name: Name to look up.
            default: Default class if no override registered.

        Returns:
            Override class or default.
        """
        return cls._overrides.get(base_name, default)

    @classmethod
    def clear(cls) -> None:
        """Clear all overrides."""
        cls._overrides.clear()

    @classmethod
    def has_override(cls, base_name: str) -> bool:
        """Check if an override is registered."""
        return base_name in cls._overrides
