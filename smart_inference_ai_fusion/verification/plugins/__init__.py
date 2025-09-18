"""Formal verification plugins."""

# Auto-discover and load plugins
import os
import importlib
import logging

logger = logging.getLogger(__name__)

# Auto-load all plugins in this directory
def _load_plugins():
    """Automatically load all plugin modules."""
    plugin_dir = os.path.dirname(__file__)
    for filename in os.listdir(plugin_dir):
        if filename.endswith('_plugin.py'):
            module_name = filename[:-3]  # Remove .py
            try:
                importlib.import_module(f'.{module_name}', package=__name__)
                logger.debug(f"Loaded plugin: {module_name}")
            except Exception as e:
                logger.warning(f"Failed to load plugin {module_name}: {e}")

# Load plugins on import
_load_plugins()

# Explicit imports for better IDE support
try:
    from .z3_plugin import Z3Verifier
    __all__ = ["Z3Verifier"]
except ImportError:
    __all__ = []

try:
    from .cvc5_plugin import CVC5Verifier
    __all__.append("CVC5Verifier")
except ImportError:
    pass  # CVC5 not available
