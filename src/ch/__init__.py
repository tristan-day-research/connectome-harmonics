"""Convenient imports for the connectome-harmonics package (ch).

Automatically imports all submodules and their functions to simplify usage.
"""

import pkgutil
import importlib
import sys
from pathlib import Path

# Automatically import all submodules
package_name = __name__
package_path = Path(__file__).parent

for module_info in pkgutil.walk_packages([str(package_path)]):
    module_name = f"{package_name}.{module_info.name}"
    try:
        module = importlib.import_module(module_name)
        for attr_name in dir(module):
            if not attr_name.startswith("_"):
                setattr(sys.modules[package_name], attr_name, getattr(module, attr_name))
    except ImportError as e:
        # Skip modules that can't be imported due to missing dependencies
        print(f"Warning: Could not import {module_name}: {e}")
        continue

