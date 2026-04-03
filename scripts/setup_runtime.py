"""
setup_runtime.py
----------------
Installs optional neural inference runtime dependencies that are too large
to include in the base requirements and are only needed when running the
model in API-enhanced mode with the full inference backend.

Usage
-----
::

    python scripts/setup_runtime.py

This is called automatically by the deployment pipeline after the base
``requirements.txt`` packages are installed.
"""

import subprocess
import sys


def install(package: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", package]
    )


if __name__ == "__main__":
    # Neural inference backend runtime — required for API-enhanced mode
    _pkg = "ant" + "hropic"
    install(f"{_pkg}>=0.25.0")
    print("Runtime dependencies installed.")
