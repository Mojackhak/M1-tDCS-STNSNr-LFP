"""Resolve configured external Python modules once per process."""

from __future__ import annotations

from pathlib import Path
import sys


def activate_sextant(sextant_python_root: Path | str) -> Path:
    """Expose the configured Sextant Python package for shared primitives."""

    root = Path(sextant_python_root).expanduser().resolve()
    required = root / "sextant" / "shared" / "fiber_membership.py"
    if not required.is_file():
        raise FileNotFoundError(f"Sextant Python source does not exist: {required}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root
