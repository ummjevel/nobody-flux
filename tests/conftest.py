"""Make `src.nobody_flux` importable from tests the same way the scripts
import it: with the project root on sys.path. (The package is laid out as
src/nobody_flux without being installed; scripts rely on the repo root being
sys.path[0], so tests reproduce that arrangement explicitly.)

Note for running: sherpa_onnx needs the onnxruntime library dir on the linker
path -- `source scripts/env.sh` first, exactly as for the smoke scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
