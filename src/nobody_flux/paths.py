"""Single source of truth for the project root.

Was being independently re-derived via Path(__file__).resolve().parents[N] in
registry.py, asr.py (twice), tts.py, and storage.py -- six copies of the same
computation, with asr.py/tts.py/storage.py using parents[2] and talk.py using
parents[1] for the same logical directory (different only because talk.py
lives one level shallower under scripts/, not src/nobody_flux/). Nothing
enforced that those stayed in sync if a file moved.

Lives in its own module rather than registry.py because registry.py imports
asr/llm/tts directly (`from . import asr, llm, tts`) -- those modules
importing PROJECT_ROOT back from registry would be circular.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
