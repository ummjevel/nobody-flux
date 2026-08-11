"""nobody-flux: a fully local ASR -> LLM -> TTS voice conversation pipeline.

Package layout
--------------

The modules are grouped by *what they are responsible for during a turn*,
rather than left flat:

``stage/``
    The three swappable model stages -- ``asr``, ``llm``, ``tts`` -- plus
    ``asr_stream`` (live incremental recognition) and ``_procio`` (the
    subprocess plumbing the isolated-venv backends share). These are the parts
    ``configs/models.yaml`` chooses between; nothing here knows about turns,
    devices, or conversation state.

``audio/``
    Everything that touches a sound device: ``session`` (duplex capture +
    playback backends), ``aec`` (frame-level echo cancellers), ``player``
    (reply playback queues) and ``resample``. Deals in samples, never text.

``turn/``
    Turn-taking: ``vad`` (speech boundaries), ``detector`` (Smart Turn v3
    endpointing), ``backchannel`` (lexical "is this a real turn?" filter) and
    ``controller`` (the state machine that drives a live conversation).

Top level keeps what does not belong to any one of those: ``pipeline`` (stage
orchestration), ``registry`` (config -> object construction), ``storage`` and
``memory`` (persistence), ``persona``, ``textchunk``, ``paths`` and
``platform_support``.

Import cost
-----------

This module deliberately does **not** re-export anything from the
subpackages. Importing ``nobody_flux`` should be nearly free: the heavy
dependencies (torch, sherpa-onnx, llama-cpp) each cost hundreds of
milliseconds to a couple of seconds to load, and a caller that only wants,
say, ``textchunk`` should not pay for all three. Import the specific module
you need -- ``from nobody_flux.turn.vad import VoiceActivityDetector`` -- and
you load exactly its dependencies.

The one thing that *must* happen at package import is native-runtime
preparation, below.
"""

from __future__ import annotations

from .platform_support import prepare_native_runtime

# Runs before any submodule of this package can be imported, which is exactly
# the ordering guarantee the native setup needs: it registers DLL directories
# (Windows), symlinks onnxruntime's shared library where sherpa-onnx looks for
# it (Linux/macOS), and pins the OpenMP environment variables that keep macOS
# from aborting on a duplicate libomp. Every one of those has to be in place
# *before* `import sherpa_onnx` / `import torch` / `import llama_cpp` runs.
#
# It is idempotent and cheap (a handful of directory checks), imports nothing
# heavier than the standard library, and is a no-op on platforms that need no
# fixups -- so paying for it unconditionally at package import is the right
# trade for never having to think about import order again.
prepare_native_runtime()

__all__ = ["prepare_native_runtime"]
