import os
import platform

# macOS: torch, onnxruntime, llama.cpp and sherpa-onnx each link their own
# libomp.dylib; whichever OpenMP runtime initializes second aborts the process
# with "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
# already initialized." (confirmed on Apple Silicon). This is the documented
# workaround, and it must be set BEFORE any of those native libs load their
# OpenMP -- this package __init__ runs before its own submodules
# (asr/llm/tts import torch/sherpa/llama), so setting it here is early enough.
# setdefault so an explicit env override still wins; scoped to macOS so Linux/
# CM4 behavior is untouched.
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
