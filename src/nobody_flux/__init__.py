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
    # KMP_DUPLICATE_LIB_OK lets the process past the "OMP Error #15" abort, but
    # its documented side effect is that the two OpenMP runtimes can then HANG
    # (or corrupt results) fighting over threads -- observed as llama.cpp's
    # generate() deadlocking on Apple Silicon. Pinning OpenMP to a single thread
    # avoids that contention. llama.cpp's own matmul uses ggml's separate thread
    # pool (n_threads), not OpenMP, so this doesn't slow the LLM; it only reins
    # in the OpenMP-using libs (torch etc.) that caused the conflict.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
