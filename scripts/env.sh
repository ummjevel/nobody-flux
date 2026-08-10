#!/usr/bin/env bash
# Source this before running anything that imports sherpa_onnx:
#   source scripts/env.sh
#   uv run python scripts/run_pipeline.py --wav-in ... --wav-out ...
#
# Why: sherpa-onnx's compiled extension dlopen()s a bare "libonnxruntime.so"
# (Linux) / "libonnxruntime.dylib" (macOS), but the onnxruntime pip wheel only
# ships a versioned file. The dynamic linker reads its search-path env var once
# at process startup, so it must be set in the shell BEFORE python/uv run
# launches -- setting os.environ from inside the process is too late.
# (src/nobody_flux/asr.py creates the matching unversioned symlink itself; this
# script only adds the directory to the linker search path.)
#
# macOS note: the macOS sherpa-onnx wheel usually self-locates its onnxruntime
# and needs none of this -- setting DYLD_LIBRARY_PATH below is harmless
# best-effort. If sherpa_onnx imports fine on your Mac without sourcing this,
# you don't need it.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Glob the python3.x dir rather than hardcoding a version.
CAPI_DIR="$(echo "$PROJECT_ROOT"/.venv/lib/python3.*/site-packages/onnxruntime/capi)"

case "$(uname -s)" in
    Darwin) export DYLD_LIBRARY_PATH="$CAPI_DIR:$DYLD_LIBRARY_PATH" ;;
    *)      export LD_LIBRARY_PATH="$CAPI_DIR:$LD_LIBRARY_PATH" ;;
esac
