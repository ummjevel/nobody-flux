#!/usr/bin/env bash
# Source this before running anything that imports sherpa_onnx:
#   source scripts/env.sh
#   uv run python scripts/run_pipeline.py --wav-in ... --wav-out ...
#
# Why: sherpa-onnx's compiled extension dlopen()s a bare "libonnxruntime.so",
# but the onnxruntime pip wheel only ships the versioned
# "libonnxruntime.so.1.27.0". glibc's dynamic linker parses LD_LIBRARY_PATH
# once at process startup, so this has to be set in the shell before
# python/uv run launches -- setting it from inside a running Python process
# (os.environ, after the fact) does not work, it's too late by then.
# (src/nobody_flux/asr.py creates the matching unversioned symlink automatically;
# this script only needs to add its directory to the search path.)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$PROJECT_ROOT/.venv/lib/python3.11/site-packages/onnxruntime/capi:$LD_LIBRARY_PATH"
