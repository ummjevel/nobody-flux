#!/usr/bin/env bash
# Shared provisioning logic for scripts/setup_local.sh (RTX 5090 dev box) and
# scripts/setup_server.sh (H100). Sourced, not run directly -- it expects
# TARGET_LABEL to already be set by the caller.
#
# What this does, in order:
#   1. uv sync this project's own venv
#   2. sanity-check the GPU is visible to torch
#   3. download the SenseVoice ASR model assets if missing (not committed to
#      git -- see models/.gitkeep and the project .gitignore)
#   4. clone MOSS-TTS-Nano into external/ if missing, and give it its OWN
#      isolated venv (see src/nobody_flux/tts.py's docstring for why: its
#      torch==2.7.0 pin would otherwise fight this project's own torch pin
#      the moment both share one venv -- this happened once during
#      development and silently broke both)
#
# Both steps that call `uv sync`/`uv pip install` force UV_LINK_MODE=copy:
# this repo has been run from a drvfs mount (WSL2, /mnt/c/...) where uv's
# default hardlink-from-cache install can silently drop files instead of
# erroring. If you're on a plain Linux filesystem this is just a bit slower,
# not required -- but it's cheap insurance, so it stays on for everyone.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export UV_LINK_MODE=copy

echo "== [$TARGET_LABEL] 1/4: uv sync (project deps) =="
uv sync

echo "== [$TARGET_LABEL] 2/4: GPU sanity check =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "nvidia-smi not found -- no GPU visible to this shell."
fi
uv run python -c "
import torch
print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('WARNING: running on CPU. ASR/LLM/TTS will all be much slower.')
"

echo "== [$TARGET_LABEL] 3/4: SenseVoice ASR model assets =="
SENSE_VOICE_DIR="$PROJECT_ROOT/models/sense-voice"
if [ ! -f "$SENSE_VOICE_DIR/model.int8.onnx" ]; then
    echo "Downloading sherpa-onnx SenseVoice-Small (int8, ~230MB)..."
    TMP_TAR="$(mktemp --suffix=.tar.bz2)"
    curl -L -o "$TMP_TAR" \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
    mkdir -p "$SENSE_VOICE_DIR"
    tar xjf "$TMP_TAR" -C "$SENSE_VOICE_DIR" --strip-components=1
    rm -f "$TMP_TAR"
else
    echo "Already present at $SENSE_VOICE_DIR, skipping."
fi

echo "== [$TARGET_LABEL] 4/4: MOSS-TTS-Nano (external, isolated venv) =="
MOSS_DIR="$PROJECT_ROOT/external/MOSS-TTS-Nano"
if [ ! -d "$MOSS_DIR" ]; then
    echo "Cloning OpenMOSS/MOSS-TTS-Nano into external/..."
    mkdir -p "$PROJECT_ROOT/external"
    git clone https://github.com/OpenMOSS/MOSS-TTS-Nano.git "$MOSS_DIR"
else
    echo "Already present at $MOSS_DIR, skipping clone."
fi
if [ ! -x "$MOSS_DIR/.venv/bin/python" ]; then
    echo "Creating MOSS-TTS-Nano's own venv (kept separate from this project's --"
    echo "see src/nobody_flux/tts.py docstring)..."
    # MOSS-TTS-Nano's own pyproject.toml dependencies omit `soundfile` (it's only
    # in its requirements.txt), so torchaudio.load() has no backend and errors
    # with "Couldn't find appropriate backend" -- confirmed by hand, so install
    # it explicitly rather than relying on -e . to pull it in.
    (cd "$MOSS_DIR" && uv venv --python 3.11 .venv \
        && uv pip install --python .venv/bin/python -e . \
        && uv pip install --python .venv/bin/python soundfile)
else
    echo "MOSS-TTS-Nano venv already set up, skipping."
fi

if [ ! -f "$PROJECT_ROOT/data/reference_voice_16k.wav" ]; then
    echo "NOTE: no data/reference_voice_16k.wav -- TTS voice_clone mode needs a"
    echo "reference clip. Place a 16kHz mono wav there (see README.md)."
fi

echo "== [$TARGET_LABEL] setup complete =="
