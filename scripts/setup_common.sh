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
#   5. clone+build microsoft/VibeASR.cpp into external/ if missing (second ASR
#      candidate, see src/nobody_flux/asr.py's VibeAsrBitnet docstring) and
#      download its two GGUF model files
#   6. set up FreyaTTS's own isolated venv if missing (second TTS candidate,
#      see src/nobody_flux/tts.py's FreyaTtsKo docstring) -- NOTE: the
#      models/freyatts-ko-voiceA/ checkpoint itself is NOT fetched here, it
#      comes from the sibling voice-announce-mcp project and isn't published
#      anywhere this script could download from yet; copy it in by hand
#   7. download sherpa-onnx's Matcha-TTS English assets (third TTS candidate,
#      see src/nobody_flux/tts.py's SherpaMatchaTts docstring -- no Korean
#      checkpoint exists upstream, this is for comparing sherpa-onnx's own
#      TTS runtime, not for the Korean pipeline)
#   8. download the TEN-VAD onnx model (src/nobody_flux/vad.py's
#      VoiceActivityDetector -- also via sherpa_onnx, no separate venv)
#
# NOTE on WSL2/drvfs (/mnt/c/...) checkouts: both external/ clones above
# involve either heavy Python package imports (MOSS-TTS-Nano) or a multi-file
# C++ compile (VibeASR.cpp), and both get noticeably slower on a drvfs mount
# (per-file I/O overhead) than on a native Linux filesystem. If you're on
# WSL2 and hit this, consider cloning to somewhere under your Linux home
# (e.g. ~/dev/VibeASR.cpp) and symlinking it into external/ instead --
# confirmed to cut MOSS-TTS-Nano's per-call `import torch` from ~46s to ~25s
# this way. Not automated here since it only matters for drvfs.
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

echo "== [$TARGET_LABEL] 1/8: uv sync (project deps) =="
uv sync

echo "== [$TARGET_LABEL] 2/8: GPU sanity check =="
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

echo "== [$TARGET_LABEL] 3/8: SenseVoice ASR model assets =="
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

echo "== [$TARGET_LABEL] 4/8: MOSS-TTS-Nano (external, isolated venv) =="
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

echo "== [$TARGET_LABEL] 5/8: VibeASR.cpp (ASR candidate, compiled binary) =="
VIBEASR_DIR="$PROJECT_ROOT/external/VibeASR.cpp"
if [ ! -d "$VIBEASR_DIR" ]; then
    echo "Cloning microsoft/VibeASR.cpp (with submodules) into external/..."
    mkdir -p "$PROJECT_ROOT/external"
    git clone --recursive --depth 1 https://github.com/microsoft/VibeASR.cpp.git "$VIBEASR_DIR"
else
    echo "Already present at $VIBEASR_DIR, skipping clone."
fi

# Two local fixes on top of upstream, see patches/vibeasr-cpp-wsl2-fixes.patch
# for the full diff+rationale in one place:
#   1. src/vae.cpp hardcodes a 128GB ggml compute context on non-Windows
#      (assumes Linux memory overcommit); ENOMEMs via posix_memalign on any
#      box whose RAM+swap is under 128GB -- patched down to 8GB.
#   2. src/asr_server.cpp runs the VAE's acoustic and semantic encode passes
#      sequentially even though they're independent (~90% of a warm request's
#      latency); patched to run them concurrently on two threads instead
#      (~15-20% faster, see src/nobody_flux/asr.py's VibeAsrBitnet docstring).
# `git apply --check` first so re-running setup on an already-patched clone
# is a no-op instead of an error (git apply isn't idempotent on its own).
VIBEASR_PATCH="$PROJECT_ROOT/patches/vibeasr-cpp-wsl2-fixes.patch"
if (cd "$VIBEASR_DIR" && git apply --check "$VIBEASR_PATCH" 2>/dev/null); then
    echo "Applying local fixes (memory size + concurrent VAE encode)..."
    (cd "$VIBEASR_DIR" && git apply "$VIBEASR_PATCH")
else
    echo "Local fixes already applied (or don't cleanly apply -- check by hand"
    echo "if this is a fresh clone), skipping."
fi

# Two binaries: asr_infer (one-shot CLI, reloads models every call -- used
# for quick manual testing) and asr_stream_server (persistent process, loads
# once then answers many requests over stdin/stdout -- what VibeAsrBitnet
# actually uses at runtime, see its docstring). Build both.
if [ ! -x "$VIBEASR_DIR/build/bin/asr_infer" ] || [ ! -x "$VIBEASR_DIR/build/bin/asr_stream_server" ]; then
    echo "Building asr_infer + asr_stream_server (cmake + gcc/clang required)..."
    if ! command -v cmake >/dev/null 2>&1; then
        echo "ERROR: cmake not found. Install it (apt install cmake, or"
        echo "'pip install --user cmake' inside a venv if you lack sudo) and re-run."
        exit 1
    fi
    (cd "$VIBEASR_DIR" && cmake -B build -DCMAKE_BUILD_TYPE=Release \
        && cmake --build build --target asr_infer -j"$(nproc)" \
        && cmake --build build --target asr_stream_server -j"$(nproc)")
else
    echo "asr_infer + asr_stream_server already built, skipping."
fi

VIBEASR_MODELS_DIR="$PROJECT_ROOT/models/vibeasr"
mkdir -p "$VIBEASR_MODELS_DIR"
for f in vibeasr-vae-encoder-i8_s.gguf vibeasr-lm-i2_s-embed-q6_k.gguf; do
    if [ ! -f "$VIBEASR_MODELS_DIR/$f" ]; then
        echo "Downloading $f..."
        curl -L -o "$VIBEASR_MODELS_DIR/$f" \
            "https://huggingface.co/microsoft/VibeVoice-ASR-BitNet/resolve/main/$f"
    else
        echo "$f already present, skipping."
    fi
done

echo "== [$TARGET_LABEL] 6/8: FreyaTTS (external, isolated venv) =="
FREYATTS_VENV_DIR="$PROJECT_ROOT/external/freyatts-venv"
if [ ! -x "$FREYATTS_VENV_DIR/bin/python" ]; then
    echo "Creating FreyaTTS's own venv (kept separate -- freyatts's own torch"
    echo "resolution shouldn't have to reconcile with this project's or"
    echo "MOSS-TTS-Nano's pins; see src/nobody_flux/tts.py's FreyaTtsKo docstring)..."
    uv venv --python 3.11 "$FREYATTS_VENV_DIR"
    uv pip install --python "$FREYATTS_VENV_DIR/bin/python" \
        'freyatts @ git+https://github.com/ummjevel/FreyaTTS.git' soundfile
else
    echo "FreyaTTS venv already set up, skipping."
fi

if [ ! -f "$PROJECT_ROOT/models/freyatts-ko-voiceA/model.safetensors" ]; then
    echo "NOTE: no models/freyatts-ko-voiceA/ checkpoint -- not published/downloadable"
    echo "yet, copy config.json + model.safetensors in by hand (e.g. from the"
    echo "sibling voice-announce-mcp project's models/freyatts-ko-voiceA/) to use"
    echo "the freyatts-ko-voicea TTS preset."
fi

echo "== [$TARGET_LABEL] 7/8: sherpa-onnx Matcha-TTS (English, comparison only) =="
MATCHA_DIR="$PROJECT_ROOT/models/sherpa-matcha-en"
if [ ! -f "$MATCHA_DIR/model-steps-3.onnx" ]; then
    echo "Downloading matcha-icefall-en_US-ljspeech (~71MB acoustic model + tokens + espeak-ng-data)..."
    TMP_TAR="$(mktemp --suffix=.tar.bz2)"
    curl -L -o "$TMP_TAR" \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2"
    mkdir -p "$MATCHA_DIR"
    tar xjf "$TMP_TAR" -C "$MATCHA_DIR" --strip-components=1
    rm -f "$TMP_TAR"
else
    echo "Already present at $MATCHA_DIR, skipping."
fi
if [ ! -f "$MATCHA_DIR/vocos-22khz-univ.onnx" ]; then
    echo "Downloading vocos-22khz-univ.onnx vocoder (~51MB)..."
    curl -L -o "$MATCHA_DIR/vocos-22khz-univ.onnx" \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx"
else
    echo "Vocoder already present, skipping."
fi

echo "== [$TARGET_LABEL] 8/8: TEN-VAD model =="
TEN_VAD_DIR="$PROJECT_ROOT/models/ten-vad"
if [ ! -f "$TEN_VAD_DIR/ten-vad.onnx" ]; then
    echo "Downloading ten-vad.onnx (~330KB)..."
    mkdir -p "$TEN_VAD_DIR"
    curl -L -o "$TEN_VAD_DIR/ten-vad.onnx" \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx"
else
    echo "Already present at $TEN_VAD_DIR, skipping."
fi

echo "== [$TARGET_LABEL] setup complete =="
