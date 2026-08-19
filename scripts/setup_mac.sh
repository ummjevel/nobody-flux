#!/usr/bin/env bash
# macOS (Apple Silicon or Intel) setup -- provisions the CPU/ONNX DEFAULT
# pipeline only. Deliberately does NOT do the GPU/CUDA-only bits that
# setup_local.sh / setup_server.sh (via setup_common.sh) do: no MOSS-TTS-Nano
# or FreyaTTS venvs (CUDA torch), no VibeASR.cpp build (uses `nproc` + a
# Linux-memory patch). Uses only portable shell (no `mktemp --suffix`, no
# `nproc`, no `nvidia-smi`).
#
# NOT hardware-verified on a real Mac yet (this project's dev boxes are
# Linux/WSL2). It's based on the fact that every default-pipeline dependency
# (sherpa-onnx, llama-cpp-python, onnxruntime, sounddevice, torch) ships macOS
# wheels, and sounddevice's mic actually works on macOS (unlike WSL2).
#
# Usage:  bash scripts/setup_mac.sh

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Portable temp file: a template ending in XXXXXX works on both GNU (Linux) and
# BSD (macOS) mktemp; plain `mktemp` with no args does NOT on BSD.
_mktmp() { mktemp "${TMPDIR:-/tmp}/nobody_flux.XXXXXX"; }

fetch_tar() {  # $1=dest_dir  $2=sentinel_file  $3=url  $4=desc
    if [ ! -f "$1/$2" ]; then
        echo "Downloading $4..."
        local tmp; tmp="$(_mktmp)"
        curl -L -o "$tmp" "$3"          # tar xjf reads bz2 regardless of filename
        mkdir -p "$1"
        tar xjf "$tmp" -C "$1" --strip-components=1
        rm -f "$tmp"
    else
        echo "Already present: $1, skipping."
    fi
}

fetch_file() {  # $1=dest_path  $2=url  $3=desc
    if [ ! -f "$1" ]; then
        echo "Downloading $3..."
        mkdir -p "$(dirname "$1")"
        curl -L -o "$1" "$2"
    else
        echo "Already present: $1, skipping."
    fi
}

echo "== [mac] 1/8: uv sync =="
uv sync

echo "== [mac] 2/8: SenseVoice ASR (default) =="
fetch_tar "$PROJECT_ROOT/models/sense-voice" model.int8.onnx \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2" \
    "SenseVoice-Small int8 (~230MB)"

echo "== [mac] 3/8: TEN-VAD =="
fetch_file "$PROJECT_ROOT/models/ten-vad/ten-vad.onnx" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx" \
    "ten-vad.onnx (~330KB)"

# Second VAD engine (configs/vad.yaml `engine: silero-vad`). Fetched by default
# because the reason to switch is a licence term, not a bug: TEN-VAD is
# Apache-2.0 "with additional conditions" (non-compete -- THIRD-PARTY-NOTICES.md
# 2.3), Silero VAD is plain MIT. ten-vad stays the default.
fetch_file "$PROJECT_ROOT/models/silero-vad/silero_vad.onnx" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" \
    "silero_vad.onnx (~629KB, alternative VAD)"

echo "== [mac] 4/8: LLM weights (GGUF) =="
# Default preset, picked by measuring conversational behaviour rather than
# benchmark scores -- see docs/llm-conversational-selection.md.
fetch_file "$PROJECT_ROOT/models/midm-2.3b-gguf/Midm-2.0-Mini-Instruct-Q4_K_M.gguf" \
    "https://huggingface.co/mykor/Midm-2.0-Mini-Instruct-gguf/resolve/main/Midm-2.0-Mini-Instruct-Q4_K_M.gguf" \
    "Mi:dm-2.0-Mini Q4_K_M (~1.3GB, default LLM)"
# Fast fallback, and what _ab_persona.py compares the default against.
fetch_file "$PROJECT_ROOT/models/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf" \
    "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf" \
    "Qwen3-0.6B-Q4_K_M.gguf (~460MB, fast fallback)"

echo "== [mac] 5/8: Smart Turn v3 (optional endpoint) + Matcha-EN espeak-ng-data =="
fetch_file "$PROJECT_ROOT/models/smart-turn-v3/smart-turn-v3.2-cpu.onnx" \
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx" \
    "smart-turn-v3.2-cpu.onnx (~8.7MB)"
# Matcha-EN provides espeak-ng-data, which the sherpa-matcha-ko default TTS
# reuses (see configs/models.yaml: data_dir points here).
fetch_tar "$PROJECT_ROOT/models/sherpa-matcha-en" model-steps-3.onnx \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2" \
    "matcha-en (acoustic + tokens + espeak-ng-data, ~71MB)"
fetch_file "$PROJECT_ROOT/models/sherpa-matcha-en/vocos-22khz-univ.onnx" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx" \
    "vocos-22khz-univ.onnx vocoder (~51MB)"

echo "== [mac] 6/8: streaming Zipformer Korean ASR (optional) =="
fetch_tar "$PROJECT_ROOT/models/streaming-zipformer-ko" encoder-epoch-99-avg-1.int8.onnx \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2" \
    "streaming-zipformer-ko int8 (~60MB)"


echo "== [mac] 7/8: Supertonic 3 TTS (comparison preset) =="
# Belongs in the mac set because it is pure CPU/ONNX -- no CUDA torch, no venv,
# nothing this script otherwise skips. Korean-capable and character-level, so
# unlike sherpa-matcha-ko it needs no espeak-ng-data at all.
#
# LICENSE: the tarball's own LICENSE file says MIT, which covers Supertone's
# sample code, not these weights -- the README beside it states the model is
# OpenRAIL-M (commercial use allowed, use-based restrictions must be passed
# downstream). See configs/models.yaml before shipping on it.
fetch_tar "$PROJECT_ROOT/models/sherpa-supertonic-3" vector_estimator.int8.onnx \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2" \
    "supertonic-3 int8 (~123MB, 31 langs, 10 speakers)"


echo "== [mac] 8/8: Supertonic 2 TTS (the fast Supertonic) =="
# v2 as well as v3: v3 raised the flow-matching steps 5->8 and is ~2.9x slower,
# and sherpa-onnx exposes no way to lower the step count, so v2 is the only route
# to the cheaper setting. Same OpenRAIL-M weights licence as v3.
fetch_tar "$PROJECT_ROOT/models/sherpa-supertonic-2" vector_estimator.int8.onnx \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2" \
    "supertonic-2 int8 (~81MB, 5 langs incl. Korean)"

cat <<'NOTE'

== [mac] setup complete ==

Default pipeline = sense-voice-small (ASR) + midm-2.3b-gguf (LLM) + sherpa-matcha-ko (TTS).

Before running talk.py, note:
- The Korean default TTS (sherpa-matcha-ko) needs models/sherpa-matcha-ko/ (a custom
  checkpoint, NOT downloadable here -- copy it in by hand, same as on Linux). To test the
  pipeline end-to-end WITHOUT it, use `--tts sherpa-matcha-en` (English, fetched above).
- GPU-only presets (freyatts-ko-voicea, moss-tts-nano) are NOT set up -- they need CUDA.
- Metal-accelerated LLM: add a models.yaml LLM preset with NobodyLLMGguf n_gpu_layers=-1
  (default is 0 = CPU). Raw-transformers presets auto-use MPS on Apple Silicon.
- You usually do NOT need `source scripts/env.sh` on macOS (sherpa-onnx self-locates its
  onnxruntime); source it only if `import sherpa_onnx` fails.

Run:  uv run python scripts/talk.py            # mic works on macOS (unlike WSL2)
NOTE
