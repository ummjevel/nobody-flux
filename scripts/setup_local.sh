#!/usr/bin/env bash
# Env setup for the local dev box (RTX 5090, currently tested from WSL2).
#
# Nothing here needs a different torch/CUDA index than setup_server.sh: plain
# `uv sync` already resolves a torch wheel with working CUDA on this GPU
# (confirmed: torch 2.13.0+cu130, torch.cuda.is_available() == True with no
# extra index config). If that ever stops being true for a newer/older card,
# add the index override here rather than in setup_common.sh, since it'd be
# specific to this machine's driver/CUDA version, not universal.
set -euo pipefail
export TARGET_LABEL="local/RTX5090"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_common.sh"

echo
echo "Local box: scripts/talk.py's mic loop should work here if WSLg audio"
echo "passthrough is set up, but that's unverified from an automated setup"
echo "script. If sounddevice can't see an input device, or the VAD loop just"
echo "hangs, don't fight it here -- test on the server instead and treat this"
echo "machine as a headless dev box for now (use scripts/run_pipeline.py with"
echo "pre-recorded wavs)."
