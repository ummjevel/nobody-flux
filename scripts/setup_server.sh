#!/usr/bin/env bash
# Env setup for the H100 server. Same steps as setup_local.sh -- H100 (Hopper,
# sm_90) is well within what any recent stable torch/CUDA wheel supports, so
# there's no server-specific torch index needed either. This script exists
# mainly so setup output is labeled correctly and so the two entrypoints can
# diverge later without touching setup_common.sh's shared logic.
set -euo pipefail
export TARGET_LABEL="server/H100"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_common.sh"

echo
echo "Server box: assume headless, no audio devices. Use scripts/run_pipeline.py"
echo "(--wav-in/--wav-out, optionally --asr/--llm/--tts to pick a preset) for"
echo "latency/quality comparisons here rather than scripts/talk.py's live loop."
