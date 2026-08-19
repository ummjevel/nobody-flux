<#
.SYNOPSIS
    Provision the native-Windows environment (.venv-win) and the models it needs.

.DESCRIPTION
    The Linux/WSL2 setup scripts (setup_local.sh, setup_server.sh) cannot be used
    here, and not only because they are bash. They provision a GPU development
    box: they check nvidia-smi, build VibeASR.cpp with cmake, and create isolated
    venvs for two PyTorch TTS projects. This environment exists for a narrower
    and quite different reason.

    Why a Windows environment at all
    --------------------------------

    Every turn-taking parameter in this project was an estimate, because the
    development box could not open a real microphone: WSL2 reaches audio through
    WSLg's PulseAudio bridge, where capture is unreliable and buffering is not
    the hardware's. Native Windows enumerates real devices through WASAPI, so the
    live loop can finally be measured rather than guessed at -- and the first
    measurements immediately found a VAD threshold that made turns never end and
    a duplex stream that could not open at all. Both had been invisible for
    months.

    It is also deliberately CPU-only. The deployment target is a CM4-class board
    with no GPU, so CPU numbers measured here are the ones that actually
    transfer.

    What this installs, and what it leaves out
    ------------------------------------------

    Installed: the core pipeline (sherpa-onnx ASR/TTS/VAD, llama.cpp for the
    default LLM preset, PortAudio bindings) and the model files those presets
    need.

    Left out on purpose: VibeASR.cpp (needs MSVC + cmake to build), MOSS-TTS-Nano
    and FreyaTTS (isolated PyTorch venvs, both GPU-oriented). Those presets stay
    available on the Linux boxes. Nothing in the default pipeline or the
    turn-taking work touches them, and requiring a C++ toolchain to measure a VAD
    threshold would be a poor trade.

.PARAMETER SkipModels
    Provision the venv only. Useful when the models are already in place and just
    the dependencies need reinstalling.

.PARAMETER Force
    Recreate .venv-win from scratch even if it already exists.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

.EXAMPLE
    .venv-win\Scripts\python.exe scripts\_smoke_imports.py
    Verify afterwards. This script runs it for you as its last step.
#>

[CmdletBinding()]
param(
    [switch]$SkipModels,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot '.venv-win'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$Requirements = Join-Path $ProjectRoot 'requirements\windows-cpu.txt'

# Python 3.12: sherpa-onnx and llama-cpp-python both publish cp312 Windows
# wheels, and this is the version the environment was validated on.
$PythonVersion = '3.12'

# torch from the CPU index rather than PyPI, whose default Windows wheel bundles
# a ~3GB CUDA runtime this box has no use for. llama-cpp-python from the
# project's prebuilt CPU wheel index -- on PyPI it is an sdist that needs MSVC
# and CMake to compile, which is exactly the dependency this environment avoids.
$TorchIndex = 'https://download.pytorch.org/whl/cpu'
$LlamaIndex = 'https://abetlen.github.io/llama-cpp-python/whl/cpu'

function Write-Step {
    param([int]$Number, [int]$Total, [string]$Message)
    Write-Host ""
    Write-Host "== [windows] $Number/$Total`: $Message ==" -ForegroundColor Cyan
}

function Get-Sherpa {
    <#
        Download and unpack one of k2-fsa's .tar.bz2 model releases.

        Expand-Archive cannot read bzip2, and Windows has no bundled tool that
        can -- but tar.exe has shipped in Windows 10 1803 and later, and handles
        bz2 through libarchive. Using it keeps this script dependency-free.
    #>
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Marker,   # a file that exists iff this is already unpacked
        [string]$Label
    )
    if (Test-Path (Join-Path $Destination $Marker)) {
        Write-Host "  $Label already present, skipping."
        return
    }
    Write-Host "  downloading $Label..."
    $temp = Join-Path ([System.IO.Path]::GetTempPath()) "todak-$([System.IO.Path]::GetRandomFileName()).tar.bz2"
    $staging = Join-Path ([System.IO.Path]::GetTempPath()) "todak-$([System.IO.Path]::GetRandomFileName())"
    try {
        # Invoke-WebRequest's progress bar makes large downloads dramatically
        # slower in Windows PowerShell (it repaints per chunk), so it is off for
        # the duration of the transfer only.
        $previous = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $Url -OutFile $temp -UseBasicParsing
        $ProgressPreference = $previous

        # Unpacked to staging first, then the single top-level directory the
        # archive contains is moved into place. This is the equivalent of the
        # bash scripts' `tar --strip-components=1`, which Windows tar supports
        # but expresses less portably.
        New-Item -ItemType Directory -Force -Path $staging | Out-Null
        tar -xjf $temp -C $staging
        if ($LASTEXITCODE -ne 0) { throw "tar failed to unpack $Label" }

        $inner = Get-ChildItem -Path $staging -Directory | Select-Object -First 1
        if ($null -eq $inner) { throw "$Label unpacked with no top-level directory" }
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
        Get-ChildItem -Path $inner.FullName -Force |
            Move-Item -Destination $Destination -Force
    }
    finally {
        Remove-Item $temp -Force -ErrorAction SilentlyContinue
        Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Get-File {
    param([string]$Url, [string]$Path, [string]$Label)
    if (Test-Path $Path) {
        Write-Host "  $Label already present, skipping."
        return
    }
    Write-Host "  downloading $Label..."
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $previous = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $Url -OutFile $Path -UseBasicParsing
    $ProgressPreference = $previous
}

$total = if ($SkipModels) { 3 } else { 11 }

# --- 1. uv ------------------------------------------------------------------

Write-Step 1 $total "uv"
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    throw @"
uv not found on PATH. Install it, then re-run:
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

(uv rather than pip: the venv here has no pip at all, and uv resolves and
installs the same lockfile-adjacent set an order of magnitude faster.)
"@
}
Write-Host "  using $uv"

# --- 2. venv ----------------------------------------------------------------

Write-Step 2 $total ".venv-win (Python $PythonVersion, CPU-only)"
if ($Force -and (Test-Path $VenvDir)) {
    Write-Host "  -Force given, removing the existing environment..."
    Remove-Item $VenvDir -Recurse -Force
}
if (Test-Path $VenvPython) {
    Write-Host "  already present at $VenvDir, skipping creation."
} else {
    & uv venv --python $PythonVersion $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
}

# --- 3. dependencies --------------------------------------------------------

Write-Step 3 $total "dependencies (requirements/windows-cpu.txt)"
Write-Host "  torch from $TorchIndex"
Write-Host "  llama-cpp-python from $LlamaIndex"
& uv pip install --python $VenvPython -r $Requirements `
    --extra-index-url $TorchIndex `
    --extra-index-url $LlamaIndex `
    --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

if ($SkipModels) {
    Write-Step 3 $total "verifying imports"
    & $VenvPython (Join-Path $ProjectRoot 'scripts\_smoke_imports.py')
    exit $LASTEXITCODE
}

# --- 4-8. models ------------------------------------------------------------
#
# Only what the CPU pipeline and the turn-taking work actually load. Each is
# skipped when already present, so re-running this script is cheap.

Write-Step 4 $total "SenseVoice ASR (default asr preset)"
Get-Sherpa -Label 'SenseVoice-Small (~230MB)' `
    -Url 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2' `
    -Destination (Join-Path $ProjectRoot 'models\sense-voice') `
    -Marker 'model.int8.onnx'

Write-Step 5 $total "streaming Zipformer Korean ASR (Phase 3)"
Get-Sherpa -Label 'streaming-zipformer-korean-2024-06-16 (~60MB int8)' `
    -Url 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2' `
    -Destination (Join-Path $ProjectRoot 'models\streaming-zipformer-ko') `
    -Marker 'encoder-epoch-99-avg-1.int8.onnx'

Write-Step 6 $total "TEN-VAD + Smart Turn v3 (turn taking)"
Get-File -Label 'ten-vad.onnx (~330KB)' `
    -Url 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx' `
    -Path (Join-Path $ProjectRoot 'models\ten-vad\ten-vad.onnx')
Get-File -Label 'smart-turn-v3.2-cpu.onnx (~8.7MB)' `
    -Url 'https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx' `
    -Path (Join-Path $ProjectRoot 'models\smart-turn-v3\smart-turn-v3.2-cpu.onnx')

Write-Step 7 $total "LLM weights"
# The default preset. Chosen by measuring conversational behaviour rather than
# benchmark scores -- see docs/llm-conversational-selection.md.
Get-File -Label 'Mi:dm-2.0-Mini Q4_K_M (~1.3GB, default LLM)' `
    -Url 'https://huggingface.co/mykor/Midm-2.0-Mini-Instruct-gguf/resolve/main/Midm-2.0-Mini-Instruct-Q4_K_M.gguf' `
    -Path (Join-Path $ProjectRoot 'models\midm-2.3b-gguf\Midm-2.0-Mini-Instruct-Q4_K_M.gguf')
# Kept as well: it is the fast fallback if the 2.3B turns out too slow on the
# target hardware, and _ab_persona.py compares against it.
Get-File -Label 'Qwen3-0.6B-Q4_K_M.gguf (~460MB, fast fallback)' `
    -Url 'https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf' `
    -Path (Join-Path $ProjectRoot 'models\qwen3-0.6b-gguf\Qwen3-0.6B-Q4_K_M.gguf')

Write-Step 8 $total "Matcha-TTS assets"
# The English checkpoint is fetched for its espeak-ng-data, which the Korean
# preset shares -- espeak-ng-data is a multi-language phoneme dictionary, not a
# per-voice asset (see configs/models.yaml's sherpa-matcha-ko entry).
Get-Sherpa -Label 'matcha-icefall-en_US-ljspeech (~71MB)' `
    -Url 'https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2' `
    -Destination (Join-Path $ProjectRoot 'models\sherpa-matcha-en') `
    -Marker 'model-steps-3.onnx'
Get-File -Label 'vocos-22khz-univ.onnx vocoder (~51MB)' `
    -Url 'https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx' `
    -Path (Join-Path $ProjectRoot 'models\sherpa-matcha-en\vocos-22khz-univ.onnx')

$koreanAcoustic = Join-Path $ProjectRoot 'models\sherpa-matcha-ko\matcha-ko-voiceA-ep499-steps10.onnx'
if (-not (Test-Path $koreanAcoustic)) {
    Write-Host ""
    Write-Warning @"
The DEFAULT TTS preset (sherpa-matcha-ko) needs models\sherpa-matcha-ko\, which
is a custom-trained Korean checkpoint with no public download -- the bash setup
scripts cannot fetch it either. Copy the directory in by hand (acoustic model,
vocoder and tokens.txt) from a machine that has it.

Until then, run with --tts sherpa-matcha-en, which this script did install.
"@
}

Write-Step 9 $total "Supertonic 3 TTS (comparison preset)"
# Korean-capable, character-level (no G2P, no espeak-ng-data), and already
# supported by the pinned sherpa-onnx -- OfflineTtsSupertonicModelConfig ships
# in 1.13.4, so no dependency bump is needed for this.
#
# LICENSE: the LICENSE file inside the tarball reads MIT, but that is
# Supertone's license for their sample *code*; the README beside it states the
# model is OpenRAIL-M. Commercial use is permitted royalty-free, and the
# use-based restrictions must be passed downstream. See configs/models.yaml.
#
# Not the default: measured on this platform it ties sherpa-matcha-ko on
# intelligibility and is roughly twice as slow (see the SherpaSupertonicTts
# docstring for the numbers).
Get-Sherpa -Label 'sherpa-onnx-supertonic-3-tts-int8 (~123MB, 31 langs, 10 speakers)' `
    -Url 'https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2' `
    -Destination (Join-Path $ProjectRoot 'models\sherpa-supertonic-3') `
    -Marker 'vector_estimator.int8.onnx'

Write-Step 10 $total "Supertonic 2 TTS (the fast Supertonic)"
# Both v2 and v3, because they are not interchangeable on speed: v3 raised the
# flow-matching steps from 5 to 8 and measures ~2.9x slower, and the step count is
# not adjustable through sherpa-onnx (no field on the model config, no key in
# tts.json). So v2 is the only route to the cheaper setting. Same OpenRAIL-M
# weights licence as v3.
Get-Sherpa -Label 'sherpa-onnx-supertonic-tts-int8 v2 (~81MB, 5 langs incl. Korean)' `
    -Url 'https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2' `
    -Destination (Join-Path $ProjectRoot 'models\sherpa-supertonic-2') `
    -Marker 'vector_estimator.int8.onnx'

# --- 11. verify -------------------------------------------------------------

Write-Step 11 $total "verifying"
& $VenvPython (Join-Path $ProjectRoot 'scripts\_smoke_imports.py')
if ($LASTEXITCODE -ne 0) { throw "import smoke test failed" }

Write-Host ""
Write-Host "== [windows] setup complete ==" -ForegroundColor Green
Write-Host @"

Next, in order -- each one checks something the previous cannot:

  .venv-win\Scripts\python.exe scripts\_smoke_turn.py
      Streaming ASR and the turn controller, on recorded audio. No microphone.

  .venv-win\Scripts\python.exe scripts\_calibrate_vad_threshold.py --apply
      Measures THIS machine's noise floor and sets configs/vad.yaml's threshold
      from it. Do not skip: the value is device dependent, and a wrong one makes
      turns never end. Stay quiet while it records.

  .venv-win\Scripts\python.exe scripts\_calibrate_aec_delay.py
      Speaker-to-mic delay into configs/audio.yaml. Only needed for --aec.

  .venv-win\Scripts\python.exe scripts\_smoke_duplex.py
      Plays a reply aloud and checks the assistant does not interrupt itself.

  .venv-win\Scripts\python.exe scripts\talk.py --streaming-asr
      The conversation loop.
"@
