"""Local TTS stage: Korean text in -> wav out.

MOSS-TTS-Nano (OpenMOSS/Fudan, Apache-2.0, 100M params, official CPU-realtime
"4-core, zero GPU" spec, Korean in its 20 supported languages -- see
docs/output/ondevice_asr_llm_tts_research_20260716.md). Shelled out to via its
own infer.py rather than imported as a library: its `moss_tts_nano` package is
really a script-style CLI tool (see MOSS-TTS-Nano/pyproject.toml's `py-modules`),
and driving it as a subprocess sidesteps import-order/global-state assumptions
in code we don't control.

Voice: voice_clone mode needs a reference clip (there's no bundled default
voice, and no Korean sample ships with the repo -- see assets/demo.jsonl,
Chinese-only). Standing in with a placeholder enrollment recording for this
prototype; swap for a properly licensed/chosen persona voice before anything
user-facing.

WeTextProcessing is disabled (--disable-wetext-processing): its language
resolver only ever returns "zh" or "en" (see
MOSS-TTS-Nano/text_normalization_pipeline.py:resolve_text_normalization_language)
so plain Korean text would silently fall through to the *Chinese* ITN
normalizer -- and avoiding it also means we don't need the pynini/
WeTextProcessing native-dependency install. The repo's own regex-only
`normalize_tts_text` robustness pass (language-agnostic, no pynini) still runs.

Interpreter: this shells out with `repo_dir/.venv/bin/python`, NOT this
project's own `sys.executable`. MOSS-TTS-Nano pins torch==2.7.0/transformers
exactly; nobody-flux's own venv tracks a newer torch independently (currently
2.13.0) for the LLM stage, and letting `uv sync` reconcile a single shared venv
against both pins would force one project's torch version to lose -- it
already happened once (see scripts/setup_common.sh, which creates this
isolated venv). If repo_dir has no .venv, this falls back to sys.executable
with a warning, which only works if someone manually installed moss_tts_nano
into this project's own venv (fragile -- uv sync will silently uninstall it
again, since it isn't a declared dependency here).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# sherpa_onnx (used by SherpaMatchaTts below) needs the same
# libonnxruntime.so symlink + LD_LIBRARY_PATH setup that asr.py's
# module-level code performs before its own `import sherpa_onnx` -- relying
# on that having already run rather than repeating it here, since
# registry.py always imports asr before tts (`from . import asr, llm, tts`)
# and that's the only real entry point into this module. Importing tts.py
# some other way, before asr.py, would need the same dance asr.py does.
import sherpa_onnx  # noqa: E402 -- see asr.py's module docstring for why this needs LD_LIBRARY_PATH set first
import soundfile as sf  # noqa: E402

from ._procio import LineReader, StderrDrainer, clean_subprocess_env
from ..paths import PROJECT_ROOT
from ..platform_support import venv_interpreter

MOSS_TTS_NANO_REPO = PROJECT_ROOT / "external" / "MOSS-TTS-Nano"
DEFAULT_REFERENCE_AUDIO = PROJECT_ROOT / "data" / "reference_voice_16k.wav"


def _synthesize_to_array(tts, text: str) -> tuple[np.ndarray, int]:
    """synthesize() a preset that only knows how to write a wav file, then read
    it back as a mono float32 array. The streaming pipeline (pipeline.run_streaming)
    plays per-sentence chunks from memory, so it needs samples, not a path -- but
    the subprocess-backed presets (FreyaTtsKo/NobodyTTS) can only produce a file.
    In-process presets (SherpaMatchaTts) override synthesize_audio() to skip this
    round-trip entirely."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        tts.synthesize(text, tmp)
        audio, sr = sf.read(tmp, dtype="float32", always_2d=False)
    finally:
        os.unlink(tmp)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


@dataclass
class NobodyTTS:
    repo_dir: Path = MOSS_TTS_NANO_REPO
    reference_audio: Path = DEFAULT_REFERENCE_AUDIO
    # Real default lives in configs/models.yaml's tts preset params, not here --
    # this is just the fallback for constructing NobodyTTS directly (tests,
    # a REPL). "auto" (CUDA if visible) crashes on the RTX 5090 dev box with
    # "no kernel image is available for execution on the device": MOSS-TTS-Nano's
    # own venv pins torch==2.7.0/cu126, whose prebuilt kernels predate Blackwell
    # (sm_120) -- confirmed by hand, not a hypothetical. CPU sidesteps this
    # entirely and matches the model's own official spec ("4-core CPU, zero GPU
    # realtime"), so this isn't even a compromise. Hardware where MOSS-TTS-Nano's
    # pinned torch build actually has matching kernels (e.g. H100/sm_90, well
    # within cu126's range) can override via --tts preset params or a new
    # preset, not a code edit -- that's the point of configs/models.yaml.
    device: str = "cpu"
    # infer.py loads its own model on each invocation (no server/warm process),
    # so a stuck load or hung generation would otherwise block forever with no
    # feedback. 120s covers a cold model load plus generation for this
    # persona's short replies with headroom to spare.
    timeout_seconds: float = 120.0

    def _interpreter(self) -> str:
        # venv_interpreter, not a hard-coded "bin/python": that layout is POSIX
        # only, and this project now also runs on native Windows, where the
        # executable lives at Scripts/python.exe.
        venv_python = venv_interpreter(self.repo_dir / ".venv")
        if venv_python.exists():
            return str(venv_python)
        warnings.warn(
            f"No isolated venv at {venv_python}; falling back to {sys.executable}. "
            "Run scripts/setup_common.sh to create MOSS-TTS-Nano's own venv.",
            stacklevel=2,
        )
        return sys.executable

    def synthesize(self, text: str, out_path: str) -> str:
        """Synthesize `text` (Korean) to a 48kHz stereo wav at `out_path`."""
        cmd = [
            self._interpreter(),
            str(self.repo_dir / "infer.py"),
            "--prompt-audio-path",
            str(self.reference_audio),
            "--text",
            text,
            "--out",
            out_path,
            "--disable-wetext-processing",
            "--device",
            self.device,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"MOSS-TTS-Nano timed out after {self.timeout_seconds}s"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(f"MOSS-TTS-Nano failed:\n{result.stderr[-2000:]}")
        return out_path

    def synthesize_audio(self, text: str) -> tuple[np.ndarray, int]:
        """Mono float32 samples + sample rate (see _synthesize_to_array)."""
        return _synthesize_to_array(self, text)


FREYATTS_VENV_DIR = PROJECT_ROOT / "external" / "freyatts-venv"
DEFAULT_FREYATTS_MODEL_DIR = PROJECT_ROOT / "models" / "freyatts-ko-voiceA"
FREYATTS_SERVER_SCRIPT = PROJECT_ROOT / "scripts" / "_freyatts_server.py"


@dataclass
class FreyaTtsKo:
    """Second TTS candidate: FreyaTTS (github.com/ummjevel/FreyaTTS), a Korean
    fork distilled from Qwen3-TTS, checkpoint "voiceA" (from a sibling project,
    voice-announce-mcp -- see its models/freyatts-ko-voiceA/, copied into this
    project's models/ dir). Flow-matching DiT + frozen VoxCPM2 AudioVAE; no
    speaker embedding -- the noise seed *is* the voice (seed 9 = voiceA, per
    that project's confirmed_voices/best_seeds.json).

    Unlike MOSS-TTS-Nano, this doesn't pin an exact torch version (just
    "torch" in its deps), so its isolated venv resolved torch==2.13.0+cu130 --
    which DOES have sm_120 (RTX 5090/Blackwell) kernels, unlike MOSS-TTS-Nano's
    forced torch==2.7.0+cu126. CUDA actually works here.

    Backed by scripts/_freyatts_server.py, a persistent process (in FreyaTTS's
    own isolated venv) that loads the model once via FreyaTTS.from_pretrained
    and then answers requests over stdin/stdout (JSON lines in, JSON lines
    out -- see that script's header comment). Started lazily on first
    synthesize() call, reused across calls, and must be shut down via close()
    (STSPipeline.close() does this) -- otherwise it lingers as an orphaned
    process holding GPU memory. This cut per-turn latency from ~7.4s (previous
    per-call subprocess, reloading the model + re-fetching the VoxCPM2 AudioVAE
    every time) to ~0.7-1.4s after the one-time ~5.7s startup cost.

    Setup (see scripts/setup_common.sh step 6): `uv venv --python 3.11
    external/freyatts-venv && uv pip install --python
    external/freyatts-venv/bin/python 'freyatts @
    git+https://github.com/ummjevel/FreyaTTS.git' soundfile` (freyatts is a
    normal pip package, no repo clone needed -- simpler setup than
    MOSS-TTS-Nano's).
    """

    venv_dir: Path = FREYATTS_VENV_DIR
    model_dir: Path = DEFAULT_FREYATTS_MODEL_DIR
    device: str = "cuda"
    steps: int = 32
    seed: int = 9
    # from_pretrained() also fetches the frozen VoxCPM2 AudioVAE from HF on
    # first use (cached after); budget for that plus a cold model load. Only
    # paid once (server startup), not per request.
    startup_timeout_seconds: float = 60.0
    request_timeout_seconds: float = 60.0

    def __post_init__(self):
        self._proc: subprocess.Popen | None = None
        self._stdout_reader: LineReader | None = None
        self._stderr_drainer: StderrDrainer | None = None

    def _interpreter(self) -> str:
        venv_python = venv_interpreter(self.venv_dir)
        if not venv_python.exists():
            raise FileNotFoundError(
                f"No freyatts venv at {venv_python}. Create it with "
                "`uv venv --python 3.11 external/freyatts-venv && uv pip install "
                "--python external/freyatts-venv/bin/python "
                "'freyatts @ git+https://github.com/ummjevel/FreyaTTS.git' soundfile` "
                "(see src/nobody_flux/tts.py's FreyaTtsKo docstring)."
            )
        return str(venv_python)

    def _ensure_started(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        self._proc = subprocess.Popen(
            [
                self._interpreter(),
                str(FREYATTS_SERVER_SCRIPT),
                "--model-dir",
                str(self.model_dir),
                "--device",
                self.device,
                "--steps",
                str(self.steps),
                "--seed",
                str(self.seed),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # bufsize=1: line-buffer *our* end of the pipes (Python-side);
            # scripts/_freyatts_server.py flushes explicitly after every
            # line it writes, so this is enough to see each response promptly.
            bufsize=1,
            # General hygiene, see clean_subprocess_env()'s docstring -- not
            # the fix for the deadlock avoided below by StderrDrainer.
            env=clean_subprocess_env(),
        )
        # Both readers MUST start immediately, before the first get_line()
        # call below -- see LineReader/StderrDrainer's docstrings in
        # _procio.py for the two different (and differently-shaped) hangs
        # this avoids.
        self._stdout_reader = LineReader(self._proc.stdout)
        self._stderr_drainer = StderrDrainer(self._proc.stderr)

        try:
            ready = self._stdout_reader.get_line(self.startup_timeout_seconds)
        except TimeoutError as exc:
            self._proc.kill()
            raise RuntimeError(f"freyatts server didn't start in time: {exc}") from exc
        if ready.strip() != "---READY---":
            self._proc.kill()
            raise RuntimeError(
                f"freyatts server failed to start ({ready!r}):\n{self._stderr_drainer.tail()[-2000:]}"
            )
        return self._proc

    def synthesize(self, text: str, out_path: str) -> str:
        """Synthesize `text` (Korean) to a 48kHz mono wav at `out_path`."""
        proc = self._ensure_started()
        proc.stdin.write(json.dumps({"text": text, "out": out_path}) + "\n")
        proc.stdin.flush()

        try:
            line = self._stdout_reader.get_line(self.request_timeout_seconds)
        except TimeoutError as exc:
            raise RuntimeError(f"freyatts server request timed out: {exc}") from exc
        if not line:
            raise RuntimeError(
                f"freyatts server exited unexpectedly:\n{self._stderr_drainer.tail()[-2000:]}"
            )

        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"freyatts server: {response.get('error')}")
        return out_path

    def synthesize_audio(self, text: str) -> tuple[np.ndarray, int]:
        """Mono float32 samples + sample rate (see _synthesize_to_array)."""
        return _synthesize_to_array(self, text)

    def close(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


DEFAULT_MATCHA_DIR = PROJECT_ROOT / "models" / "sherpa-matcha-en"
DEFAULT_MATCHA_ACOUSTIC_MODEL = DEFAULT_MATCHA_DIR / "model-steps-3.onnx"
DEFAULT_MATCHA_VOCODER = DEFAULT_MATCHA_DIR / "vocos-22khz-univ.onnx"
DEFAULT_MATCHA_TOKENS = DEFAULT_MATCHA_DIR / "tokens.txt"
DEFAULT_MATCHA_DATA_DIR = DEFAULT_MATCHA_DIR / "espeak-ng-data"


@dataclass
class SherpaMatchaTts:
    """TTS candidate: sherpa-onnx's Matcha-TTS support (flow-matching,
    non-autoregressive -- same general shape as SenseVoice's speed profile).

    k2-fsa itself only ships English (en_US-ljspeech) and Chinese (zh-baker)
    Matcha-TTS checkpoints (confirmed by hand against every sherpa-onnx
    release, 1.12.19 through 1.13.2) -- no official Korean one.

    The sherpa-matcha-ko preset points at **this project's own Korean acoustic
    model**: trained in-house, with the voice design and the training corpus
    generated using Qwen3-TTS. `matcha-ko-voiceA-ep499-steps10` reads as voice A,
    epoch 499, 10 flow-matching steps. Written down here because it was not
    recorded anywhere and the ambiguous phrase "community/custom-trained" led a
    later reader (see the Supertonic docstring below) to conclude the checkpoint
    was of unknown origin and therefore a liability. It is the opposite: it is the
    one stage whose weights this project controls outright.

    The practical consequence worth stating is distribution, not licensing: it
    cannot be downloaded by any setup script because it has not been published
    anywhere, not because its origin is doubtful -- so a fresh machine needs it
    copied in by hand (see scripts/setup_windows.ps1's warning).

    Licensing is settled. The weights are ours, and using Qwen3-TTS output as
    training data was confirmed permissible by the project owner (2026-08-18).
    So unlike every third-party candidate evaluated against it, this preset has no
    outstanding licence question and no downstream use restrictions -- which is
    the opposite of what an earlier version of this file assumed.

    It also means this preset is improvable rather than merely replaceable, which
    is what docs/tts-conversational-build-design.md's Path A is about.

    Historical note: earlier text here inferred the lineage from ONNX metadata
    (`maintainer: freyatts-ko`) and read it as a third-party community model (see
    models/sherpa-matcha-ko/, `maintainer: freyatts-ko` in its own ONNX
    metadata -- the same lineage as this project's freyatts-ko-voicea TTS
    preset). That checkpoint ships as *just* the acoustic model, no
    tokens.txt/vocoder/espeak-ng-data of its own -- confirmed via its ONNX
    metadata (`comment: "Korean Matcha-TTS, espeak-ng ko phonemes, icefall
    tokens.txt ids"`, `sample_rate: 22050`) that it uses the SAME espeak-ng
    phoneme scheme and mel spec as the English checkpoint, so the
    sherpa-matcha-ko preset reuses the English preset's tokens.txt/vocoder/
    data_dir rather than needing its own copies. espeak-ng-data itself is a
    multi-language phoneme dictionary package (not English-specific despite
    living under models/sherpa-matcha-en/) -- it already bundles Korean
    (espeak-ng-data/lang/ko, ko_dict).

    espeak-ng is load-bearing here, not incidental, and that has two
    consequences worth knowing before anyone edits the preset (measured
    2026-08-19, docs/output/research-delta-20260818.md §13):

    1. Removing data_dir kills the process, and not via an exception. Point it
       at a missing directory and espeak-ng first falls back to its hardcoded
       /usr/share/espeak-ng-data, then aborts at the C level -- rc=1, no Python
       traceback, nothing for try/except to catch. The voice agent does not
       degrade to a silent TTS; it exits. (On a Linux target that fallback is
       itself a hazard: a system-installed espeak-ng would be used instead of
       ours, with whatever phoneme tables its version ships.)
    2. tokens.txt is the espeak IPA inventory, not Hangul or jamo -- 159 tokens
       whose first 14 are byte-identical to the English preset's and whose tail
       is IPA diacritics. So when sherpa-onnx 2.0.0 removes espeak-ng
       (k2-fsa/sherpa-onnx#3731) this preset stops working until someone
       supplies a Korean lexicon.txt or an external phonemizer emitting exactly
       these phonemes. That is one of the reasons the sherpa-onnx version is
       pinned; it is not a reason to switch away from this model, which we own
       and can retrain.

    Each of the four model files is its own independently-overridable `Path`
    field (matching e.g. VibeAsrBitnet's vae_model/lm_model in asr.py) rather
    than a shared model_dir + filenames, specifically so a preset's acoustic
    model and its (shared, reused) vocoder/tokens/data_dir can live in
    different directories without any path-juggling in this class.

    Unlike MOSS-TTS-Nano/FreyaTTS, this needs no subprocess or isolated venv
    at all: sherpa_onnx is already an in-process dependency of this project
    (see asr.py's NobodyASR), and its Matcha-TTS support is just another
    config on the same `sherpa_onnx.OfflineTts` API family as
    `OfflineRecognizer` -- so this class loads the model once in
    __post_init__ and calls it directly, no LineReader/StderrDrainer/warm
    server machinery needed.
    """

    acoustic_model: Path = DEFAULT_MATCHA_ACOUSTIC_MODEL
    vocoder: Path = DEFAULT_MATCHA_VOCODER
    tokens: Path = DEFAULT_MATCHA_TOKENS
    data_dir: Path = DEFAULT_MATCHA_DATA_DIR
    num_threads: int = 2
    # CPU-only, explicitly (sherpa_onnx's own default is also "cpu", but
    # spelling it out here means that stays true even if upstream ever
    # changes its default -- this project has no CUDA-capable onnxruntime
    # build set up for sherpa_onnx anyway, see asr.py's sense-voice-small,
    # which is CPU for the same reason).
    provider: str = "cpu"
    speaker_id: int = 0
    speed: float = 1.0

    def __post_init__(self):
        matcha_config = sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model=str(self.acoustic_model),
            vocoder=str(self.vocoder),
            tokens=str(self.tokens),
            data_dir=str(self.data_dir),
        )
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            matcha=matcha_config, num_threads=self.num_threads, provider=self.provider
        )
        self.tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model_config))

    def synthesize(self, text: str, out_path: str) -> str:
        """Synthesize `text` (English only -- see class docstring) to a wav
        at `out_path`, at whatever sample rate the acoustic model/vocoder
        pair uses (22050Hz for vocos-22khz-univ.onnx)."""
        samples, sr = self.synthesize_audio(text)
        sf.write(out_path, samples, samplerate=sr)
        return out_path

    def synthesize_audio(self, text: str) -> tuple[np.ndarray, int]:
        """In-process native path -- no file round-trip (see
        _synthesize_to_array). sherpa_onnx returns mono float32 samples
        directly, which is exactly what the streaming playback queue wants."""
        audio = self.tts.generate(text, sid=self.speaker_id, speed=self.speed)
        return np.ascontiguousarray(audio.samples, dtype=np.float32), int(audio.sample_rate)


DEFAULT_SUPERTONIC_DIR = PROJECT_ROOT / "models" / "sherpa-supertonic-3"


@dataclass
class SherpaSupertonicTts:
    """TTS candidate: Supertone's Supertonic 3 via sherpa-onnx.

    Reaches this project through the same `sherpa_onnx.OfflineTts` API family as
    SherpaMatchaTts, so it needs no subprocess, no isolated venv, and no
    dependency bump -- `OfflineTtsSupertonicModelConfig` is already present in
    the pinned sherpa-onnx 1.13.4 (verified by introspection, not by release
    notes: the `unicode_indexer` and `voice_style` fields are the Supertonic 3
    signature).

    ## Why this is interesting: no G2P

    Every other Korean TTS route in this project runs into the Korean
    grapheme-to-phoneme problem, and the survey of that landscape is bleak --
    the g2pK family all import mecab in __init__, KoG2P and KoNLPy are GPL,
    KoNLPy additionally wants a JVM, and pynini (so NeMo) publishes no aarch64
    wheel. SherpaMatchaTts sidesteps it only by borrowing the English preset's
    espeak-ng-data, which this project would rather not depend on (GPL-3.0 plus
    a C dependency, and weak Korean rules).

    What this does NOT buy is a smaller license surface, and an earlier version
    of this docstring implied otherwise. espeak-ng is statically linked into the
    sherpa-onnx wheel we install -- sherpa-onnx-c-api.dll carries the whole
    espeak_ng_* API, piper-phonemize's phonemize_eSpeak symbol, and a hardcoded
    /usr/share/espeak-ng-data fallback path. SenseVoice ASR and TEN-VAD load
    that same DLL, so choosing this preset over SherpaMatchaTts drops the 18MB
    data directory and nothing else; the linked GPL-3.0 code stays. Upstream
    agrees the conflict is real and plans to remove espeak-ng in sherpa-onnx
    2.0.0 (k2-fsa/sherpa-onnx#3731); it has not shipped.

    The real advantage is forward compatibility: when 2.0.0 lands this class
    keeps working untouched and SherpaMatchaTts does not. See that class's
    docstring, and docs/output/research-delta-20260818.md §13.

    Supertonic needs none of it. The paper states the model "operates directly
    on raw character-level text and employs cross-attention for text-speech
    alignment, thus eliminating the need for grapheme-to-phoneme (G2P) modules
    and external aligners" (arXiv 2503.23108), and the shipped assets agree:
    there is no tokens.txt and no phoneme inventory, only `unicode_indexer.bin`.

    Correspondingly there is **no language parameter**. `tts.json` reports
    `n_langs: 0` and `lang_emb_dim: 0` for both the text encoder and the vector
    field, so the model has no language conditioning at all -- the language is
    implicit in the characters. Hangul in, Korean out. That also means the
    `--lang` flag in sherpa-onnx's CLI docs has no analogue here, and
    `generate()` accepts only (text, sid, speed).

    ## What it does NOT solve

    G2P-free is not text-normalization-free. Nothing in this path expands
    digits or Latin letters, so "3시" is undefined behaviour -- the model may
    read the character or skip it. That job still belongs to the caller (today,
    to persona.py's prompt instruction).

    ## Licensing -- read this before shipping

    The bundled `LICENSE` in the sherpa-onnx redistribution is **MIT, and that
    is the license of Supertone's sample *code*, not of these weights.** The
    upstream README shipped in the same directory says plainly: "The
    accompanying model is released under the OpenRAIL-M License." So the file
    sitting next to the .onnx files understates the obligations, which is
    exactly the trap docs/llm-conversational-selection.md warns about ("do not
    take a license from a quantizer's repo or from memory").

    OpenRAIL-M (BigScience, 2022-08-18) does permit commercial use royalty-free,
    but it carries use-based restrictions that **must be passed downstream as an
    enforceable provision**. Three are directly relevant to a voice companion:
    undisclosed machine-generated content requires a clear disclaimer;
    impersonation/deepfakes without consent are prohibited; and providing
    medical advice is prohibited. Adopting this preset is a product decision,
    not just a technical one.

    ## Measured, on this project's Windows CPU box (NOBODY_CPU_BUDGET=4)

    10 speakers, 44100 Hz. Intelligibility by ASR round-trip, 3 repeats of an
    8-sentence Korean set, median with the observed range (numbers excluded --
    SenseVoice's inverse text normalization writes "3시 20분" for a correctly
    spoken "세 시 이십 분", so it undoes the thing under test):

        preset             CER~   range          rtf~   rtfMx
        sherpa-matcha-ko   0.043  0.021-0.053    0.27   0.32
        supertonic-3-ko    0.064  0.064-0.074    0.47   0.60
        supertonic-2-ko    0.074  0.064-0.117    0.16   0.21

    Repeats matter here and one run would have misled: synthesis is stochastic
    across processes, and an earlier single-sample reading of this same comparison
    put both presets at 0.074 and called it a tie. With ranges, matcha-ko's does
    not overlap either Supertonic -- so **the incumbent is more intelligible, and
    that difference is real**, while this preset is ~1.7x slower than it.

    Supertonic **2** is the interesting one on speed: 2.9x faster than v3 and 1.7x
    faster than matcha-ko, because v3 raised the flow-matching steps from 5 to 8
    and that step count is not exposed through sherpa-onnx (checked:
    OfflineTtsSupertonicModelConfig has no such field, nor does tts.json). See the
    supertonic-2-ko preset. It does not change the recommendation -- matcha-ko is
    already inside budget at 0.27 -- but it is the option if a slower board ever
    makes TTS the binding constraint rather than the LLM.

    Its one real advantage over sherpa-matcha-ko is voice choice: ten speakers
    instead of one, which matters because this module's DEFAULT reference voice is
    flagged as a placeholder to replace before anything user-facing.

    An earlier version of this docstring also claimed an advantage in provenance,
    on the grounds that the Korean Matcha checkpoint is an unattributable
    community model. That was wrong, and the correction runs the other way:
    sherpa-matcha-ko is **this project's own trained model** (see
    SherpaMatchaTts). Owning the weights is a stronger position than borrowing
    permissively-licensed ones -- it means the model can be retrained rather than
    merely swapped, and it carries no third-party use restrictions, which
    Supertonic's OpenRAIL-M weights do.
    """

    duration_predictor: Path = DEFAULT_SUPERTONIC_DIR / "duration_predictor.int8.onnx"
    text_encoder: Path = DEFAULT_SUPERTONIC_DIR / "text_encoder.int8.onnx"
    vector_estimator: Path = DEFAULT_SUPERTONIC_DIR / "vector_estimator.int8.onnx"
    vocoder: Path = DEFAULT_SUPERTONIC_DIR / "vocoder.int8.onnx"
    tts_json: Path = DEFAULT_SUPERTONIC_DIR / "tts.json"
    unicode_indexer: Path = DEFAULT_SUPERTONIC_DIR / "unicode_indexer.bin"
    voice_style: Path = DEFAULT_SUPERTONIC_DIR / "voice.bin"
    num_threads: int = 2
    # CPU-only for the same reason as SherpaMatchaTts: there is no CUDA-capable
    # onnxruntime build wired up for sherpa_onnx in this project.
    provider: str = "cpu"
    # Not 0. sid=0 measured worse than sid=7 on the ASR round-trip, and picking
    # the first index by default would have quietly shipped a mid-table voice.
    speaker_id: int = 7
    speed: float = 1.0

    def __post_init__(self):
        supertonic_config = sherpa_onnx.OfflineTtsSupertonicModelConfig(
            duration_predictor=str(self.duration_predictor),
            text_encoder=str(self.text_encoder),
            vector_estimator=str(self.vector_estimator),
            vocoder=str(self.vocoder),
            tts_json=str(self.tts_json),
            unicode_indexer=str(self.unicode_indexer),
            voice_style=str(self.voice_style),
        )
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            supertonic=supertonic_config,
            num_threads=self.num_threads,
            provider=self.provider,
        )
        self.tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model_config))

    def synthesize(self, text: str, out_path: str) -> str:
        """Synthesize `text` to a wav at `out_path`, at the model's own 44100Hz.

        Note the rate: this is the highest-rate TTS in the project (Matcha is
        22050), so the streaming playback path resamples more per chunk. See
        audio/resample.py -- linear, deliberately.
        """
        samples, sr = self.synthesize_audio(text)
        sf.write(out_path, samples, samplerate=sr)
        return out_path

    def synthesize_audio(self, text: str) -> tuple[np.ndarray, int]:
        """In-process native path, mono float32 -- same contract as every other
        TTS class here, so the sentence-chunk playback queue needs no special
        case for this preset."""
        audio = self.tts.generate(text, sid=self.speaker_id, speed=self.speed)
        return np.ascontiguousarray(audio.samples, dtype=np.float32), int(audio.sample_rate)
