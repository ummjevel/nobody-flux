# nobody-flux

온디바이스 ASR → LLM → TTS 파이프라인 프로토타입. 목표는 클라우드 STS(Gemini Live/OpenAI
Realtime류)에 의존하지 않는, 완전 로컬로 도는 음성 대화 파트너("퀜")를 만드는 것.
지금은 GPU가 있는 개발 머신(RTX 5090 로컬 / H100 서버)에서 파이프라인 구조와 모델 선택을
검증하는 단계고, 최종 타깃은 CM4급 저사양 온디바이스 하드웨어.

리서치 배경은 [`docs/output/ondevice_asr_llm_tts_research_20260716.md`](docs/output/ondevice_asr_llm_tts_research_20260716.md),
지금 구현된 기능/범위는 [`docs/FEATURES.md`](docs/FEATURES.md), 향후 개인화(기억) 설계는
[`docs/memory-design.md`](docs/memory-design.md) 참고.

## 아키텍처

```
 마이크 입력                                                    스피커 출력
     │                                                              ▲
     ▼                                                              │
┌─────────┐   wav    ┌─────────┐   text   ┌─────────┐   text  ┌─────────┐
│  VAD    │ ───────▶ │   ASR   │ ───────▶ │   LLM   │ ──────▶ │   TTS   │
│(vad.py) │  발화 감지 │(asr.py) │ SenseVoice │(llm.py) │Qwen3-0.6B│(tts.py) │MOSS-TTS-Nano
└─────────┘          └─────────┘          └─────────┘         └─────────┘
```

각 스테이지가 뭘 쓰는지는 코드가 아니라 `configs/models.yaml`이 결정한다 (아래 "모델 스왑" 참고).
모든 대화는 `data/conversations.db` (SQLite)에 기록된다.

## 빠른 시작

### 1. 환경 세팅

```bash
# 로컬 (RTX 5090)
./scripts/setup_local.sh

# 서버 (H100)
./scripts/setup_server.sh

# macOS (Apple Silicon/Intel) — CPU/ONNX 기본 파이프라인만
bash scripts/setup_mac.sh
```

`setup_local.sh`/`setup_server.sh`는 `scripts/setup_common.sh`를 공유하며 다음을 처리한다:
`uv sync`, GPU 인식 확인, SenseVoice ASR 모델 자산 다운로드, MOSS-TTS-Nano를 `external/`에
클론하고 **독립된 venv**를 만들어줌 (이유는 아래 "알려진 제약" 참고).

**macOS**: `scripts/setup_mac.sh`는 CPU/ONNX 기본 프리셋(sense-voice-small / qwen3-0.6b-gguf /
sherpa-matcha-ko)만 세팅한다. CUDA 전용 프리셋(freyatts-ko-voicea, moss-tts-nano)과 VibeASR.cpp
빌드는 건너뛴다. 기본 LLM은 Apple Silicon에서 Metal 가속되고(GGUF), raw-transformers 프리셋은
MPS를 자동 사용한다. 마이크(`sounddevice`)가 macOS에선 네이티브로 동작해서 **WSL2에서 못 하던
실시간 barge-in/VAD/endpoint 테스트를 실제로 할 수 있다.** (아직 실기 검증은 안 됨 — 의존성 휠
지원 기반 예측.) macOS에선 보통 `source scripts/env.sh` 불필요(안 되면 그때만).

TTS의 voice-clone 참조 음성이 필요하다 — 저장소에 포함돼 있지 않으니 직접 준비해서 둘 것
(개인 음성 녹음이라 공개 레포에 커밋 안 함, `.gitignore` 참고). 두 가지 방법:

- **여러 목소리 중 선택** (`--voice` 플래그): `configs/voices.yaml`에 등록된 이름(`male-1`,
  `male-2`, `female-1`, `female-2`)에 맞춰 `data/voices/{male_1,male_2,female_1,female_2}.wav`에
  파일을 두면 `--voice male-1` 식으로 선택 가능. 없는 이름을 쓰거나 파일을 안 두면
  `resolve_voice()`가 바로 에러를 내서 어디에 뭘 둬야 하는지 알려준다.
- **기본값 하나만**: `data/reference_voice_16k.wav`에 파일 하나만 두면 `--voice` 없이도
  동작 (모든 프리셋의 기본 참조 음성).

포맷은 mono wav, 16kHz 권장.

### 2. 대화하기

```bash
source scripts/env.sh   # sherpa-onnx가 필요로 하는 LD_LIBRARY_PATH 세팅, uv run 전에 필수

# 연속 음성 루프 (마이크로 계속 대화, VAD가 자동으로 턴을 끊음)
uv run python scripts/talk.py

# 1회성 wav-in/wav-out (자동화 테스트, 프리셋 비교용)
uv run python scripts/run_pipeline.py --wav-in in.wav --wav-out out.wav
```

`scripts/env.sh`가 필요한 이유: sherpa-onnx의 컴파일된 확장이 버전 없는
`libonnxruntime.so`를 `dlopen()`하는데, onnxruntime pip wheel은 버전이 붙은
`libonnxruntime.so.1.27.0`만 제공한다. glibc 동적 링커는 프로세스 시작 시 한 번만
`LD_LIBRARY_PATH`를 읽으므로, 셸에서 `python`/`uv run` 실행 전에 미리 소싱해야 한다
(파이썬 프로세스 안에서 `os.environ`으로 설정해도 이미 늦음).

### 3. 모델 스왑

```bash
uv run python scripts/run_pipeline.py --wav-in in.wav --wav-out out.wav \
    --asr sense-voice-small --llm qwen3-0.6b --tts moss-tts-nano --voice female-2
```

ASR/LLM/TTS 프리셋은 `configs/models.yaml`, 목소리는 `configs/voices.yaml`에 등록돼 있다
(목소리는 별도 프리셋 축이 아니라 TTS 스테이지의 파라미터 하나라 따로 관리됨). ASR은
`sense-voice-small`/`vibeasr-bitnet` 두 프리셋이 있고(`--asr vibeasr-bitnet`으로 전환),
TTS도 `moss-tts-nano`/`freyatts-ko-voicea` 두 프리셋이 있다(기본값은 `freyatts-ko-voicea`).
LLM은 아직 1개뿐이지만 새 후보 모델을 붙이는 구조는 갖춰져 있다 — 자세한 절차는
`docs/FEATURES.md`의 "프리셋 추가하는 법" 참고.

### 4. 대화 기록 확인

```bash
sqlite3 data/conversations.db "SELECT turn_index, user_text, reply_text, asr_ms, llm_ms, tts_ms FROM turns ORDER BY id DESC LIMIT 10;"
```

## 프로젝트 구조

```
src/nobody_flux/
  asr.py, llm.py, tts.py   # 각 스테이지 구현 (NobodyASR/NobodyLLM/NobodyTTS)
  persona.py               # 시스템 프롬프트 ("퀜" 페르소나)
  pipeline.py               # ASR→LLM→TTS 오케스트레이션 + 스테이지별 소요시간 계측
  registry.py               # configs/{models,voices}.yaml → 프리셋/음성 인스턴스 생성
  vad.py                    # TEN-VAD 기반 발화 구간 검출
  storage.py                 # SQLite 대화 저장 (sessions/turns/memories)
scripts/
  talk.py                   # 연속 음성 루프 (마이크/스피커, 상시 프로세스, 끼어들기 지원)
  run_pipeline.py            # 1회성 wav-in/wav-out CLI
  benchmark.py                # 고정 테스트셋으로 프리셋 조합별 latency 표 뽑기
  setup_local.sh / setup_server.sh / setup_common.sh
  env.sh                     # LD_LIBRARY_PATH 세팅 (source 필수)
configs/models.yaml         # ASR/LLM/TTS 프리셋 정의
configs/voices.yaml          # TTS 참조 음성(voice-clone) 프리셋 정의
configs/vad.yaml             # VAD(TEN-VAD) 튜닝 파라미터
docs/                        # 리서치, 기능 정의, 기억 설계 문서
```

## 알려진 제약

- **WSL2 마이크**: `scripts/talk.py`의 실시간 마이크 루프는 WSL2의 오디오 패스스루가
  불안정할 수 있어 로컬 개발 머신에서 동작을 보장 못 한다. 안 되면 H100 서버(네이티브 Linux)
  에서 테스트하거나, `run_pipeline.py`로 사전 녹음된 wav를 써서 로직만 검증할 것.
- **VAD 임계값**: `vad.py`의 침묵/발화 임계값은 고정 상수 기본값이다. 마이크·환경마다 재튜닝이
  필요할 가능성이 높다.
- **MOSS-TTS-Nano는 별도 venv**: `torch==2.7.0`을 정확히 요구해서, 이 프로젝트 자체 venv(더
  최신 torch를 쓰는 LLM 스테이지와 공유)에 같이 설치하면 `uv sync`가 둘 중 하나를 깨뜨린다
  (실제로 개발 중 한 번 발생 — `uv sync`가 수동 설치돼 있던 moss-tts-nano를 조용히
  삭제해버림). 그래서 `external/MOSS-TTS-Nano/.venv`를 따로 두고 `tts.py`가 그 인터프리터를
  직접 가리켜 서브프로세스로 호출한다. 그 venv엔 `soundfile`도 별도로 설치해야 한다 (MOSS-TTS-Nano
  자체 `pyproject.toml`이 `requirements.txt`와 다르게 이걸 누락하고 있어서, 안 깔면 torchaudio가
  오디오 백엔드를 못 찾고 실패한다 — `setup_common.sh`가 처리해둠).
- **drvfs(WSL2, `/mnt/c/...`)에서 `uv sync`**: 이 파일시스템에서 uv의 기본 하드링크 설치가
  일부 패키지 파일을 조용히 누락시키는 걸 확인했다 (torch는 임포트되지만 `libcudnn.so.9`가
  없어서 크래시하는 식 — dist-info엔 설치됐다고 나오는데 실제 파일이 없었음). `pyproject.toml`의
  `[tool.uv] link-mode = "copy"`로 고정해뒀다. 순수 Linux 파일시스템에서는 필요 없지만 더 느려질
  뿐 해는 없어서 그대로 둠.
- **기억(개인화) 미구현**: 저장 스키마만 있고 추출 로직은 없다 (`docs/memory-design.md`).
