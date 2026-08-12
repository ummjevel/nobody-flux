# nobody-flux

온디바이스 ASR → LLM → TTS 파이프라인 프로토타입. 목표는 클라우드 STS(Gemini Live/OpenAI
Realtime류)에 의존하지 않는, 완전 로컬로 도는 음성 대화 파트너("퀜")를 만드는 것.
최종 타깃은 CM4급 저사양 온디바이스 하드웨어라, 개발은 GPU 머신(RTX 5090 / H100)에서 하되
**성능 판단은 CPU 기준**으로 한다.

리서치 배경은 [`docs/output/ondevice_asr_llm_tts_research_20260716.md`](docs/output/ondevice_asr_llm_tts_research_20260716.md),
지금 구현된 기능/범위와 **무엇이 실측이고 무엇이 추정치인지**는
[`docs/FEATURES.md`](docs/FEATURES.md), 기억(개인화) 설계는
[`docs/memory-design.md`](docs/memory-design.md), 끼어들기 설계는
[`docs/barge-in-design.md`](docs/barge-in-design.md) 참고.

## 아키텍처

```
   마이크 (연속 캡처, 전용 스레드)                              스피커
        │                                                       ▲
        ▼                                                       │
  ┌───────────┐                                          ┌────────────┐
  │  turn/    │  발화 경계·끼어들기·상태(IDLE/LISTENING/    │  audio/    │
  │ controller│  RESPONDING)                              │  player    │
  └─────┬─────┘                                          └─────▲──────┘
        │ 완료된 턴 (queue)          취소 (event) ─────────┐     │ 문장 청크
        ▼                                                │     │
  ┌──────────┐  text   ┌──────────┐  토큰 스트림  ┌──────────┐  │
  │  stage/  │ ──────▶ │  stage/  │ ───────────▶ │  stage/  │──┘
  │   asr    │         │   llm    │              │   tts    │
  └──────────┘         └──────────┘              └──────────┘
   SenseVoice           Qwen3-0.6B                Matcha-TTS
   (또는 stage/asr_stream: 말하는 동안 실시간 인식)
```

핵심 두 가지:

- **캡처가 멈추지 않는다.** 응답을 생성하는 동안에도 마이크를 계속 읽으므로, 생성 중 끼어들기가
  *관측된다*. 예전엔 그 구간 내내 아무도 마이크를 읽지 않아서 끼어들기가 유실됐다.
- **응답이 문장 단위로 흐른다.** LLM 토큰을 문장으로 잘라 나오는 대로 TTS→재생하므로 첫 음성까지가
  `ASR + 첫 문장 LLM + 첫 문장 TTS`로 줄어든다.

각 스테이지가 뭘 쓰는지는 코드가 아니라 `configs/models.yaml`이 결정한다 (아래 "모델 스왑" 참고).
모든 대화는 `data/conversations.db` (SQLite)에 기록된다.

## 빠른 시작

### 1. 환경 세팅

```bash
./scripts/setup_local.sh          # 로컬 (RTX 5090)
./scripts/setup_server.sh         # 서버 (H100)
bash scripts/setup_mac.sh         # macOS — CPU/ONNX 기본 파이프라인만
```

```powershell
# 네이티브 윈도우 — CPU 전용, 실기 마이크 측정용
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

`setup_local.sh`/`setup_server.sh`는 `scripts/setup_common.sh`를 공유한다: `uv sync`, GPU 인식 확인,
모델 자산 다운로드, MOSS-TTS-Nano/FreyaTTS의 **독립 venv** 생성, VibeASR.cpp 빌드.

**macOS**: CPU/ONNX 기본 프리셋만. CUDA 전용 프리셋과 VibeASR.cpp 빌드는 건너뛴다. 기본 LLM은
Apple Silicon에서 Metal 가속되고(GGUF), raw-transformers 프리셋은 MPS를 자동 사용한다.
보통 `source scripts/env.sh` 불필요.

**윈도우**: `.venv-win`에 CPU 전용으로 세팅한다(`requirements/windows-cpu.txt`). VibeASR.cpp와
PyTorch 격리 venv 두 개는 제외 — MSVC/cmake 툴체인을 요구하지 않기 위해서다.
**이 환경이 존재하는 이유**: WSL2는 WSLg 오디오 브리지로만 마이크에 닿아 캡처가 불안정하고
버퍼링이 하드웨어의 것이 아니라서, 턴테이킹 파라미터를 전부 추정치로 둘 수밖에 없었다.
네이티브 윈도우는 WASAPI로 실제 장치를 열거하므로 드디어 측정이 된다.

TTS voice-clone 참조 음성은 저장소에 없다(개인 음성이라 커밋 안 함, `.gitignore` 참고). 필요한
프리셋을 쓸 때만 직접 준비:

- **여러 목소리 중 선택**(`--voice`): `configs/voices.yaml`에 등록된 이름(`male-1`, `male-2`,
  `female-1`, `female-2`)에 맞춰 `data/voices/{male_1,...}.wav`를 두면 `--voice male-1` 식으로 선택.
- **기본값 하나만**: `data/reference_voice_16k.wav` 하나만 둬도 동작.

포맷은 mono wav, 16kHz 권장. 기본 TTS 프리셋(`sherpa-matcha-ko`)은 voice-clone이 아니라서 불필요.

### 2. 대화하기

```bash
source scripts/env.sh   # Linux 전용: sherpa-onnx가 필요로 하는 LD_LIBRARY_PATH, uv run 전에 필수

uv run python scripts/talk.py                    # 연속 음성 루프
uv run python scripts/talk.py --streaming-asr    # 말하는 동안 인식 (ASR을 임계경로에서 제거)
uv run python scripts/talk.py --aec auto         # 단일 duplex 스트림 + 에코 제거
uv run python scripts/run_pipeline.py --wav-in in.wav --wav-out out.wav   # 1회성
```

윈도우에선 `.venv-win\Scripts\python.exe scripts\talk.py --streaming-asr` (env.sh 불필요).

주요 플래그:

| 플래그 | 하는 일 |
|---|---|
| `--streaming-asr` | 프레임이 도착하는 대로 인식(Phase 3). 말이 끝나면 인식도 끝나 있다. |
| `--aec auto\|refgate\|speex\|os\|vpio` | 캡처+재생을 **하나의** duplex 스트림으로. 에코 제거 + macOS err-50 회피. |
| `--endpoint-detect` | Smart Turn v3로 "문장 중간 멈춤"과 "말 끝남"을 구분해 자연스러운 멈춤이 잘리는 걸 완화. |
| `--no-barge-in` | 끼어들기로 응답을 취소하지 않음(캡처는 계속 → 발화는 유실되지 않음). |

`scripts/env.sh`가 Linux에서 필요한 이유: sherpa-onnx의 컴파일된 확장이 버전 없는
`libonnxruntime.so`를 `dlopen()`하는데 pip wheel은 버전 붙은 것만 준다. glibc 동적 링커는 프로세스
시작 시 한 번만 `LD_LIBRARY_PATH`를 읽으므로 실행 **전에** 소싱해야 한다(파이썬 안에서
`os.environ`으로 설정하면 이미 늦다). 윈도우/macOS는 `platform_support.py`가 프로세스 안에서
처리하므로 불필요.

### 3. 마이크 캘리브레이션 (새 기기에선 필수)

```powershell
.venv-win\Scripts\python.exe scripts\_calibrate_vad_threshold.py --apply
```

`configs/vad.yaml`의 `threshold`는 **기기마다 다시 재야 한다.** 이 스크립트는 실제 방의 노이즈
플로어를 녹음해(조용히 있으면 된다) 임계값을 정한다. 양쪽 실패 모드가 모두 실재한다 — 너무 높으면
발화를 놓치고, **너무 낮으면 무음을 발화로 읽어 턴이 영원히 끝나지 않는다**(즉 응답이 아예 안 나온다).

`--aec`를 쓸 거면 `scripts/_calibrate_aec_delay.py`로 스피커→마이크 지연도 함께 잰다.

### 4. 모델 스왑

```bash
uv run python scripts/run_pipeline.py --wav-in in.wav --wav-out out.wav \
    --asr sense-voice-small --llm qwen3-0.6b-gguf --tts sherpa-matcha-ko
```

| 스테이지 | 기본값 | 다른 프리셋 |
|---|---|---|
| ASR | `sense-voice-small` | `streaming-zipformer-ko`, `vibeasr-bitnet` |
| LLM | `midm-2.3b-gguf` | `qwen3-1.7b-gguf`, `qwen3-0.6b-gguf`, `kanana-2.1b-gguf`\*, `exaone-2.4b-gguf`\* |
| TTS | `sherpa-matcha-ko` | `sherpa-matcha-en`, `freyatts-ko-voicea`, `moss-tts-nano` |

\* 비상업 라이선스 — 성능 참고용. LLM 선정 근거와 라이선스 분석은
[`docs/llm-conversational-selection.md`](docs/llm-conversational-selection.md),
모델 비교는 `scripts/_ab_persona.py`(페르소나 준수 + 대화 지속성 + 레이턴시).

프리셋은 `configs/models.yaml`, 목소리는 `configs/voices.yaml`(목소리는 별도 축이 아니라 TTS
스테이지의 파라미터라 따로 관리). 새 모델을 붙이는 절차는 `docs/FEATURES.md`의
"프리셋 추가하는 법" 참고.

### 5. 대화 기록 확인

```bash
sqlite3 data/conversations.db "SELECT turn_index, user_text, reply_text, asr_ms, llm_ms, tts_ms FROM turns ORDER BY id DESC LIMIT 10;"
```

## 프로젝트 구조

```
src/nobody_flux/
  stage/          # 모델을 소유
    asr.py llm.py tts.py     # 배치 스테이지 (프리셋)
    asr_stream.py            # 라이브 스트리밍 인식 (LocalAgreement)
  turn/           # "언제 말할지"를 소유
    vad.py                   # TEN-VAD 상태기계 (VadStream: 프레임 in, 이벤트 out)
    detector.py              # Smart Turn v3 엔드포인트
    backchannel.py           # "어"/"응" 어휘 판정
    controller.py            # 연속 캡처 + 3-상태 턴 + 끼어들기
  audio/          # 디바이스를 소유
    session.py               # duplex 스트림 (캡처+재생 한 소유자)
    aec.py player.py resample.py
  pipeline.py     # 스테이지 오케스트레이션 + 스테이지별 계측 (run / run_streaming)
  registry.py     # configs/*.yaml → 객체
  persona.py memory.py storage.py textchunk.py platform_support.py
scripts/
  talk.py                    # 연속 음성 루프
  run_pipeline.py            # 1회성 wav-in/wav-out
  benchmark.py               # 프리셋 조합별 latency 표
  _smoke_imports.py _smoke_turn.py _smoke_duplex.py     # 검증
  _calibrate_vad_threshold.py _calibrate_aec_delay.py _calibrate_turn_params.py
  _debug_vad_mic.py _debug_segment.py _debug_silence.py _debug_roomtone.py
  setup_{local,server,mac,common}.sh  setup_windows.ps1  env.sh
configs/          # models, voices, vad, audio, turn_detector, streaming_asr
docs/             # 리서치, 기능 정의, 기억/barge-in/TTS 설계
```

## 알려진 제약

- **VAD 임계값은 기기 의존적이다.** 위 "마이크 캘리브레이션" 참고. 복사해 오면 안 되는 값이다.
- **디지털 무음(`np.zeros`)은 TEN-VAD 입력으로 쓰면 안 된다.** 특징 추출이 퇴화해 모델이 무음
  구간 내내 발화라고 답한다(측정 확인). 패딩은 실제 노이즈로.
- **WSL2 마이크**: 실시간 루프는 WSLg 패스스루가 불안정해 보장 못 한다. 네이티브 윈도우
  (`.venv-win`)나 macOS를 쓰거나, `run_pipeline.py`로 사전 녹음 wav를 써서 로직만 검증할 것.
- **AEC는 아직 실제로 시험되지 않았다.** 듀플렉스 경로와 "자기 자신에게 끼어들지 않는지"는
  실기에서 확인됐지만(`_smoke_duplex.py`), 현재 측정 셋업이 헤드폰이라 취소할 에코가 거의 없다.
  스피커+한 통 구성(=CM4)에서 다시 확인 필요.
- **MOSS-TTS-Nano는 별도 venv**: `torch==2.7.0`을 정확히 요구해서 이 프로젝트 venv에 같이 깔면
  `uv sync`가 둘 중 하나를 깨뜨린다(실제로 한 번 발생 — `uv sync`가 수동 설치된 moss-tts-nano를
  조용히 삭제). `external/MOSS-TTS-Nano/.venv`를 따로 두고 서브프로세스로 호출한다. 그 venv엔
  `soundfile`도 별도 설치가 필요하다(MOSS-TTS-Nano의 `pyproject.toml`이 누락 — `setup_common.sh`가 처리).
- **drvfs(WSL2, `/mnt/c/...`)에서 `uv sync`**: 기본 하드링크 설치가 일부 파일을 조용히 누락시키는
  걸 확인했다(torch는 임포트되는데 `libcudnn.so.9`가 없어서 크래시). `pyproject.toml`의
  `[tool.uv] link-mode = "copy"`로 고정해뒀다.
