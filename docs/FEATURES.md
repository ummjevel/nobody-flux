# 기능 정의

> `nobody-flux`가 지금 실제로 하는 일 / 안 하는 일 / 다음에 붙일 것. 세부 설계는 각 절이 링크하는
> 설계 문서 참고(기억 `memory-design.md`, barge-in `barge-in-design.md`, TTS 표현력
> `tts-expressivity-design.md` 외 3부작, 모델 리서치 `output/ondevice_asr_llm_tts_research_20260716.md`).

## 파이프라인 & 현재 기본값

캐스케이드 **ASR → LLM → TTS**. 어느 모델이 도는지는 코드가 아니라 `configs/models.yaml`이 결정
(`registry.py`가 로드). `run_pipeline.py`/`talk.py` 둘 다 `--asr/--llm/--tts <preset>`로 선택.

**스트리밍 출력** (`pipeline.run_streaming`, Phase 1): 응답을 통째로 만든 뒤 재생하는 대신, LLM을
토큰 스트리밍(`llm.reply_stream`)으로 받아 문장 단위로 자르고(`textchunk.SentenceChunker`) 나오는
대로 TTS(`tts.synthesize_audio`)해 재생 큐에 넣는다 → 첫 음성까지(ttfa)가 `asr+첫문장 llm+첫문장
tts`로 줄고, 이후 문장 합성이 앞 문장 재생과 겹친다. 실측(개발 박스, sherpa-matcha-ko 기본):
첫 오디오가 LLM 응답 완료보다 먼저 나옴. 배치 경로(`pipeline.run`)는 `run_pipeline.py`/`benchmark.py`
용으로 그대로 유지. 발화 불가 문자(이모지 등)는 합성 전 `sanitize_for_tts`가 걸러 빈 청크를 스킵.

| 스테이지 | 현재 기본값 | 구현 |
|---|---|---|
| ASR | `sense-voice-small` | `src/nobody_flux/stage/asr.py` |
| LLM | `qwen3-0.6b-gguf` | `src/nobody_flux/stage/llm.py` (페르소나 `persona.py` — "퀜", 20~30대 또래 반말) |
| TTS | `sherpa-matcha-ko` | `src/nobody_flux/stage/tts.py` (ONNX·CPU, CM4 타깃에 맞아 기본값으로 선택) |

### 소스 레이아웃

Phase 3~4에서 `src/nobody_flux/`를 역할별 패키지로 나눴다 (평평한 12개 모듈 → 3개 묶음).
경계는 "무엇을 소유하는가"로 그었다:

- **`stage/`** — 모델을 소유. `asr`·`llm`·`tts`(배치 프리셋) + `asr_stream`(Phase 3 라이브 인식).
- **`turn/`** — *언제* 말할지를 소유. `vad`(TEN-VAD 상태기계)·`detector`(Smart Turn)·
  `backchannel`(어휘 판정)·`controller`(Phase 4 3-상태 턴).
- **`audio/`** — 디바이스를 소유. `session`(듀플렉스 스트림)·`aec`·`player`·`resample`.
- 그 외 최상위: `pipeline`(스테이지 오케스트레이션)·`registry`(설정→객체)·`storage`·`memory`·
  `persona`·`textchunk`·`platform_support`(OS별 처리 일원화).

## 프리셋 (모델 스왑)

**ASR**
- `sense-voice-small` (기본) — SenseVoice-Small, sherpa-onnx, 한국어.
- `vibeasr-bitnet` — microsoft/VibeVoice-ASR-BitNet. CPU-only ggml/BitNet, ARM NEON 커널로 CM4 직결.
  상주 서버(`asr_stream_server`)로 웜 ~2초대.
- `streaming-zipformer-ko` — k2-fsa 스트리밍 Zipformer 한국어(Apache-2.0, int8). 두 가지 방식으로 씀:
  **배치 디코드**(이 프리셋, 다른 ASR과 같은 조건 비교용 — sense-voice와 속도 비슷 ~130ms, 내용
  정확하나 어절 스페이스 없음)와 **라이브 스트리밍**(Phase 3, `stage/asr_stream.py` —
  프리셋이 아니라 `configs/streaming_asr.yaml`, 아래 참고).

**LLM**
- `qwen3-0.6b-gguf` (기본) — Qwen3-0.6B GGUF Q4_K_M, llama-cpp-python. raw transformers 대비 CPU ~2배.
- `qwen3-0.6b` — 같은 weights, raw transformers.
- `lfm2-350m`/`lfm2-700m`/`lfm2-1.2b` — LiquidAI LFM2(엣지 특화). **벤치 결과: 이 페르소나(짧은 반말)엔
  드롭인 개선 아님** — 350m 마크다운 남발, 700m 반말 톤 좋으나 장황·느림(~2.7s vs qwen-gguf ~1.25s).
  qwen3-0.6b-gguf 기본 유지. (`configs/models.yaml` 주석에 상세)

**TTS**
- `sherpa-matcha-ko` (기본) — sherpa-onnx Matcha-TTS 한국어 커스텀 acoustic. ONNX·CPU·격리 venv 불필요
  → CM4 타깃에 가장 적합.
- `freyatts-ko-voicea` — FreyaTTS 한국어. 격리 venv 서브프로세스, CUDA 동작. 톤 자연스러우나 발음 약함.
- `moss-tts-nano` — voice-clone, 격리 venv. CPU 전용 느림(비교용).
- `sherpa-matcha-en` — 영어 Matcha(런타임 비교용).

프리셋 추가 방법은 맨 아래 "프리셋 추가하는 법".

## 대화 경험 (`scripts/talk.py`)

- **연속 음성 루프**: 상시 프로세스. 한 번 띄우면 `NobodyLLM` 히스토리가 세션 내내 유지 → 실제 멀티턴.
- **연속 캡처 + 3-상태 턴** (`turn/controller.py`, Phase 4): 캡처가 **전용 스레드에서 세션 내내**
  돈다. 이전에는 `listen_for_utterance()`(블로킹) → `produce()`(블로킹) 순서라 **LLM 생성 구간 내내
  마이크를 아무도 읽지 않았다** — 그 구간의 끼어들기는 "늦게 처리"된 게 아니라 아예 *관측되지 않았다*.
  이제 프레임을 계속 읽으므로 생성 중 barge-in이 잡히고, 응답 재생 중 시작된 발화도 이미 녹음 중이라
  다음 턴 첫 음절이 잘리지 않는다. 상태는 `IDLE`/`LISTENING`/`RESPONDING` 세 개뿐이고,
  barge-in 규칙 자체가 상태에 대한 문장("RESPONDING일 때만 응답을 취소한다")이라 명시적으로 뒀다.
  스레드 경계: 캡처 스레드가 VAD·스트리밍 인식기를 **단독 소유**, 완료된 턴은 `queue.Queue`로,
  취소는 `threading.Event`로 건넨다. 파이프라인·SQLite·LLM 히스토리는 메인 스레드 전용(락 불필요).
- **스트리밍 재생** (`audio/player.py`): 응답이 문장 청크로 스트리밍돼 순차 재생, barge-in 시 큐 전체를
  clip. `StreamPlayer`(전용 출력 스트림)와 `SessionPlayer`(듀플렉스 세션 경유) 둘 다 같은 인터페이스.
  청크마다 스트림을 여닫지 않고 하나를 유지 — 문장 경계마다 생기던 무음(=말 끝난 것처럼 들림) 제거.
- **턴 경계 = TEN-VAD** (`turn/vad.py`, sherpa_onnx 내장, `configs/vad.yaml`로 튜닝). 버튼 불필요.
  `VadStream`(프레임 push → 이벤트 yield)이 원시 API이고, 블로킹 `listen_for_utterance`는 그 위의
  편의 래퍼 — 이 분리가 Phase 4 연속 캡처를 가능하게 한 지점이다.
  **`threshold`는 기기별로 실측해야 한다**(`scripts/_calibrate_vad_threshold.py`) — 아래 참고.
- **끼어들기(barge-in)**: 재생 중 말하면 끊김. 생성 중이면 `pipeline.run_streaming`의
  `should_cancel`이 LLM 델타마다 폴링돼 그 자리에서 멈춘다(취소는 예외가 아니라 반환 — 부분 소비된
  LLM 스트림과 TTS 백엔드를 정의되지 않은 상태로 두지 않기 위해). 끊긴 응답은 히스토리에 남지 않는다.
- **에코 제거(AEC) / 듀플렉스** (`--aec`, `audio/session.py`·`audio/aec.py`, Phase 1.5): 기본은
  legacy(별도 마이크/스피커 스트림, AEC 없음). `--aec`를 주면 **단일 duplex `sd.Stream`**(캡처+재생
  한 소유자)으로 라우팅 → 응답 에코를 마이크에서 제거하고, macOS err-50 듀플렉스 충돌도 회피(그래서
  `--aec` 켜면 `--no-barge-in` 불필요). 백엔드: `refgate`(무의존 억제 게이트) / `speex`(진짜 AEC,
  `speexdsp` 필요) / `os`(Linux/CM4 module-echo-cancel) / `vpio`(macOS, 네이티브 바인딩 미구현—hook) /
  `auto`(플랫폼·라이브러리 자동선택, `configs/audio.yaml`). 지연 정렬은 `scripts/_calibrate_aec_delay.py`로
  실측해 `delay_frames`에 기록(윈도우 박스 실측: 28ms = 1프레임).
  - **디바이스 샘플레이트 협상**: 내부 계약은 16kHz 그대로 두되 스트림은 기기가 여는 레이트로 연다.
    WASAPI 공유 모드는 윈도우 믹스 포맷 레이트(측정 박스는 48kHz)로만 열리므로 16kHz 고정 duplex는
    `Invalid sample rate [-9997]`로 실패 — 즉 **윈도우에서 `--aec` 경로가 아예 열리지 않았다**.
    16kHz를 먼저 시도하고 안 되면 기기 기본값으로 열어 콜백 경계에서만 변환한다. 16kHz를 지원하지
    않는 USB 마이크(대다수) 전반에 해당.
  - **캡처 워밍업 폐기**(`audio.session.WARMUP_FRAMES`): 입력 스트림을 열면 트랜지언트가 나온다
    (윈도우 USB 마이크 실측: 첫 ~150ms가 rms 0.27·피크 풀스케일, 실제 바닥은 rms 0.003). VAD는 이걸
    발화로 읽으므로 세션 첫 동작이 "장치 클릭에 대답하기"가 된다. 첫 0.5초를 **버린다**(0으로 채우지
    않는다 — 디지털 무음은 VAD 입력으로 퇴화적이다, 아래 참고).
  - **backchannel("어"/"응") 구분** (`docs/barge-in-design.md`): ① 지연-정지(`barge_in_confirm_ms`
    250ms 이상 지속돼야 재생 끊음) + ② 사후 어휘 판정(`turn/backchannel.py`의 `is_backchannel()`이
    맞장구면 `pipeline.py`의 `should_continue_after_asr` 훅으로 LLM/TTS/저장 스킵).
    두 파라미터는 아직 실측 전 추정치 — 실제 발화 샘플이 필요하다(`_calibrate_turn_params.py`).
  - **엔드포인트 감지(옵션 `--endpoint-detect`)**: Smart Turn v3(`turn/detector.py`, 8M ONNX, CPU ~12ms)로
    "말 끝났나 vs 문장 중간 멈춤"을 판단해 자연스러운 멈춤이 잘리는 걸 완화. 기본 꺼짐.
    원래 backchannel용이었으나 실측상 end-of-turn 모델이라 엔드포인트로 재활용.
    Phase 2a에서 **적응형 grace**: `P(complete)`가 낮을수록(=확실한 문중 멈춤) 최대
    `endpoint_grace_ms`까지 기다리고, 애매하면 `endpoint_grace_min_ms`쪽으로 줄인다.
- **스트리밍 ASR (`--streaming-asr`, Phase 3)** — `stage/asr_stream.py`. 기본 꺼짐.
  켜면 마이크 프레임이 도착하는 대로 스트리밍 Zipformer에 들어가, **말이 끝나는 시점에 인식도 끝나
  있다** → ASR이 턴 임계경로에서 빠진다(`pipeline.run_streaming(pretranscribed=...)`이 ASR 스테이지를
  통째로 건너뛰고 `asr_ms`를 0으로 기록 — 없는 값이 아니라 실제로 0이다).
  - **LocalAgreement 안정화**: transducer의 가설은 불안정하다("그래서내가" → "그래서냈어"). 최근 N개
    가설의 최장 공통 접두사만 확정(N=2, whisper-streaming 기본값). 확정분은 **줄어들지 않는다** —
    이미 그걸 보고 행동한 소비자는 되돌릴 수 없으므로. 한국어는 이 체크포인트가 어절 스페이스를
    안 내므로 **문자 단위**로 비교한다.
  - **이중 엔드포인팅**: recognizer 자체 엔드포인트(디코더 상태 기반)와 TEN-VAD(음향 에너지 기반)가
    둘 다 돌고, 어느 쪽이 턴을 끝낼지는 `turn/controller.py`가 정한다. 둘은 유용하게 어긋난다 —
    TEN-VAD는 숨소리를 발화로 보고, 디코더는 단어 중간 멈춤을 붙들고 있는다.
- **`scripts/run_pipeline.py`**: 1회성 wav-in/wav-out CLI. 자동화 테스트·프리셋 결정론적 비교용.

## 저장 & 기억(개인화)

- **`storage.py`**: SQLite(`data/conversations.db`, 커밋 안 됨). 테이블 — `sessions` / `turns`(user·reply·
  프리셋·스테이지 ms) / `memories`(category/key/value/confidence).
- **기억** (`memory.py` + `talk.py`, `docs/memory-design.md`): 세션 종료 시 추출 →
  **Mem0식 consolidation**(기존 기억과 비교해 ADD/UPDATE/NOOP; DELETE는 0.6B 신뢰도 문제로 제외) →
  저장. 세션 시작 시 최신·고신뢰 상위 N개를 recall해 `system_prompt_suffix`로 주입. 방어적 파싱 +
  세션당 상한 + 2단 중복 정리. 추출/consolidation 모두 원샷 예시 프롬프트로 0.6B 안정화.

## 벤치마크 (`scripts/benchmark.py`)

고정 테스트셋(`--wav-dir`, 커밋 안 됨)을 ASR×LLM×TTS 프리셋 조합마다 돌려 스테이지별 평균 latency
표 출력. `--asr/--llm/--tts`로 좁히기, `--verbose`로 transcript도 출력(품질은 사람이 판단). ASR·LLM·TTS
프리셋 전반에 동작 검증됨.

## 환경 세팅

- **Linux/GPU**: `scripts/setup_local.sh`(RTX 5090)·`setup_server.sh`(H100) → 공용 `setup_common.sh`.
  `uv sync`, GPU 확인, 모델 다운로드, 격리 venv(MOSS-TTS-Nano·FreyaTTS) + VibeASR.cpp 빌드까지.
- **네이티브 윈도우**: `scripts/setup_windows.ps1` → `.venv-win` + `requirements/windows-cpu.txt`.
  **의도적으로 CPU 전용이고 범위가 좁다** — 코어 파이프라인(sherpa-onnx·llama.cpp·PortAudio)과 그게
  쓰는 모델만. VibeASR.cpp(MSVC+cmake 필요)와 PyTorch 격리 venv 두 개는 제외했다. 기본 파이프라인과
  턴테이킹 작업은 그것들을 건드리지 않고, VAD 임계값 측정하자고 C++ 툴체인을 요구하는 건 나쁜 거래다.
  - **왜 윈도우 환경을 만들었나**: WSL2는 WSLg PulseAudio 브리지로만 오디오에 닿아 캡처가 불안정하고
    버퍼링이 하드웨어의 것이 아니다 → 턴테이킹 파라미터를 전부 "추정치"로 둘 수밖에 없었다. 네이티브
    윈도우는 WASAPI로 실제 장치를 열거하므로 **드디어 측정**할 수 있다. 그리고 첫 측정에서 바로
    두 개가 나왔다: 턴이 절대 끝나지 않게 만드는 VAD 임계값, 그리고 아예 열리지도 않던 duplex 스트림.
    둘 다 몇 달간 보이지 않았다. CM4(무GPU) 타깃이라 여기서 잰 CPU 수치가 실제로 이전된다.

## 측정된 것 / 아직 추정치인 것

이 프로젝트의 턴테이킹 파라미터는 오래 "실측 전 추정치"였다. 지금 상태를 정확히 적는다.

| 파라미터 | 상태 | 근거 |
|---|---|---|
| `vad.yaml: threshold` | **실측** (0.5) | `_calibrate_vad_threshold.py` — 이 방의 실제 노이즈 플로어 대상 |
| `audio.yaml: delay_frames` | **실측** (1 = 28ms) | `_calibrate_aec_delay.py` 5회 중앙값 |
| 자기 자신에게 barge-in 안 함 | **실측** | `_smoke_duplex.py` — 단, 이 셋업(헤드폰)은 에코가 거의 없어 AEC 자체는 미검증 |
| Phase 3 스트리밍 인식 정확도 | **실측** | `_smoke_turn.py` — 배치 디코드와 유사도 1.00 |
| `barge_in_confirm_ms` (250) | 추정치 | 실제 맞장구/끼어들기 발화 녹음 필요 (`_calibrate_turn_params.py`) |
| `BACKCHANNEL_MAX_DURATION_S` | 추정치 | 위와 같음 |
| `streaming_asr.yaml: rule2_*` | 추정치 | 실제 발화로 확인 필요 |
| 대화 전체 루프(사람이 말하는) | **미검증** | 사람의 발화가 필요 — 아래 "다음 단계" |

**디지털 무음은 무음이 아니다.** TEN-VAD에 `np.zeros`를 먹이면 특징 추출이 퇴화해 모델이 무음
구간 내내 *발화*라고 답한다(측정 확인). 이 모델에 신호를 패딩하는 코드는 전부 실제 노이즈로 패딩해야
한다 — 스모크 하네스가 이것 때문에 멀쩡한 코드를 실패시킨 적이 있다.

## 명시적으로 범위 밖

- **웨이크워드**: 없음. 항상 듣고 있다.
- **디바이스용 표현력 TTS**: 한숨·웃음 등 NVV는 아직 어느 프리셋도 기본 지원 안 함(`docs/tts-*` 참고).

## 프리셋 추가하는 법

1. `src/nobody_flux/stage/{asr,llm,tts}.py`에 새 클래스 추가 — 기존 인터페이스에 맞춤:
   ASR `transcribe_file(wav)->str` / LLM `reply(text)->str`(+`history`,`reset`) / TTS `synthesize(text,out)->str`.
2. `registry.py`의 `_CLASSES`에 등록(화이트리스트 방식).
3. `configs/models.yaml`의 해당 스테이지에 프리셋 추가(`class`,`params`).
4. `--asr/--llm/--tts <이름>`으로 테스트.
5. 외부 모델/레포 필요 시 `scripts/setup_common.sh`(+ 윈도우에서도 필요하면 `setup_windows.ps1`)에
   다운로드/클론 단계 추가.

스트리밍 ASR은 프리셋이 아니다 — `registry.build_streaming_transcriber()` + `configs/streaming_asr.yaml`.
`models.yaml`의 ASR 프리셋들은 파일 넣고 텍스트 받는 배치 스테이지(벤치마크가 서로 비교하는 대상)인데
이건 라이브 프레임 스트림을 먹으므로 그 자리에 대입할 수 없다. 섞으면 프리셋 표에 절반의 호출자가
쓸 수 없는 객체가 들어간다.

## 다음 단계

- **사람이 말하는 대화 검증** — 남은 단 하나의 큰 미검증 항목이고, 자동화할 수 없다. 윈도우 박스에서:

      .venv-win\Scripts\python.exe scripts\talk.py --streaming-asr

  확인할 것: 발화가 잘리지 않고 잡히는지, 응답 재생 중 끼어들면 즉시 멈추는지, 맞장구("어", "응")에는
  안 멈추는지. 어긋나면 `_calibrate_turn_params.py`로 맞장구/끼어들기를 라벨 녹음해
  `barge_in_confirm_ms`를 확정한다(지금은 추정치).
- **AEC를 실제로 시험하기** — 현재 셋업은 헤드폰이라 재생이 마이크를 1.2배밖에 못 올린다 = 취소할
  에코가 사실상 없다. 스피커로 바꾸고 볼륨을 올린 뒤 `_smoke_duplex.py`를 다시 돌리면 `refgate`
  corr_threshold가 처음으로 실제 시험대에 오른다. CM4(한 통에 스피커+마이크)에선 필수.
- **턴테이킹·레이턴시 Phase 2~4 — 구현 완료.** 적응형 엔드포인팅(Phase 2a), 튜닝 도구(2b),
  스트리밍 ASR(3), 연속 캡처 3-상태 턴(4) 모두 들어갔다. 위 두 항목이 남은 검증이다.
- **TTS 표현력(NVV)** — TTS 3부작 문서(`tts-expressivity-design.md`/`tts-small-expressive-research.md`/
  `tts-conversational-build-design.md`) 참고. 방향: 대형 GPU 표현 모델은 서버 teacher로만, 디바이스는
  경량 CPU 모델(FreyaTTS 등)에 한국어 NVV 데이터+조건화를 얹기. CosyVoice2 한국어 NVV 실측은
  서버(H100) 작업으로 보류.
- **CM4 실기 실측** — 리서치 문서가 요구한 전제("모델 선정보다 CM4 PoC 선행").
