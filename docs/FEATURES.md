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
| LLM | `midm-2.3b-gguf` | `src/nobody_flux/stage/llm.py` (페르소나 `persona.py` — "퀜", 20~30대 또래 반말) |
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

**LLM** — 전부 llama.cpp GGUF Q4_K_M(=CM4와 같은 CPU 경로). 선정 근거와 라이선스 분석은
[`llm-conversational-selection.md`](llm-conversational-selection.md), 비교 도구는
`scripts/_ab_persona.py`.
- `midm-2.3b-gguf` (**기본**, 2026-08 교체) — KT Mi:dm 2.0 Mini. **MIT**, 한국어 특화,
  온디바이스 목적으로 pruning+distillation. 실측 위반 0/18·판박이 0%·턴 0.73s로 전 후보 중 최상.
  **주의**: 자체 시스템 프롬프트를 ~1000토큰 삽입하며 그 안에 "경어체 사용" 지시가 있다(우리
  프롬프트가 뒤에 붙어 이긴다). 첫 턴 프리필 6.7초는 `warm_up()`이 인사말 뒤로 숨긴다.
- `qwen3-1.7b-gguf` — Apache-2.0. 위반 1/18, 되묻기 89%(최고). 라이선스가 자유로운 대안.
- `kanana-2.1b-gguf` / `exaone-2.4b-gguf` — 한국어 특화지만 **둘 다 비상업 라이선스**라 참고용.
  EXAONE은 가장 큰데도 가장 나빴다(99자 장황, 이모지 5회).
- `qwen3-0.6b-gguf` (구 기본) — 빠르지만(0.35s) 위반 5/18, 되묻기 33%. 앵무새·이모지·붕괴가 잦다.
- `qwen3-0.6b` — 같은 weights, raw transformers.
- `lfm2-350m`/`lfm2-700m`/`lfm2-1.2b` — LiquidAI LFM2(엣지 특화). **벤치 결과: 이 페르소나(짧은 반말)엔
  드롭인 개선 아님** — 350m 마크다운 남발, 700m 반말 톤 좋으나 장황·느림(~2.7s vs qwen-gguf ~1.25s).
  qwen3-0.6b-gguf 기본 유지. (`configs/models.yaml` 주석에 상세)

**페르소나 준수: 규칙 대신 예시** (`persona.FEWSHOT_MESSAGES`). 시스템 프롬프트의 규칙은 명시적인데
0.6B가 한 세션에서 세 개를 동시에 어겼다 — 존댓말, 이모지, 사용자 말 앵무새. 규칙을 더 조이는 대신
**실제 대화 턴 5개를 예시로** 히스토리 앞에 넣었더니(각 예시가 실제 관측된 실패 하나씩을 시연):

| | 위반 |
|---|---|
| 규칙만 | **13/15** |
| 예시 추가 | **3/15** |

(같은 모델·같은 샘플링, 실패했던 입력 5개 × 3회. 채점: 문장 끝 존댓말 / 이모지 / 사용자 문장 그대로 반복)

남은 한계는 모델 용량이다 — 문법이 자주 깨지고, 숫자를 한글로 안 풀어쓰고(`20대`), "나도 잘
모르겠는데"를 과하게 붙인다. **예시 문구를 과잉 복사하는 성질 자체가 0.6B의 특징**이라, 예시를 넣을
땐 한 경로만 시연하면 그 문구가 모든 답변에 붙는다(감정 표현용 예시를 추가로 넣어 완화했다).

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
  실측해 `delay_ms`에 기록(윈도우 박스 실측: 28ms). 참조 신호는 샘플 단위로 지연시킨다 —
  30ms 프레임 단위로 양자화하던 시절엔 ±15ms 오차만으로 파형 상관이 0 근처가 돼
  `corr_threshold`를 어떤 값으로도 튜닝할 수 없었다(code-review #5). `delay_frames`는
  옛 설정을 위한 폴백으로만 남아 있다.
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
- **`scripts/run_pipeline.py`**: 1회성 wav-in/wav-out CLI. 수동 프리셋 비교·디버깅용 —
  assert도 골든 출력도 없고 LLM 샘플링 때문에 비결정적이므로 자동화 테스트가 아니다
  (그건 `tests/`와 스모크 스크립트의 몫).

## 저장 & 기억(개인화)

- **`storage.py`**: SQLite(`data/conversations.db`, 커밋 안 됨). 테이블 — `sessions` / `turns`(user·reply·
  프리셋·스테이지 ms) / `memories`(category/key/value/confidence).
- **기억** (`memory.py` + `talk.py`, `docs/memory-design.md`): 세션 종료 시 추출 →
  **Mem0식 consolidation**(기존 기억과 비교해 ADD/UPDATE/NOOP; DELETE는 0.6B 신뢰도 문제로 제외) →
  저장. 세션 시작 시 최신·고신뢰 상위 N개를 recall해 `system_prompt_suffix`로 주입. 방어적 파싱 +
  세션당 상한 + 2단 중복 정리. 추출/consolidation 모두 원샷 예시 프롬프트로 0.6B 안정화.

## 벤치마크 (`scripts/benchmark.py`)

고정 테스트셋(`--wav-dir`, 커밋 안 됨)을 ASR×LLM×TTS 프리셋 조합마다 돌려 스테이지별 평균 latency
표 출력. `--asr/--llm/--tts`로 좁히기, `--verbose`로 transcript도 출력(품질은 사람이 판단).
프리셋 전반 동작은 과거에 수동으로 확인한 것 — 지속 보증이 아니므로 프리셋이 깨지면 다음
수동 실행 전까지는 모른다.

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
| `audio.yaml: delay_ms` | **실측** (28ms) | `_calibrate_aec_delay.py` 5회 중앙값 — 이제 샘플 단위로 적용된다(#5) |
| 자기 자신에게 barge-in 안 함 | **실측** | `_smoke_duplex.py` — 단, 이 셋업(헤드폰)은 에코가 거의 없어 AEC 자체는 미검증 |
| Phase 3 배선(파이프라인) | **실측** | `_smoke_turn.py` — 깨끗한 테스트 wav로 배치와 유사도 1.00 |
| Phase 3 **실사용 정확도** (`engine: zipformer`) | **실패** | 실제 마이크 발화를 못 읽음 (16개 중 14개 빈 결과) — 아래 참고. 상류 #2886, 우리가 못 고침 |
| Phase 3 **실사용 정확도** (`engine: chunked-sensevoice`, 신규 기본) | **실측 — 정확도 OK** | 배치 대비 CER 0.000, 빈 결과 없음. 단 **스트리밍 이득은 없다**: 최초 커밋 1.29초, 16개 중 7개는 아무것도 못 커밋 |
| chunked 디코드 비용 | **실측** | 증폭 3.0x, wall RTF 0.21 (4코어 프록시). **CM4 미측정** — per-core가 느려 여유가 훨씬 작다 |
| 커밋된 부분 전사를 믿을 수 있는가 | **실측 — 아니다** | 재디코드가 해석을 바꾼다: `"오늘 산책 코스 추천"` → `"오늘 산체코 추천해줘."` |
| 순수 로직 단위 테스트 | **있음** (141개, <3s) | `tests/` — 가중치·오디오 장치 불필요. 컨트롤러 상태기계, VadStream duration, chunker, memory 파싱/consolidation, storage 쿼리, `_AudioRing`, resample, 스레드 예산, backchannel/LocalAgreement/grace. 파라미터 실측과는 다른 축: 이건 "코드가 명세대로 동작"만 보증한다 |
| `barge_in_confirm_ms` (250) | 추정치 | 실제 맞장구/끼어들기 발화 녹음 필요 (`_calibrate_turn_params.py`) |
| `BACKCHANNEL_MAX_DURATION_S` | 추정치 | 위와 같음 — 단, 게이트에 들어가는 duration이 pre-roll 제외한 실발화 길이(`speech_duration_s`)라는 것 자체는 회귀 테스트로 고정(2026-08-14 전까지는 pre-roll 포함 길이가 들어가 게이트가 죽은 코드였다) |
| `streaming_asr.yaml: rule2_*` | 추정치 | 실제 발화로 확인 필요 |
| 대화 전체 루프(사람이 말하는) | **미검증** | 사람의 발화가 필요 — 아래 "다음 단계" |

**디지털 무음은 무음이 아니다.** TEN-VAD에 `np.zeros`를 먹이면 특징 추출이 퇴화해 모델이 무음
구간 내내 *발화*라고 답한다(측정 확인). 이 모델에 신호를 패딩하는 코드는 전부 실제 노이즈로 패딩해야
한다 — 스모크 하네스가 이것 때문에 멀쩡한 코드를 실패시킨 적이 있다. 실사용에서도 같은 성질이
나온다: 마이크가 죽으면 조용해지는 게 아니라 **가짜 턴을 계속 만들어낸다**(아래 참고).

### streaming-zipformer-ko는 실사용 발화를 못 읽는다 (Phase 3 미해결)

`--streaming-asr`은 **깨끗한 테스트 wav에선 완벽하고**(배치와 유사도 1.00) **실제 마이크 발화에선
빈 문자열을 낸다.** 스트리밍 래퍼 문제가 아니다 — 같은 체크포인트의 **배치** 경로도 똑같이 빈 결과다.
반면 SenseVoice는 같은 오디오를 정상적으로 읽는다. 실측:

| 입력 | Zipformer | SenseVoice |
|---|---|---|
| `ko.wav` 원본 | 정상 | 정상 |
| 실제 캡처 0.94초 ("누구세요") | **`''`** | `'누구세요.'` |
| 실제 캡처 1.73초 | **`''`** | `'코 추천해 줘.'` |

원인 후보를 하나씩 배제했다:
- **레벨 아님** — 실제 캡처를 16배 증폭해도 빈 결과.
- **잡음 아님** — 깨끗한 발화는 SNR 5dB에서도 정상.
- **리드인은 필요하다** — 발화 시작 지점부터 자르면 빈 결과, 앞에 0.2초만 붙여도 살아난다.
  (`pre_roll_ms` 300은 이 요건은 충족한다.)
- **길이도 영향** — 리드인을 줘도 0.5초 발화는 빈 결과, 0.8초는 부분, 완전 인식엔 ~2.8초 필요.
- **그런데 리드인·길이를 다 맞춰줘도 실제 캡처는 여전히 빈 결과.** 남은 건 화자/마이크 특성이다.

즉 **Phase 3의 배선은 옳고 모델이 안 맞는다.** 기본값이 꺼짐인 이유이고, 켜도 `pipeline`이 배치
ASR로 폴백하므로 대화는 계속 된다 — 다만 그 폴백이 **완전히 조용했었다.** 플래그는 "enabled"라고
하고 로그는 아무 말도 안 해서, 한 세션 내내 배치로 돌면서 동작하는 것처럼 보였다. 이제 빈 결과마다
로그를 남기고 2회 연속이면 경고한다.

**다음**: 다른 한국어 스트리밍 체크포인트를 찾거나, SenseVoice를 청크 단위로 굴리는 쪽을 검토.

### → 2026-08-18: 두 번째 선택지를 구현했다 (`engine: chunked-sensevoice`)

첫 번째 선택지(다른 체크포인트 찾기)는 접었다. sherpa-onnx
[#2886](https://github.com/k2-fsa/sherpa-onnx/issues/2886)이 **우리와 똑같은 증상**을
보고하는데("always returns empty string") 2025-12-10 오픈 이후 메인테이너 응답도 PR도 없고,
원인 추정이 **인코더 ONNX export 결함**이라 우리가 고칠 수 있는 게 아니다. `asr.py:69-72`가
기다리던 그 픽스는 오지 않는다.

그래서 `ChunkedSenseVoiceTranscriber`를 만들었다 — 발화를 누적하면서 `hop_s`마다
**처음부터 전체를 SenseVoice로 다시 디코드**하고, 기존 `LocalAgreementStabilizer`로 안정화한다.
청크를 독립적으로 디코드해 이어붙이는 방식이 아닌 이유: SenseVoice는 전체 입력에 대한
비자기회귀 인코더이고, 위 표대로 **0.5초면 빈 결과, 완전 인식엔 ~2.8초**가 필요하다.
독립 청크는 어떻게 이어붙여도 개별적으로 판독 불가다.

**실측 (`NOBODY_CPU_BUDGET=4`, 실캡처 15개 + clean-ko):**

| | 배치 SenseVoice 대비 CER | 빈 결과 | 디코드 증폭 | wall RTF |
|---|---|---|---|---|
| `chunked-sensevoice` | **0.000** | **4/16** (배치와 동일 — 원래 무음인 것들) | 3.0x | 0.21 |
| `zipformer` (기존) | — | **14/16** | 1.0x | — |

**성공 기준은 통과했다**: 실제 캡처에서 빈 문자열이 안 나온다.
`s9t1`→`'누구세요.'`, `s9t2`→`'코 추천해 줘.'`, `s3t3`→`'너 이름이 뭐야?'`
(zipformer는 셋 다 `''`, 또는 `'어'`).

**그런데 "스트리밍"으로는 실패다. 이게 중요하다.**

- 최초 커밋 시점이 **항상 1.29초**다. 구조적으로 `min_decode_s + hop_s`이고(가설 1개 만들고,
  하나 더 만들어 합의해야 커밋) SenseVoice가 0.8초 미만에서 아무것도 못 내므로 **줄일 수 없다.**
- 우리 캡처 셋 대부분이 0.6~0.9초라 **16개 중 7개가 finalize 전에 아무것도 커밋하지 못했다.**
- 최종 텍스트가 배치와 동일한 건 당연하다 — `finalize()`가 전체를 한 번 더 디코드하니까.
  즉 **정확도 이득은 0이고, 얻은 것은 "조용한 폴백이 사라진 것"뿐**이다.
- 게다가 독립 재디코드는 **이미 본 오디오의 해석을 바꿀 수 있다.**
  실측: `"오늘 산책 코스 추천"` → `"오늘 산체코 추천해줘."`. LocalAgreement의 단조 가드는
  접두사가 *짧아지는* 것만 막고 내용이 바뀌는 건 못 막는다 — 트랜스듀서를 상정해 만든
  보장이라 여기선 더 약하다. **커밋된 텍스트를 믿고 행동하면 안 된다.**

→ 결론: `--streaming-asr`을 **정직하게** 만들었지 빠르게 만든 게 아니다.
기본 엔진은 `chunked-sensevoice`로 바꿨다(실제 발화에 침묵하는 엔진보다 늦는 엔진이 낫다).
**투기적 프리필(계획 트랙 C-2)의 입력으로도 이 수치가 중요하다** — 부분 전사가
1.3초 이상 발화에만 존재하고 내용이 뒤집힐 수 있으므로, 적중률 기대를 낮춰야 한다.

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

## 사람이 직접 해야 할 검증 (미완료)

**여기 있는 건 자동화할 수 없다.** 사람이 말하고, 듣고, 판단해야 하는 것들이다. `tests/`의
146개 테스트는 "코드가 명세대로 동작한다"만 보증하며 아래 어느 것도 대신하지 못한다.
이 목록이 비기 전까지, 턴테이킹 경로의 동작은 **설계상 그럴 것**이지 **확인된 것**이 아니다.

1~4번은 **같은 세션에서 한 번에** 하는 게 맞다. 말을 걸어야 1·2번이 되고, 그 부산물로 3번의
기억 테이블과 4번의 실패 턴이 쌓인다 — 따로 시간을 내는 게 아니라 대화하는 동안 함께 관찰하는
항목들이다. 다만 3번은 세션이 여러 번 필요하다(기억은 세션 종료 시 저장되고 다음 세션에
주입된다).

각 항목은 "어긋나면 무엇을 하는가"까지 적어뒀다. 어긋난 걸 발견하는 것보다 **그때 무엇을
측정할지 아는 것**이 중요하다 — 추측으로 파라미터를 돌리면 되돌릴 근거가 없어진다.

### [ ] 1. 대화 루프 — 사람이 말하는 실사용 (최우선)

```powershell
.venv-win\Scripts\python.exe scripts\talk.py --streaming-asr
```

10분이면 된다. 확인할 것:

- [ ] **발화가 잘리지 않고 잡히는가** — 문장 중간에 숨을 쉬면 거기서 턴이 끊기는지.
      끊기면 `--endpoint-detect`(Smart Turn v3 적응형 grace)를 켜고 다시.
- [ ] **응답 재생 중 끼어들면 즉시 멈추는가** — 반응이 굼뜨면 `barge_in_confirm_ms`(현재 250,
      추정치)가 너무 크고, 자기 목소리에 스스로 멈추면 너무 작다.
- [ ] **맞장구("어", "응", "그래")에는 안 멈추는가** — ⚠️ **이 동작은 실사용 경험이 0회다.**
      `pre_roll_ms`가 300→500으로 오른 뒤로 게이트가 닫혀 있어 판정 자체가 실행된 적이 없고
      (code-review #1), 2026-08-14에야 되살아났다. 즉 2단계 barge-in 설계가 처음으로
      실제로 돌아가는 순간이다. 여기가 제일 깨지기 쉽다.
- [ ] **응답이 끊기지 않고 이어 재생되는가** — TTS 합성이 LLM 디코드와 겹치도록 바뀌었으므로
      (code-review #2) 문장 사이 공백이 예전과 다르게 들릴 수 있다.

어긋나면 추측하지 말고 측정한다 — `scripts/_calibrate_turn_params.py`로 맞장구/끼어들기를
라벨 녹음하면 지속시간 분포에서 `barge_in_confirm_ms`와 `BACKCHANNEL_MAX_DURATION_S`를
제안해준다(`--apply`로 `vad.yaml`에 기록).

**새 기기라면 먼저**: `scripts/_calibrate_vad_threshold.py --apply`. `vad.yaml`의 `threshold`는
기기마다 다시 재야 하고, 너무 낮으면 무음을 발화로 읽어 **턴이 영원히 안 끝난다**(=응답이 아예
안 나온다). 실제로 겪은 실패다.

### [ ] 2. 대화 품질 — 자동 지표가 못 재는 것

`_ab_persona.py`가 규칙 위반·되묻기·판박이·문맥·드리프트를 재지만, 그 전부가 "확실히
아닌 것"을 거르는 용도다. **다시 대화하고 싶은지는 사람만 안다** — 재미있는지, 흐름이
자연스러운지, 어색한 지점이 어딘지. 기본 LLM(`midm-2.3b-gguf`)으로 몇 세션 해보고 판단.

이게 파인튜닝 착수 여부의 입력이다(`docs/llm-conversational-selection.md`의 "파인튜닝" 참고).

### [ ] 3. 기억(개인화) — 세션을 여러 번 해야만 보인다

기억은 **세션 종료 시** 추출되고 **다음 세션 시작 시** 주입된다. 그래서 한 세션만으로는
아무것도 검증할 수 없다 — 최소 2~3세션을 연속으로 하고 나서 봐야 한다.

현재 `data/conversations.db`에는 9세션/15턴이 있지만 실제 대화가 오간 건 4세션뿐이고,
추출된 기억은 2행이다. **그 2행 중 하나가 이미 불량이다**:

```
id=1  identity  key="이름"  value="이름"  confidence=0.0
```

값이 키를 그대로 반복한 행 — 이름을 못 뽑았는데 뽑은 척한 것이다(`confidence=0.0`이라 정렬
최하위로 밀리긴 한다). 아래 항목들이 가정이 아니라 **이미 관측된 실패**라는 뜻이고, 표본을
늘려서 이게 얼마나 자주 나오는지부터 봐야 한다.

세션 종료 시 로그부터 본다:

```
[memory] added 3, updated 1, skipped 2
```

- [ ] **`added`가 매 세션 늘기만 하는가** — 같은 사실을 다시 말했는데도 `updated`/`skipped`가
      아니라 `added`로 잡히면 consolidation이 일을 안 하는 것이다.
- [ ] **`[memory] consolidation output unusable` 경고가 뜨는가** — 뜨면 모델이 op 배열을
      제대로 못 내고 있다는 뜻. 조용히 전부-ADD로 퇴화하던 걸 2026-08-14에 로그로 드러냈다
      (code-review #6, #7).
- [ ] **`extraction failed, skipping`이 뜨는가** — 긴 세션에서 뜨면 컨텍스트 초과다
      (#10에서 윈도우 분할을 넣었으므로 이제는 안 떠야 한다).

그다음 테이블을 직접 본다:

```bash
sqlite3 data/conversations.db \
  "SELECT id, session_id, category, key, value, confidence FROM memories ORDER BY id;"
```

- [ ] **틀린 사실이 저장됐는가** — 특히 UPDATE 오판정(엉뚱한 행을 덮어썼는지). 이게
      비가역 데이터 손실이라 제일 위험하다.
- [ ] **값이 키를 그대로 반복하는 행이 또 나오는가** — 위 `이름: 이름`처럼. 자주 나오면
      `_extract_json_array`에 "value가 key와 같으면 버린다"는 검증을 넣을 값어치가 있다
      (정보량이 0인 행이라 안전하게 거를 수 있다). 지금은 표본 1건이라 판단 보류.
- [ ] **`confidence` 값이 의미 있는가** — 모델이 0.9를 남발하면 정렬 기준으로 쓸모가 없다.
      실제 분포를 보고 판단.
- [ ] **말투/지시가 사실로 둔갑해 저장됐는가** — "앞으로 존댓말 써" 같은 발화가
      `preference: 말투` 행이 되면 다음 세션 페르소나를 조용히 뒤집는다. 카테고리 검증과
      "지시가 아니다" 프레이밍을 넣어뒀지만(#4) 실제로 뚫리는지는 사람이 시도해봐야 안다.
- [ ] **다음 세션에서 실제로 기억하는가** — 이전 세션에서 말한 이름/취미를 물어본다.

`docs/memory-design.md`의 "다음 단계"가 요구하는 게 정확히 이 관찰이다.

### [ ] 4. 실패한 턴 수집 — 평가 세트를 키우는 일

대화하다 어긋난 턴이 나오면 **그게 다음 회귀 테스트다.** 지금 `_ab_persona.py`의 단발 입력
6개와 `configs/persona_scenarios.yaml`의 시나리오 15개가 정확히 그렇게 만들어졌다 — 실사용에서
터진 턴을 먼저 넣고, 그것들이 못 건드린 축만 손으로 채웠다.

```bash
sqlite3 data/conversations.db \
  "SELECT turn_index, user_text, reply_text FROM turns WHERE session_id = (SELECT MAX(id) FROM sessions);"
```

- [ ] **어긋난 턴을 골라 `_ab_persona.py`의 `INPUTS`에 추가** — 단발로 재현되는 것.
- [ ] **여러 턴에 걸친 실패는 `persona_scenarios.yaml`에 시나리오로 추가** — 드리프트,
      문맥 망각, 같은 되물음 반복처럼 단발로는 안 잡히는 것. 사용자 발화는 **고정 스크립트**로
      쓴다(모델 응답에 의존하면 재현이 안 된다).
- [ ] **ASR 오인식도 그대로 남긴다** — `asr-오인식` 시나리오가 실제 오인식("산책 코스" →
      "산체코")을 그대로 쓰는 이유다. 깨끗한 입력만 테스트하면 실사용에서 터진다.

파인튜닝을 하든 프롬프트를 고치든 **전후 비교의 근거가 여기서 나온다.** 세트가 작으면
좋아졌는지 알 방법이 없다.

### [ ] 5. AEC 실검증 — 스피커 필요

현재 측정 셋업은 헤드폰이라 재생이 마이크 레벨을 1.2배밖에 못 올린다 = **취소할 에코가 사실상
없다.** 스피커로 바꾸고 볼륨을 올린 뒤 `_smoke_duplex.py`를 다시 돌려야 `refgate`의
`corr_threshold`가 처음으로 실제 시험대에 오른다.

2026-08-14에 참조 신호 정렬이 프레임 단위 → **샘플 단위**로 바뀌었다(code-review #5). 그
전에는 ±15ms 오차만으로 파형 상관이 0 근처가 돼 어떤 threshold로도 튜닝이 불가능했으므로,
**예전에 안 되던 것이 지금은 될 수 있다** — 이 수정 이후 아직 아무도 시험하지 않았다.
CM4는 한 통에 스피커+마이크가 들어가므로 필수 항목.

### [ ] 6. CM4 실기 측정 — 하드웨어 필요 (현재 보류)

보드가 없어 진행 불가. 대신 스레드만 4코어로 묶은 프록시 측정은 끝났고
(`NOBODY_CPU_BUDGET=4`, `docs/llm-conversational-selection.md`), 결론은 **격차가 코어 수가
아니라 코어당 성능이라 설정으로는 못 줄인다**는 것. 즉 실기가 느리면 답은 스레드 조정이
아니라 더 작은 모델이나 distillation이다. **보드를 구할지 타깃을 재검토할지는 결정 사항.**

## 다음 단계

- **턴테이킹·레이턴시 Phase 2~4 — 구현 완료.** 적응형 엔드포인팅(Phase 2a), 튜닝 도구(2b),
  스트리밍 ASR(3), 연속 캡처 3-상태 턴(4) 모두 들어갔다. 남은 건 위 "사람이 직접 해야 할 검증".
- **스트리밍 ASR 살리기** — `--streaming-asr`은 지금 사실상 켜지지 않는다(위 "실사용 정확도"
  참고). 다른 한국어 스트리밍 체크포인트를 찾거나 SenseVoice를 청크 단위로 굴리는 쪽 검토.
- **TTS 표현력(NVV)** — TTS 3부작 문서(`tts-expressivity-design.md`/`tts-small-expressive-research.md`/
  `tts-conversational-build-design.md`) 참고. 방향: 대형 GPU 표현 모델은 서버 teacher로만, 디바이스는
  경량 CPU 모델(FreyaTTS 등)에 한국어 NVV 데이터+조건화를 얹기. CosyVoice2 한국어 NVV 실측은
  서버(H100) 작업으로 보류.
- **CM4 실기 실측** — 리서치 문서가 요구한 전제("모델 선정보다 CM4 PoC 선행").
