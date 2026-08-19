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
| Phase 3 **실사용 정확도** (`engine: chunked-sensevoice`, 신규 기본) | **실측 — 정확도 OK** | 배치 대비 CER 0.000, 빈 결과 없음. 단 **스트리밍 이득은 없다**: 최초 커밋 1.29초, 16개 중 8개는 쓸 만한 걸 못 커밋 |
| chunked 디코드 비용 | **실측** | 증폭 3.0x, wall RTF 0.21 (4코어 프록시). **CM4 미측정** — per-core가 느려 여유가 훨씬 작다 |
| 커밋된 부분 전사를 믿을 수 있는가 | **실측 — 아니다** | 재디코드가 해석을 바꾼다: `"오늘 산책 코스 추천"` → `"오늘 산체코 추천해줘."` |
| **TTS 기본값 결정** | **결론 — 자체 `matcha-ko` 유지** | 3회 반복 측정: CER 중앙값 0.043(구간 0.021~0.053)로 supertonic-3(0.064~0.074)·supertonic-2(0.064~0.117)와 **구간이 겹치지 않는다** = 명료도 우위 실재. 자체 학습이라 재학습 가능 + 제3자 사용제한 없음 |
| ⚠️ **TTS 합성이 프로세스마다 다르다** | **실측 — 측정법 결함** | Matcha는 flow-matching + `noise_scale`이고 sherpa가 프로세스 단위 시드 → 같은 문장이 md5·샘플수까지 다름. 시드 파라미터 없음. **94자 세트 CER 해상도 ±0.04** → 1회 측정은 측정이 아니다. `_ab_tts.py --repeat` 기본 3으로 수정 |
| ~~화자 10명 CER 순위~~ | **무효** | sid=7(0.025)~sid=2(0.089) 순위 전체가 위 노이즈 밴드 안이었다. `speaker_id: 7`은 임의 선택 |
| Supertonic 2 vs 3 속도 | **실측** | v2가 v3보다 **2.9배**, matcha보다 **1.7배** 빠름(v3가 스텝 5→8). 스텝 수는 sherpa로 조절 불가 → v2가 유일한 경로 |
| matcha-ko 출처·라이선스 | **확인 완료** | 자체 학습 모델. 목소리 디자인·학습 데이터를 **Qwen3-TTS로 생성**. 그 출력물의 학습 이용 가능 여부는 **프로젝트 오너가 확인**(2026-08-18) → 라이선스 미결 사항 없음. 가중치도 자체 소유 |
| ⚠️ **espeak-ng이 sherpa-onnx DLL에 정적 링크** | **실측** | `sherpa-onnx-c-api.dll`에 `espeak_ng_*` 전체 API + piper-phonemize `phonemize_eSpeak` 심볼 + 하드코딩 `/usr/share/espeak-ng-data`. **ASR·VAD가 같은 DLL을 쓴다** → TTS 프리셋을 바꿔도 GPL-3.0 노출은 안 사라진다. "Supertonic은 espeak 의존 없음 = 라이선스 이득"이라 적었던 것은 **데이터 디렉터리 수준에서만 참** |
| espeak-ng GPL-3.0 충돌 | **1차확인 — upstream 인정** | k2-fsa/sherpa-onnx#3731(2026-07-08): "GPL … incompatible with the Apache-2.0 license of sherpa-onnx" → **2.0.0에서 제거 예고**. 아직 미출시(PyPI 최신 1.13.6). 마이그레이션은 `lexicon.txt` 또는 외부 음소화 + 신규 `tokens` 필드 |
| matcha-ko가 2.0.0에서 좌초된다 | **실측** | `tokens.txt`가 **espeak IPA 음소 목록**(159개, 앞 14개가 matcha-en과 바이트 동일, 꼬리가 IPA 구별기호). ONNX 메타데이터도 `"espeak-ng ko phonemes"`. → 2.0.0에선 한국어 lexicon이나 외부 음소화기가 있어야 동작. Supertonic은 character-level이라 무영향 = **이게 v3의 진짜 장점(라이선스가 아니라 전방 호환성)** |
| `sherpa-onnx` 상한 없었음 | **수정** | 핀이 `>=1.13.4`(상한 없음)이라 2.0.0 출시 후 새 설치가 기본 TTS를 조용히 깨뜨릴 수 있었다 → `>=1.13.4,<2`로 상한 추가(`pyproject.toml`, `requirements/windows-cpu.txt`) |
| `data_dir` 부재 = **프로세스 사망** | **실측** | 잘못된/빈 `data_dir` → rc=1, **Python 예외 없음**(C 레벨 abort, try/except로 못 잡음). TTS만 degrade되는 게 아니라 에이전트 전체가 종료. 먼저 하드코딩 `/usr/share/espeak-ng-data`로 폴백하므로 **리눅스 타깃에선 시스템 espeak 사전을 쓸 위험** |
| espeak 측정 시 프로세스 분리 필수 | **실측 — 측정법** | espeak-ng은 프로세스 전역 싱글턴(`espeak_ng_Initialize`). 한 프로세스에서 정상 경로를 먼저 초기화하면 뒤이은 잘못된 경로가 그 상태를 재사용해 **거짓 통과**한다(처음 이렇게 재서 "없어도 동일 오디오"라는 오답을 얻었다) |
| GPL-3.0 고지·소스 제공 | **부분 해결** | `THIRD-PARTY-NOTICES.md` 신규 — 라이선스 전문 위치와 **대응 소스 위치**(espeak-ng `dictsource/`·`phsource/`) 명시. **실측: 모델 번들 15개 중 라이선스 파일을 싣는 건 3개**이고 하나는 URL 한 줄(`sense-voice`), 하나는 가중치에 틀린 MIT(Supertonic 샘플코드용). 이미지에 무엇을 동봉해야 충분한지는 **여전히 법무 판단** |
| 🚨 **기본 VAD(TEN-VAD)가 Apache-2.0이 아니다** | **1차확인 — 라이선스 함정 7번** | "Apache License v2.0 **with additional conditions**"이고 추가 조항이 **Agora 경쟁 금지**다(*"may not Deploy the ten-vad in a way that competes with Agora's offerings"*). 파생물도 같은 조건 유지. SenseVoice(§11.1)와 같은 계열 — **기본값 스테이지의 라이선스를 확인한 적이 없었다.** 제품화 전 법무 |
| ⚠️ **기본 VAD가 열화된 export다** | **1차확인 — ONNX 메타데이터** | `ten-vad.onnx`의 `comment`: *"It uses 0 as the pitch feature, which may degrade the performance."* sherpa export가 **pitch feature를 버렸다.** `_calibrate_vad_threshold.py`로 짜낼 수 있는 성능의 상한이 여기서 정해진다 |
| VAD 엔진이 설정값이 됐다 | **있음** | `configs/vad.yaml`의 `engine`: `ten-vad`(기본) \| `silero-vad`(MIT). 라이선스 판단이 코드 변경이 되지 않게 하는 것이 목적 — 기본값은 **안 바꿨다**. 코드 접점은 `vad.build_sherpa_vad_config` 한 곳, 테스트 16개 |
| 엔진별 `threshold` 분리 | **있음 — 의도적** | `threshold: 0.5`는 이 방 마이크 **실측값**이고 다른 모델로 이전되지 않는다. 그래서 공유 키가 아니라 **엔진 블록 안**에 뒀다. `_calibrate_vad_threshold.py`도 이제 yaml의 `engine`을 읽는다 — 안 그러면 silero 설정에 ten-vad를 캘리브레이션해 보고했을 것 |
| 받은 Silero의 정체 | **1차확인 — ONNX 메타데이터** | **v4**, 16kHz 브랜치만(`SAMPLE_RATE` 16000이라 무관). 입력 `x: [1, 512]`로 **window_size 512가 그래프에 박혀 있다**(TEN-VAD는 256) → `window_size`를 공유하지 않고 엔진별 sherpa 기본값을 쓴다. v5는 같은 릴리스에 별 파일로 있음 |
| 두 엔진 분절 비교 (깨끗한 wav) | **실측 — 예비** | `ko.wav`(5.61s)에서 speech 3.01s(ten) vs 2.88s(silero), 이벤트 시퀀스 동일. **실제 캡처·사람 청취는 미실시** — 품질 비교로 읽지 말 것 |
| 🐛 `_load_yaml`이 캐시를 오염시켰다 | **실측 — 기존 버그 수정** | mtime 캐시된 **같은 dict 객체**를 반환하는데 모든 빌더가 그걸 `update`/`pop`한다 → 한 프로세스에서 두 번 빌드하면 **첫 호출의 override가 남고 엔진 블록이 사라져** dataclass 기본값으로 조용히 폴백. 재현: streaming_asr.yaml 재로드가 `num_threads: 99`와 블록 0개를 반환. `talk.py`는 스테이지를 한 번만 빌드해서 안 걸렸지만 `benchmark.py`·`_ab_*`는 반복 빌드한다. deepcopy 반환으로 수정, 회귀 테스트 15개 |
| 재배포 모델 출처 추적법 | **1차확인 — 방법** | sherpa 문서·릴리스노트·export PR(#2012)에 없던 `vocos` 출처가 **ONNX 메타데이터 안에** 있었다(`model_author: BSC-LT`, `url1/url2`) → Apache-2.0 확정. `ten-vad`(라이선스 URL + 열화 경고)와 `matcha-ko`(`has_espeak`)도 같은 방식으로 나왔다. **문서보다 파일이 정확하다** |
| 미확인 가중치 라이선스 6건 | **해결 — 전건 확인** | TEN-VAD(Apache+조건, 위), Mi:dm **MIT**(원본 `K-intelligence` 확인 — 우리가 받는 건 `mykor` 재양자화본), vocos **Apache-2.0**, matcha-en(icefall Apache-2.0 + LJSpeech, **가중치 선언 없음**), zipformer-ko(**태그 없음** + KsponSpeech 신청 필요), Qwen3 GGUF(원본 Apache-2.0, **bartowski 파생본은 선언·LICENSE 파일 둘 다 없음**). 파이썬 의존 10개는 dist-info로 전건 확인, 전부 permissive |
| espeak 데이터 경로 우선순위 | **1차확인 — 소스** | `espeak_ng_InitializePath`: **넘긴 경로 → `ESPEAK_DATA_PATH` → `$HOME` → 컴파일 기본값**. 즉 우리가 유효한 경로를 주면 **env var도 레지스트리도 못 덮어쓴다** → `ESPEAK_DATA_PATH`는 위험이 아니다(검색 요약은 반대로 말했고 소스가 반박) |
| ~~시스템 `/usr/share/espeak-ng-data` 폴백 위험~~ | **해결 — 가드** | `check_data_path`가 **디렉터리 여부만** 보고 phontab 유무는 안 본다 → 존재하지만 빈 디렉터리가 통과 후 사망. `SherpaMatchaTts._check_data_dir`이 4개 파일을 먼저 검증해 `FileNotFoundError`를 던진다. 프로브 재실행으로 두 실패 모드 모두 `C-ABORT`→`PY-RAISE` 확인. 리눅스에서의 **조용한 오발음**도 같이 막힘 |
| 2.0.0 lexicon 마이그레이션 | **조사 완료 — 착수는 형식 확정 후** | 프론트엔드가 **ONNX 메타데이터로** 선택된다(`has_espeak`→lexicon 무시, 분기 없으면 `EXIT(-1)`). matcha-ko는 `has_espeak: 1` → **`lexicon` 설정해도 무시됨**. 업스트림의 "모델마다 lexicon.txt 추가"는 **자기 레포 모델 한정** → 우리 건 우리가 만들어야. lexicon 조회는 UTF-8 **문자 단위**·OOV는 조용히 버림 → 한국어는 **음절 단위**가 맞고 음절 경계 음운규칙을 잃는다 |
| espeak 한국어 규칙의 실제 규모 | **1차확인 — 디스크** | `ko_dict` **47KB** (en 167KB, cmn 1.5MB), `lang/ko`는 **51바이트**(name/language/pitch/intonation 4줄뿐). 레포가 예전부터 적어둔 "한국어 규칙 약함"의 정량 근거 — 대체 시 잃을 것도 그만큼 적다 |
| ⚠️ 가짜 espeak 번들로 테스트하면 pytest가 죽는다 | **실측 — 측정법** | 0바이트 `phontab`을 만들어 가드를 통과시키면 espeak이 C에서 파싱하다 **프로세스를 abort**한다 → 실패 리포트 없이 출력만 잘리고 exit 1. 가드 테스트는 `__post_init__`을 통하지 말고 duck-typed stub으로 메서드를 직접 부를 것 |
| **프롬프트 프리픽스 KV를 디스크에 저장** | **실측 — 구현 완료** | `warm_up()`이 `llama_state_seq_save_file`/`load_file`로 스냅샷. 재시작 후 warm_up **3.29s → 0.14s**(24배). 프리픽스 648토큰, 스냅샷 **75.6MB**(= 648 × 28층 × 2(K+V) × 1024 × 2B = 74.3MB, 산술 일치). 검증: `scripts/_verify_kv_prefix.py` |
| KV 복원이 실제로 맞는지 | **실측 — 맞다** | 복원 상태에서 greedy 생성한 답이 정상 프리필 답과 **문자 단위 동일**. 이 확인이 필요한 이유: 복원이 어긋나면 예외가 아니라 **그럴듯한 오답**이 나온다 |
| ⚠️ 프리픽스 토큰화가 `create_completion`과 달랐다 | **실측 — 버그 수정** | `Llama.tokenize` 기본값은 `add_bos=True, special=False`인데 `_create_completion`은 `add_bos=False, special=True`를 쓴다. 기본값으로 재면 `<\|im_start\|>`가 특수토큰 1개가 아니라 **리터럴 텍스트**가 되어 **744 vs 659토큰**으로 갈렸다. 조용한 실패 — 스냅샷을 복원해도 `generate()`가 거부하고 재프리필해서 **이득이 0**이 된다 |
| 스냅샷 누적 누수 | **수정** | 키에 모델 identity + 프리픽스 토큰이 들어가므로 **페르소나를 고칠 때마다 75MB가 고아로 남는다**. `prune_stale_kv_snapshots()`가 저장 시 나머지를 삭제(현재 1개만 유용). SD카드 타깃에서 아무도 못 알아채는 종류의 누수였다 |
| CM4/CM5에서 이 이득 | **미측정 — 추정** | SD카드 순차 읽기 20~40MB/s 가정 시 75MB 로드는 **2~4초**. 같은 프리필 추정치가 **130~230초**(§10.2)이므로 여전히 압도적 이득이지만, 개발 박스(NVMe, 0.14s)와는 자릿수가 다르다 |
| **CM4 타깃 판정** | **결론 — 사지 말 것** | A72는 Armv8.0-A로 dotprod·i8mm·FP16 산술이 없고, llama.cpp의 ARM 양자화 빠른 경로가 **전부** dotprod를 요구한다(GCC/LLVM 소스 확인). `ggml_vdotq_s32`가 NEON 6개로 에뮬레이션됨. 동일 실리콘(Pi 4) 실측 환산 → 45토큰 응답 **13~18초**, warm-up **130~230초**. → **CM5(A76, dotprod 있음)로 타깃 상향 권고.** `research-delta-20260818.md` §10 |
| CM4에서 죽는 것 / 사는 것 | **실측** | **LLM만 죽는다.** Matcha-TTS는 Pi 4 실측 RTF 0.411@4T로 실시간. 단 우리 `runtime.yaml`이 TTS에 **1스레드**만 줘서 CM4 환산 RTF ≈1.13 → **스테이지 배분 재검토 필요**(설정 문제, 보드와 무관) |
| 🚨 **기본 ASR 가중치 라이선스** | **미확인 → 확인됨, 문제 있음** | `models/sense-voice/LICENSE`가 링크 한 줄(`Ref to FunASR#license`). Apache/MIT 아님 — **FunASR Model Open Source License v1.1**(Alibaba). 비교용이 아니라 **기본값**이다. 제품화 전 법무 확인 |
| Smart Turn ARM 추론 시간 | **미측정 — 예산에 없음** | 우리 전제는 "CPU ~12ms"지만 Graviton 1 vCPU에서 **159ms** 보고. VAD 침묵 뒤 실행이라 턴 지연에 그대로 더해진다 → CM4 측정 항목에 추가 |
| ⚠️ **Smart Turn V2의 `ACC_incp` 62%** | **2차 — 제3자 측정** | Easy Turn 논문 표(RTX 4090). "incomplete"(생각 중간 멈춤) 판정이 **우리 적응형 grace가 정확히 의존하는 신호**다(`grace_frames_for_prob`). 단서: (a) 그들의 테스트셋, (b) **V2이고 우리는 v3.2**. 그래도 `complete_threshold: 0.5`가 아직 스톡값(`detector.py:50` "not tuned here")인 상황과 겹쳐 **임계값 스윕 우선순위가 올라간다** |
| 턴테이킹 학술 대안 (2025–2026) | **조사 완료 — 대안 없음 확인** | Easy Turn(공개·Apache-2.0인데 **850MB·263ms·2559MB @RTX4090**, 한국어 언급 없음), Phoenix-VAD(가중치 미공개, Qwen2.5-0.5B, 50ms @A6000), MuVAP(**카메라 필요**), 다국어 VAP(영·중·일, **한국어 없고 전이도 안 된다고 논문이 명시**). §2 결론이 **검증된 유지**로 승급 |
| 4-state 턴 어휘의 외부 근거 | **1차확인** | Easy Turn이 예측하는 것이 *complete / incomplete / backchannel / wait* — 우리 `TurnVerdict` 네 상태와 독립적으로 수렴. 트랙 C-1이 "TEN의 3개가 아니라 4개"로 간 판단의 외부 검증 |
| **라벨 없이 EOT 타깃 만들기** | **경로 확보 — 미착수** | Next-Turn: 학습 타깃을 *time-to-next-speech-onset*으로 두면 **"require no additional annotation"**(타임스탬프에서 유도), 320ms 내 정확도 **+25.9%p**. Thai EOT: **YODAS 자막**으로 비영어 EOT를 부트스트랩(종결어미 같은 언어별 단서 활용). → `labels.json`·"한 글자 네"가 **"라벨이 없어서 못 한다"는 더 이상 정확한 서술이 아니다** |
| CM5 캐리어 호환성 | **1차확인 — 데이터시트 원문** | 폼팩터 동일, **23핀 재배치**(Appendix B Table 14). 우리에게 유리: 최대 변경인 **CAM0·DSI0가 USB 3.0으로** 바뀌는데 카메라·DSI를 안 쓰므로 무해하고 USB 3.0 포트 2개를 얻는다. 주의 2건 — **ADC 2채널 소멸**(핀 94/96 → USB-C PD CC), **전력 5V 2.5A**. CM4를 안 사기로 했으므로 마이그레이션 비용이 아니라 **설계 입력** |
| CM5 = A76 @2.4GHz | **1차확인** | BCM2712 quad-core **Cortex-A76**, 2.4GHz, LPDDR4x-4267 ECC. §10.1의 근거(A72는 Armv8.0-A로 dotprod 없음 → llama.cpp 빠른 경로 전부 상실)에 대해 **A76은 Armv8.2-A로 dotprod 보유**. 클럭도 1.6배 |
| ⚠️ CM5 전력이 스레드 배분과 얽힌다 | **1차확인 — 신규** | 데이터시트: *"accommodate 5V at up to 2.5A"*, 완화책으로 *"lowering the CPU clock rate"*를 직접 제시. 우리 워크로드가 4코어를 다 쓰는 LLM 디코드라 피크가 실제로 걸릴 쪽 → `runtime.yaml` 스레드 배분이 성능 문제만이 아니라 **전력 문제이기도 하다** |
| PDF 텍스트 추출 | **해결 — 방법** | WebFetch가 CM5 데이터시트에서 구조 메타데이터만 뱉었다. 저장된 PDF에 `uv run --no-project --with pypdf`로 임시 의존성을 붙여 뽑았다. **`--no-project` 필수** — 프로젝트 안에서 그냥 `uv run`을 쓰면 `.venv`를 건드리려다 실패한다. §6-5의 "PDF 추출 실패"를 이 방법이 닫는다 |
| LLM 프리필 비용 (턴당) | **실측** | 고정 ~117ms + 토큰당 ~11.6ms (3스레드, 약 86 tok/s). 정적 프리픽스 1144토큰은 `warm_up()`이 이미 지불 → 턴당 남는 건 발화 3~16토큰. 동일 프롬프트 재호출은 **4ms** |
| 프리픽스 안정성 | **불변식** | ⚠️ 프롬프트 앞부분이 턴마다 바뀌면(시간 표시, 동적 메모리 위치 등) 위 4ms가 300ms로 돌아온다. 프리픽스를 안정적으로 유지하는 것이 진짜 레버다 |
| 턴 판정 어휘 통합 (`turn/verdict.py`) | **있음** (동작 무변경, 등가성 테스트로 고정) | `FINISHED/UNFINISHED/WAIT/EMPTY`. vad.py·talk.py의 기존 조건과 1:1로 같음을 테스트로 증명 |
| ~~**한 음절 발화가 EMPTY로 버려진다**~~ | ~~실측 — 결함~~ **→ 2026-08-19 수정됨** | 아래 `is_empty_transcript` 섀도잉 행으로 대체. 남은 미해결은 한 글자 "네"의 의미(맞장구 vs 응답)뿐 |
| 숫자·로마자 확장 (`korean_tn.expand`) — 텍스트 변환 | **있음** (63개 테스트) | 한자어/고유어 수사, 만·억 그룹, 유월·시월, 소수·퍼센트·전화번호, 두문자어 |
| 같은 것 — **로마자·고유어 수사 실제 발음** | **실측 개선** | 두 프리셋 모두: `ABC`→"엠비이"/"W파이 Premier번 on ABCia" 였다가 "에이비시"로. `23살`→"2십3 살"/"이샘살" 이었다가 "스물세 살"로 |
| 같은 것 — **한자어 수사 실제 발음** | **미검증** | ⚠️ ASR 라운드트립으로는 **원리적으로 검증 불가**. SenseVoice가 역정규화를 해서 올바른 "이십 분"을 "20분"으로 되돌린다 = 판정자가 검증 대상 변환을 되돌린다. `"만 이천 원"`이 `"120 원"`으로 돌아온 사례가 있는데 TTS 탓인지 ASR 탓인지 **듣지 않고는 알 수 없다** → 아래 사람 검증 목록으로 |
| 순수 로직 단위 테스트 | **있음** (337개, <3s) | `tests/` — 가중치·오디오 장치 불필요. 컨트롤러 상태기계, VadStream duration, chunker, memory 파싱/consolidation, storage 쿼리, `_AudioRing`, resample, 스레드 예산, backchannel/LocalAgreement/grace. 파라미터 실측과는 다른 축: 이건 "코드가 명세대로 동작"만 보증한다 |
| `barge_in_confirm_ms` (250) | 추정치 | 실제 맞장구/끼어들기 발화 녹음 필요 (`_calibrate_turn_params.py`) |
| `BACKCHANNEL_MAX_DURATION_S` | 추정치 | 위와 같음 — 단, 게이트에 들어가는 duration이 pre-roll 제외한 실발화 길이(`speech_duration_s`)라는 것 자체는 회귀 테스트로 고정(2026-08-14 전까지는 pre-roll 포함 길이가 들어가 게이트가 죽은 코드였다) |
| ~~`is_empty_transcript`의 `len <= 1` 섀도잉~~ | **수정 (2026-08-19)** | 한 글자라 `BACKCHANNEL_WORDS` 20개 중 **10개(네 넵 아 어 예 오 와 음 응 헐)가 도달 불가**였다 = **어시스턴트가 "네"를 못 들었다**. 그리고 "뭐?"·"왜?"를 침묵으로 버려 talk.py의 마이크 사망 경고를 거짓 발동시켰다. 이제 "짧은가"가 아니라 **"조각인가"**를 묻는다 — `ONE_SYLLABLE_WORDS = {뭐, 뭘, 왜}`(→FINISHED)와 `BACKCHANNEL_WORDS`(→WAIT)는 통과, 나머지 1글자는 그대로 EMPTY. 회귀 테스트로 **모든 맞장구 단어의 도달 가능성**을 고정 |
| 한 글자 "네"의 의미 | **미해결 — 정책 판단 필요** | 맞장구("응, 계속해")와 응답("네, 해줘")이 텍스트+길이로 구분 불가. 0.6s 이하는 WAIT(응답 안 함), 초과는 FINISHED로 두었다. 실제 라벨 녹음(`_calibrate_turn_params.py`)이 있어야 결정 가능 — 단 이전(EMPTY=마이크 사망 카운트)보다는 확실히 낫다 |
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
- 우리 캡처 셋 대부분이 0.6~0.9초라 **16개 중 **8개**가 finalize 전에 쓸 만한 걸 아무것도 커밋하지 못했다(하나는 `"."`만 커밋 = 없는 것과 같다).**
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
337개 테스트는 "코드가 명세대로 동작한다"만 보증하며 아래 어느 것도 대신하지 못한다.
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
      그리고 2026-08-19까지는 이 항목의 예시 중 **"어"·"응"이 아예 판정에 도달하지도
      못했다**(`len <= 1` 섀도잉 — 위 표). 즉 실제로 검증할 수 있게 된 것도 그때부터다.
      확인할 것: 짧은 "네"가 WAIT로 넘어가는 게 맞는가, 아니면 "네, 해줘"라는 응답을
      무시하는 것으로 느껴지는가 — 이게 지금 미결로 남긴 정책 질문이다.
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
