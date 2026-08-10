# 기능 정의

> `nobody-flux`가 지금 실제로 하는 일 / 안 하는 일 / 다음에 붙일 것. 세부 설계는 각 절이 링크하는
> 설계 문서 참고(기억 `memory-design.md`, barge-in `barge-in-design.md`, TTS 표현력
> `tts-expressivity-design.md` 외 3부작, 모델 리서치 `output/ondevice_asr_llm_tts_research_20260716.md`).

## 파이프라인 & 현재 기본값

캐스케이드 **ASR → LLM → TTS**. 어느 모델이 도는지는 코드가 아니라 `configs/models.yaml`이 결정
(`registry.py`가 로드). `run_pipeline.py`/`talk.py` 둘 다 `--asr/--llm/--tts <preset>`로 선택.

| 스테이지 | 현재 기본값 | 구현 |
|---|---|---|
| ASR | `sense-voice-small` | `src/nobody_flux/asr.py` |
| LLM | `qwen3-0.6b-gguf` | `src/nobody_flux/llm.py` (페르소나 `persona.py` — "퀜", 20~30대 또래 반말) |
| TTS | `sherpa-matcha-ko` | `src/nobody_flux/tts.py` (ONNX·CPU, CM4 타깃에 맞아 기본값으로 선택) |

## 프리셋 (모델 스왑)

**ASR**
- `sense-voice-small` (기본) — SenseVoice-Small, sherpa-onnx, 한국어.
- `vibeasr-bitnet` — microsoft/VibeVoice-ASR-BitNet. CPU-only ggml/BitNet, ARM NEON 커널로 CM4 직결.
  상주 서버(`asr_stream_server`)로 웜 ~2초대.
- `streaming-zipformer-ko` — k2-fsa 스트리밍 Zipformer 한국어(Apache-2.0, int8). 지금은 **드롭인 배치
  디코드**로만 씀. 실측(합성 wav): sense-voice와 속도 비슷(~130ms)·내용 정확하나 어절 스페이스 없음.
  진짜 이득(라이브 부분 transcript+엔드포인팅)은 아직 미사용.

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
- **턴 경계 = TEN-VAD** (`vad.py`, sherpa_onnx 내장, `configs/vad.yaml`로 튜닝). 버튼 불필요.
  재튜닝은 `scripts/_debug_vad_mic.py`로 진단.
- **끼어들기(barge-in)**: 다음 발화 감지 대기가 응답 재생과 동시에 시작 → 재생 중 말하면 끊김
  (`play_async`/`on_barge_in_confirmed`). 에코 제거(AEC) 없음(WSL2 환경선 오탐 미관측).
  - **backchannel("어"/"응") 구분** (`docs/barge-in-design.md`): ① 지연-정지(`barge_in_confirm_ms`
    250ms 이상 지속돼야 재생 끊음) + ② 사후 어휘 판정(`backchannel.py`의 `is_backchannel()`이 맞장구면
    `pipeline.py`의 `should_continue_after_asr` 훅으로 LLM/TTS/저장 스킵). 파라미터는 실측 전 추정치.
  - **엔드포인트 감지(옵션 `--endpoint-detect`)**: Smart Turn v3(`turn_detector.py`, 8M ONNX, CPU ~12ms)로
    "말 끝났나 vs 문장 중간 멈춤"을 판단해 자연스러운 멈춤이 잘리는 걸 완화. 기본 꺼짐(실시간 루프
    마이크 미검증). 원래 backchannel용이었으나 실측상 end-of-turn 모델이라 엔드포인트로 재활용.
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

`scripts/setup_local.sh`(RTX 5090)·`setup_server.sh`(H100) → 공용 `setup_common.sh`. `uv sync`, GPU 확인,
모델 다운로드(SenseVoice·TEN-VAD·GGUF·Matcha·Smart Turn·스트리밍 Zipformer 등), 격리 venv 생성까지.

## 명시적으로 범위 밖

- **웨이크워드 / 풀 스트리밍 ASR**: `vad.py`는 발화 단위 녹음→일괄 ASR. `streaming-zipformer-ko`도
  아직 배치 디코드로만 사용(라이브 스트리밍 미구현).
- **디바이스용 표현력 TTS**: 한숨·웃음 등 NVV는 아직 어느 프리셋도 기본 지원 안 함(`docs/tts-*` 참고).

## 프리셋 추가하는 법

1. `src/nobody_flux/{asr,llm,tts}.py`에 새 클래스 추가 — 기존 인터페이스에 맞춤:
   ASR `transcribe_file(wav)->str` / LLM `reply(text)->str`(+`history`,`reset`) / TTS `synthesize(text,out)->str`.
2. `registry.py`의 `_CLASSES`에 등록(화이트리스트 방식).
3. `configs/models.yaml`의 해당 스테이지에 프리셋 추가(`class`,`params`).
4. `--asr/--llm/--tts <이름>`으로 테스트.
5. 외부 모델/레포 필요 시 `scripts/setup_common.sh`에 다운로드/클론 단계 추가.

## 다음 단계

- **TTS 표현력(NVV)** — TTS 3부작 문서(`tts-expressivity-design.md`/`tts-small-expressive-research.md`/
  `tts-conversational-build-design.md`) 참고. 방향: 대형 GPU 표현 모델은 서버 teacher로만, 디바이스는
  경량 CPU 모델(FreyaTTS 등)에 한국어 NVV 데이터+조건화를 얹기. CosyVoice2 한국어 NVV 실측은
  서버(H100) 작업으로 보류.
- **barge-in/엔드포인트 파라미터 마이크 실측** (`docs/barge-in-design.md`) — WSL2 마이크 불안정 →
  H100에서.
- **CM4 실기 실측** — 리서치 문서가 요구한 전제("모델 선정보다 CM4 PoC 선행").
