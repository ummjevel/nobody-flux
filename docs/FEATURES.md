# 기능 정의 문서

`nobody-flux`가 지금 실제로 하는 일, 명시적으로 안 하는 일, 그리고 다음에 붙일 만한 것들의 목록.
구현 배경/근거는 `docs/output/ondevice_asr_llm_tts_research_20260716.md`(모델 리서치)와
`docs/memory-design.md`(기억 설계)를 참고.

## 지금 구현된 것

### 파이프라인 (ASR → LLM → TTS)
- **ASR**: SenseVoice-Small (sherpa-onnx, 한국어) — `src/nobody_flux/asr.py`
- **LLM**: Qwen3-0.6B (transformers) — `src/nobody_flux/llm.py`, 페르소나는 `persona.py`
  ("퀜", 20~30대 또래 친구 톤, 반말)
- **TTS**: MOSS-TTS-Nano (voice-clone, 참조 음성 필요) — `src/nobody_flux/tts.py`, 자기 자신의
  격리된 venv(`external/MOSS-TTS-Nano/.venv`)에서 서브프로세스로 실행됨

### 모델 스왑 인프라
- `configs/models.yaml`에 스테이지별(asr/llm/tts) 프리셋을 등록. `src/nobody_flux/registry.py`가
  로드/생성.
- `scripts/run_pipeline.py`, `scripts/talk.py` 둘 다 `--asr/--llm/--tts <preset>`으로 선택 가능.
- LLM은 아직 스테이지당 프리셋 1개뿐. ASR/TTS는 두 번째 프리셋이 붙음:
  - ASR: `vibeasr-bitnet` (microsoft/VibeVoice-ASR-BitNet, arXiv:2607.21075 — CPU-only
    ggml/BitNet, ARM NEON 커널 포함해 CM4 타깃과 직결. `src/nobody_flux/asr.py`의
    `VibeAsrBitnet` 참고). 같은 테스트 wav에서 `sense-voice-small`의 어절 중간
    스페이스 문제가 없음 — `--asr vibeasr-bitnet`으로 바로 비교 가능. 매 턴 모델을
    재로딩하는 대신 상주 서버 프로세스(`asr_stream_server`)로 돌아가서 웜 상태에서
    ~2초대.
  - TTS: `freyatts-ko-voicea` (**기본값**. FreyaTTS 한국어 voiceA 체크포인트,
    voice-announce-mcp 프로젝트에서 가져옴. `src/nobody_flux/tts.py`의 `FreyaTtsKo`
    참고). `moss-tts-nano`와 달리 torch 버전을 고정하지 않아 CUDA(sm_120)가 실제로
    동작 — MOSS-TTS-Nano의 CPU 전용 50~96초 대비 웜 상태에서 ~1초대. 마찬가지로
    상주 서버 프로세스(`scripts/_freyatts_server.py`)로 돎.
- 그 외 후보 모델(Vosk, Gemma 4 E2B, Kokoro-82M 등)은 아직 구현 안 됨.

### 대화 경험
- `scripts/talk.py`: 상시 프로세스로 도는 연속 음성 루프. 한 번 띄우면 `NobodyLLM`의 히스토리가
  세션 내내 유지되어 실제 멀티턴 대화가 됨.
- 턴 경계는 `src/nobody_flux/vad.py`의 TEN-VAD (sherpa_onnx 내장, `configs/vad.yaml`로 임계값
  튜닝) — 버튼 누를 필요 없음. 마이크/환경마다 재튜닝이 필요하면 `scripts/_debug_vad_mic.py`로
  진단.
- **끼어들기(barge-in)**: 다음 발화 감지 대기가 이번 턴 응답 재생 시작과 동시에 시작됨(재생이
  끝날 때까지 기다리지 않음) — 응답 재생 중에 말을 시작하면 즉시 재생이 끊기고 그 발화가 다음
  턴이 됨. `scripts/talk.py`의 `play_async`/`on_speech_start` 참고. 에코 제거(AEC)는 없음 —
  스피커→마이크로 응답 소리가 새어 들어가 VAD가 오탐하는 환경에서는 자기 자신의 응답에 스스로
  끼어드는 것처럼 보일 수 있음 (이 프로젝트의 WSL2/WSLg 패스스루 환경에서는 관측 안 됨).
- `scripts/run_pipeline.py`: 기존 1회성 wav-in/wav-out CLI. 자동화 테스트, 프리셋 간 결정론적
  비교(같은 입력 wav로 latency/출력 비교)용으로 유지.

### 저장 & 기억(개인화)
- `src/nobody_flux/storage.py`: SQLite (`data/conversations.db`, 커밋 안 됨).
  - `sessions`: talk.py 세션 단위
  - `turns`: 매 턴의 user_text/reply_text, 사용된 프리셋, 스테이지별 소요시간(ms)
  - `memories`: 세션 종료 시 추출된 사실 (category/key/value/confidence)
- `src/nobody_flux/memory.py` + `talk.py` 연결 (`docs/memory-design.md` 참고):
  - 세션 종료 시 그 세션의 모든 turns를 한 번에 LLM에 넣어 JSON 배열로 사실을 추출
    (`extract_memories`) → `ConversationStore.save_memory`로 저장. 방어적 파싱(코드펜스/잡설이
    섞여도 첫 `[`~마지막 `]` 구간만 파싱, 실패하면 빈 배열로 취급)과 세션당 상한
    (`MAX_MEMORIES_PER_SESSION=10`)이 있음.
  - 세션 시작 시 `ConversationStore.recent_memories()`(confidence·최신순 상위 10개)를
    `format_recall_block()`으로 불릿 리스트 텍스트로 만들어
    `NobodyLLM`/`NobodyLLMGguf`의 `system_prompt_suffix`에 주입 (persona의
    `SYSTEM_PROMPT` 뒤에 붙음). 기억이 하나도 없으면(첫 실행) 빈 문자열이라 아무 변화 없음.
  - 중복 정리: 한 세션 안에서 같은 (category, key)가 여러 값으로 뽑히면 confidence 높은 쪽만
    남김 (`memory.py`의 `_dedupe_memories`, 상한 자르기 전에 적용). 세션을 넘나드는 중복(같은
    사실이 나중에 또 추출됨)은 `recent_memories`가 SQL window function으로 (category, key)별
    최고 confidence/최신 행만 recall (테이블 자체는 안 지움, 읽을 때만 접는 뷰).
  - 검증된 리스크: 0.6B급 모델이 사소해 보이는 사실을 자꾸 빈 배열로 건너뛰는 경향이 있어서,
    추출 프롬프트에 원샷 예시를 넣어야 했음 (`memory.py`의 `EXTRACTION_SYSTEM_PROMPT`).

### 벤치마크
- `scripts/benchmark.py`: 고정 테스트셋(`--wav-dir`, 기본 `data/benchmark_wavs/`, 커밋 안 됨 —
  직접 채워야 함)을 ASR×LLM×TTS 프리셋 조합마다 돌려서 스테이지별 평균 latency 표를 출력.
  `--asr/--llm/--tts`로 특정 프리셋만 좁힐 수 있고, 생략하면 등록된 전체 프리셋의 카티전 곱을
  돈다 (조합 수가 빠르게 커지니 주의). `--verbose`로 프리셋 조합별 user_text/reply_text도 같이
  출력 — 품질은 사람이 읽고 판단해야 해서 자동 채점은 안 함. 내부적으로
  `ConversationStore.turns_by_preset`/`turns_for_session`(둘 다 이 스크립트 전용으로 추가)이
  `turns` 테이블을 집계.

### 환경 세팅
- `scripts/setup_local.sh` (RTX 5090), `scripts/setup_server.sh` (H100) — 둘 다
  `scripts/setup_common.sh`를 공유. `uv sync`, GPU 확인, ASR 모델 다운로드, MOSS-TTS-Nano
  클론+격리 venv 생성까지 한 번에.

## 명시적으로 범위 밖 (이번 라운드)

- **웨이크워드 / 풀 스트리밍 ASR**: `vad.py`는 발화 단위 녹음→일괄 ASR이지, 실시간 스트리밍
  전사가 아님.
- **다른 후보 모델의 실제 구현**: ASR(`vibeasr-bitnet`)과 TTS(`freyatts-ko-voicea`)는 두 번째
  프리셋이 붙었지만, LLM은 여전히 레지스트리 인프라만 있고 두 번째 구현은 없음.

## 프리셋 추가하는 법 (모델 스왑 확장)

1. `src/nobody_flux/{asr,llm,tts}.py` 중 해당하는 파일에 새 클래스를 만든다. 기존
   `NobodyASR`/`NobodyLLM`/`NobodyTTS`와 같은 인터페이스를 맞춘다:
   - ASR: `transcribe_file(wav_path: str) -> str`
   - LLM: `reply(user_text: str) -> str` (+ `history`, `reset()`)
   - TTS: `synthesize(text: str, out_path: str) -> str`
2. `registry.py`의 `_CLASSES` 딕셔너리에 새 클래스를 추가한다 (임의의 문자열로 import되는 걸
   막기 위해 일부러 화이트리스트 방식).
3. `configs/models.yaml`의 해당 스테이지 아래에 프리셋을 추가한다 (`class`, `params`).
4. `--asr/--llm/--tts <새-프리셋-이름>`으로 바로 테스트.
5. 외부 모델/레포가 필요하면 `scripts/setup_common.sh`에 다운로드/클론 단계를 추가한다
   (MOSS-TTS-Nano 클론 부분이 참고 예시).

## 다음 단계로 제안하는 것 (구현 안 함, 우선순위 순 아님)

- 기억 정리/병합 로직 (중복·모순되는 항목이 쌓이는 문제, `docs/memory-design.md` 참고)
- CM4 실기에서의 실측 (리서치 문서가 애초에 요구했던 전제 — "모델 선정보다 CM4 실측 PoC가
  선행되어야 함")
