# 기능 정의 문서

`nobody-flux`가 지금 실제로 하는 일, 명시적으로 안 하는 일, 그리고 다음에 붙일 만한 것들의 목록.
구현 배경/근거는 `docs/output/ondevice_asr_llm_tts_research_20260716.md`(모델 리서치)와
`docs/memory-design.md`(기억 설계)를 참고.

## 지금 구현된 것

### 파이프라인 (ASR → LLM → TTS)
- **ASR**: SenseVoice-Small (sherpa-onnx, 한국어) — `src/nobody_flux/asr.py`
- **LLM**: Qwen3-0.6B (transformers) — `src/nobody_flux/llm.py`, 페르소나는 `persona.py`
  ("루카스", 20~30대 또래 친구 톤, 반말)
- **TTS**: MOSS-TTS-Nano (voice-clone, 참조 음성 필요) — `src/nobody_flux/tts.py`, 자기 자신의
  격리된 venv(`external/MOSS-TTS-Nano/.venv`)에서 서브프로세스로 실행됨

### 모델 스왑 인프라
- `configs/models.yaml`에 스테이지별(asr/llm/tts) 프리셋을 등록. `src/nobody_flux/registry.py`가
  로드/생성.
- `scripts/run_pipeline.py`, `scripts/talk.py` 둘 다 `--asr/--llm/--tts <preset>`으로 선택 가능.
- **지금은 스테이지당 프리셋이 1개씩뿐** (현재 쓰는 모델). 새 후보 모델을 코드로 구현한 뒤
  `configs/models.yaml`에 프리셋만 추가하면 바로 스위치 가능한 구조까지만 만들어져 있음 —
  후보 모델 자체(Vosk, Gemma 4 E2B, Kokoro-82M 등)는 아직 구현 안 됨.

### 대화 경험
- `scripts/talk.py`: 상시 프로세스로 도는 연속 음성 루프. 한 번 띄우면 `NobodyLLM`의 히스토리가
  세션 내내 유지되어 실제 멀티턴 대화가 됨.
- 턴 경계는 `src/nobody_flux/vad.py`의 간단한 에너지 기반 VAD (자동 침묵 감지) — 버튼 누를 필요
  없음. 임계값 튜닝이 필요할 수 있음 (마이크/환경마다 다름, `vad.py` 문서 참고).
- `scripts/run_pipeline.py`: 기존 1회성 wav-in/wav-out CLI. 자동화 테스트, 프리셋 간 결정론적
  비교(같은 입력 wav로 latency/출력 비교)용으로 유지.

### 저장
- `src/nobody_flux/storage.py`: SQLite (`data/conversations.db`, 커밋 안 됨).
  - `sessions`: talk.py 세션 단위
  - `turns`: 매 턴의 user_text/reply_text, 사용된 프리셋, 스테이지별 소요시간(ms) — 나중에 여러
    프리셋을 실측 비교할 때 그대로 쿼리해서 쓸 수 있음
  - `memories`: 스키마만 존재, 아직 아무것도 안 씀 (`docs/memory-design.md` 참고)

### 환경 세팅
- `scripts/setup_local.sh` (RTX 5090), `scripts/setup_server.sh` (H100) — 둘 다
  `scripts/setup_common.sh`를 공유. `uv sync`, GPU 확인, ASR 모델 다운로드, MOSS-TTS-Nano
  클론+격리 venv 생성까지 한 번에.

## 명시적으로 범위 밖 (이번 라운드)

- **웨이크워드 / 풀 스트리밍 ASR**: `vad.py`는 발화 단위 녹음→일괄 ASR이지, 실시간 스트리밍
  전사가 아님.
- **끼어들기(barge-in)**: TTS 재생 중엔 녹음 안 함. 사람이 응답 중간에 끼어들 수 없음.
- **다른 후보 모델의 실제 구현**: 레지스트리 인프라만 있고, 두 번째 ASR/LLM/TTS는 없음.
- **여러 프리셋 자동 비교 벤치마크 스크립트**: `turns` 테이블에 필요한 데이터(프리셋, ms)는
  이미 쌓이므로, 이 데이터를 모아 표로 뽑는 스크립트를 추가하면 됨 — 아직 없음.
- **기억 추출 로직**: 스키마만 있고 실제로 뽑아서 채우는 코드 없음.

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

- 여러 프리셋을 순회하며 고정 테스트셋으로 latency/품질을 표로 뽑는 벤치마크 스크립트
  (`turns` 테이블 데이터를 그대로 활용 가능 — 리서치 문서의 "다음 단계: 실측"과 직결됨)
- 기억 추출 실제 구현 (`docs/memory-design.md`)
- CM4 실기에서의 실측 (리서치 문서가 애초에 요구했던 전제 — "모델 선정보다 CM4 실측 PoC가
  선행되어야 함")
- 끼어들기(barge-in): TTS 재생 중 VAD가 계속 듣고 있다가 발화 감지되면 재생 중단
