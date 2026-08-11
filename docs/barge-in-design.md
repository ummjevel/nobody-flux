# barge-in vs backchannel 구분 설계

> 응답 재생 중 사용자의 "맞장구(backchannel)"와 "진짜 끼어들기(barge-in)"를 구분하는 설계.
> **2단계 방식(지연-정지 + 사후 어휘 판정) 구현 완료.** Smart Turn v3는 backchannel엔 부적합해
> 엔드포인트 감지로 재활용.

## 요약

- **구현**: `vad.py`의 `barge_in_confirm_ms`/`on_barge_in_confirmed`(1단계) +
  `backchannel.py`의 `is_backchannel()`(2단계) + `pipeline.py`의 `should_continue_after_asr` 훅으로
  `talk.py`에 연결.
- **검증**: `is_backchannel()` 단위 케이스 + `pipeline.run()`이 backchannel 판정 시 LLM 미호출을
  end-to-end 확인 — 통과. **마이크 실사용 파라미터 튜닝은 아직**(WSL2 마이크 불안정, H100에서 할 것).
- **Smart Turn v3**: 원래 backchannel 필터 대체로 붙였으나 실측상 부적합(end-of-turn 모델) →
  엔드포인트 감지로 재활용(옵트인, 마이크 미검증).

## 왜 문제가 되나

`persona.py`의 퀜은 캐주얼 반말 페르소나라 사용자가 "어/응/오/진짜?/그렇구나" 같은 맞장구를
자주 낸다. `vad.py`의 `min_speech_duration`이 0.15초로 낮은 이유도 "이런 짧은 발화를 놓치지 말자"
였다. 그런데 초기 barge-in 구현은 그 짧은 발화가 감지되는 순간 무조건 재생을 끊어서 — 맞장구만
쳐도 퀜이 말을 하다 뚝 끊기는 어색한 경험이 됐다.

## 구분 신호 선택

| 신호 | 설명 | 채택 |
|---|---|---|
| **지속시간** | backchannel은 대개 200~400ms·1~2음절, 진짜 barge-in은 더 김 | ✅ 1차(실시간). 비용 0, VAD가 이미 가짐 |
| **어휘** | ASR 결과가 정해진 맞장구 단어("응/어/네/오/진짜/그렇구나")인지 | ✅ 2차(사후). ASR 후에만 앎, 프리셋 무관 재사용 |
| **운율(pitch/energy)** | 맞장구는 짧고 낮은 억양 | ❌ 별도 모델·특징추출 필요, 이 규모엔 과함 |

## 구현된 설계 — 2단계 (지연-정지 + 사후 어휘)

### 1단계 — 지연-정지 (실시간, 재생 제어)

`on_speech_start`에서 바로 `sd.stop()`하지 않는다. 발화 시작 시점을 기록하고,
**`barge_in_confirm_ms`(250ms) 동안 계속 말하는 중이면 그때 재생을 끊는다.** 맞장구는 대개 이 창
안에서 끝나 재생이 안 끊기고, 진짜 끼어들기는 창을 넘겨 자연스럽게 걸린다.

- `vad.py`: `listen_for_utterance`에 콜백 두 개 — `on_speech_start`(UI 로그) +
  `on_barge_in_confirmed`(지속시간 조건 충족 시 1회). `talk.py`는 후자에서만 `sd.stop()`.
- `configs/vad.yaml`: `barge_in_confirm_ms` (코드 안 건드리고 yaml만).
- 트레이드오프: 진짜 barge-in 반응이 250ms쯤 늦어짐. 대신 AEC 없는 환경의 오탐(자기 응답에 스스로
  끼어듦) 위험도 줄어 나쁘지 않은 교환.
- 값 근거: 초안 400ms → **250ms**(LiveKit 사례의 216ms 중앙값 참고, 아래 "관련 연구"). 여전히
  실측 전 추정치.

### 2단계 — 사후 어휘 판정 (ASR 후 turn 처리 여부)

발화가 끝나고 ASR이 돌면, 결과 텍스트가 backchannel 단어 목록에 속하고 길이도 짧으면
(`duration_s < BACKCHANNEL_MAX_DURATION_S`) **새 대화 턴으로 처리하지 않는다** — LLM/`log_turn`
스킵, 로그만 남기고 다음 청취로.

- `src/nobody_flux/backchannel.py`: `BACKCHANNEL_WORDS` 집합(normalize 후 정확 매칭) +
  `is_backchannel(text, duration_s)`. 순수 함수라 단위 테스트 쉬움.
- **gray zone(알려진 한계)**: 250ms 넘게 길게 끈 맞장구("그으으래?")는 1단계에서 이미 재생이
  끊긴 뒤라 복구 불가(오디오 이어붙이기는 스코프 밖) → 그냥 일반 턴으로 처리(침묵보다 나음).
  반대로 250ms 안에 끝난 진짜 짧은 문장은 재생 안 끊긴 채 흐름 → 겹쳐 들리는 건 감수. 판정
  오류의 대칭적 반대쪽이라 완전 제거는 불가.

### 파라미터 (실측 전 추정치)

```yaml
# configs/vad.yaml
barge_in_confirm_ms: 250
```
```python
# src/nobody_flux/backchannel.py
BACKCHANNEL_WORDS = {"어","어어","응","으응","네","넵","오","오오","헐","와",
                     "진짜","정말","그렇구나","그래","맞아","아하","음","아"}
BACKCHANNEL_MAX_DURATION_S = 0.6
```

확정 방법: `scripts/_debug_vad_mic.py`를 확장(맞장구/barge-in 샘플 녹음 → 지속시간 분포 실측)해
정한다. vad.yaml의 기존 threshold들도 같은 방식(추측 아닌 실측)으로 정해졌다.

## Smart Turn v3 실험 — backchannel엔 부적합, 엔드포인트로 재활용

"소형 오디오 분류기는 학습 데이터 필요 → 스코프 밖"이라던 전제가 틀렸다. pipecat-ai의 **Smart
Turn v3**가 이미 공개돼 있다(8M, int8 ONNX ~8.7MB, CPU ~12ms, 한국어 포함, BSD-2). 이걸 2단계
어휘 필터 대체로 붙여봤다(`src/nobody_flux/turn_detector.py`).

**실측 결과 backchannel 구분엔 안 맞았다** (합성 음성, prob_complete = "완결된 턴일 확률"):

| 발화 | 종류 | prob_complete |
|---|---|---|
| "응" | 맞장구 | 0.723 |
| "오늘 날씨 정말 좋다" | 진짜 턴 | 0.728 |
| "그래서 있잖아... 있었는데" | 진짜 턴(말끝 흐림) | 0.506 |
| "어" | 맞장구 | 0.588 |

맞장구 "응"(0.723)이 말끝 흐리는 진짜 문장(0.506)보다 높다. 이유는 명확 — Smart Turn은
**"말을 끝냈나(end-of-turn)"** 판단 모델이지 "맞장구냐 진짜 턴이냐"가 아니다. "응"은 완결된 발화라
높게 나오는 게 정상.

**결론**: 2단계에 넣으면 오히려 악화 → 안 넣음(어휘 필터 유지). 대신 원래 용도인 **엔드포인트
감지**로 재활용 — `listen_for_utterance(turn_detector=...)`가 TEN-VAD가 침묵으로 끊었을 때 바로
반환하지 않고 "완결 턴이냐 문장 중간 멈춤이냐" 물어, 미완결이면 `endpoint_grace_ms` 동안 더
기다려 이어붙인다("음... 그러니까..." 같은 멈춤이 별개 턴으로 잘리는 문제 완화). `talk.py
--endpoint-detect`로 켬(기본 꺼짐 — 실시간 누적 루프 마이크 미검증, `predict()`는 검증됨).

## 관련 연구/업계 사례 (참고)

업계는 대체로 **음향 신호(지속시간+운율+onset 형태)만의 소형 분류기**로 풀고 어휘 기반은 덜 쓴다.
이 프로젝트는 규모상 분류기 신규 학습은 스코프 밖이라 지속시간+어휘를 쓰되, 아래로 파라미터/기대치를
조정했다.

- **[LiveKit — Adaptive Interruption Handling](https://livekit.com/blog/adaptive-interruption-handling)**:
  오디오만으로(어휘 X) 86% precision/100% recall, VAD 대비 오탐 51%↓, 추론 30ms, **중앙값 216ms**로
  판단. → `barge_in_confirm_ms`를 400→250ms로 낮춘 근거.
- **[Krisp](https://krisp.ai/blog/turn-taking-for-voice-ai/)**: 6.1M CPU 모델로 end-of-turn은
  프로덕션화했지만 **backchannel 구분은 이들조차 "향후 과제"** → 원래 어려운 문제임을 확인(기대치
  설정).
- **[Kyutai Moshi](https://kyutai.org/Moshi.pdf)** / **[OpenAI GPT-Live](https://openai.com/index/continuous-voice-interaction-with-gpt-live/)**:
  풀-듀플렉스 종단간 음성 모델이라 turn detector 자체가 없음(턴 전환이 모델 내부에서 암묵 처리).
  이 프로젝트(캐스케이드 ASR→LLM→TTS)엔 직접 적용 불가 — "VAD를 지우자"가 아니라 "VAD가 필요
  없어지려면 아키텍처를 바꿔야" 한다는 결론. CM4 온디바이스엔 그런 대형 모델 비현실적.
- **[Amazon — Contextual Acoustic Barge-in](https://www.amazon.science/publications/contextual-acoustic-barge-in-classification-for-spoken-dialog-systems)**:
  Alexa도 진짜/가짜 barge-in을 오디오만으로 분류(LSTM, F1 +4.5%). 어휘보다 오디오 우선.
- **[Alibaba — Duplex Conversation](https://arxiv.org/pdf/2205.15060)**: 가짜 barge-in 4분류 —
  (1) 끼어들 의도 없음(맞장구 = 이 문서 케이스), (2) 소음, (3) 에코(`talk.py`가 "No AEC"로 문서화한
  한계), (4) 턴 오검출. (1)과 (3)이 서로 다른 범주임을 명확히 해, 디버깅 시 혼동 방지.
- **[EdgeAI 온디바이스 barge-in](https://www.runedge.ai/blog/barge-in-interruption-handling-on-device-voice)**:
  CPU 제약 환경 3단 필터(VAD → 스트리밍 ASR endpointing → 부분 transcript 소형 분류기). 단 **스트리밍
  ASR 전제** — 이 프로젝트 ASR은 발화 단위 일괄이라 부분 transcript가 없음 → 그래서 2·3단계를
  "발화 종료 후"로 미룬 게 이 문서의 2단계 사후 판정.

## 다음 단계

- **마이크 실측 튜닝**: `scripts/_calibrate_turn_params.py`(맞장구/barge-in 샘플을 라벨링 녹음 →
  지속시간 분포 → `barge_in_confirm_ms`/`BACKCHANNEL_MAX_DURATION_S` 제안·`--apply`)로 확정.
  Smart Turn 엔드포인트는 이제 **적응형**(`vad._grace_frames_for_prob`, `P(complete)`로
  `endpoint_grace_ms`↔`endpoint_grace_min_ms` 스케일 — Phase 2a) — `complete_threshold`/grace 범위도
  같은 스크립트류로 확정. WSL2 마이크 불안정 → macOS 네이티브 마이크 또는 H100에서.
- **훨씬 나중(스코프 밖)**: 풀-듀플렉스 종단간 음성 모델로 가면 VAD/turn detector 자체가 불필요.
  근본 아키텍처 변경이라 이 문서(캐스케이드 안 VAD 튜닝) 범위 밖.
