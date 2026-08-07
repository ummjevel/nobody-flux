# barge-in vs backchannel 구분 설계 문서

**상태: 구현됨.** 아래 "권장 설계"에 적힌 그대로 구현: `vad.py`의
`barge_in_confirm_ms`/`on_barge_in_confirmed`(1단계, 지연-정지)와
`src/nobody_flux/backchannel.py`의 `is_backchannel()`(2단계, 사후 어휘 판정) +
`pipeline.py`의 `should_continue_after_asr` 훅으로 `talk.py`에 연결됨. 검증:
`is_backchannel()` 단위 케이스(짧은 맞장구/길게 끄는 맞장구/맞장구로 시작하는 실제 문장 구분)
+ `pipeline.run()`이 backchannel 판정 시 LLM을 실제로 호출 안 하는지 end-to-end 확인 — 둘 다
통과. 마이크 실사용 튜닝(파라미터 실측, "검증 계획" 절)은 아직 안 함 — 이 프로젝트 개발 환경
(WSL2)에서 마이크 테스트 자체가 불안정하다는 기존 제약(`talk.py` 문서) 때문.

## 왜 문제가 되나

`persona.py`의 퀜은 캐주얼 반말 대화 페르소나다. 사용자도 자연스럽게 "어", "응", "오",
"진짜?", "그렇구나" 같은 맞장구를 낼 것이고, `vad.py`의 `min_speech_duration`이 0.15초로
낮게 잡혀 있는 이유 자체가 "이런 짧은 발화도 놓치지 말자"였다 (vad.py 주석 참고). 그런데
barge-in 구현(`talk.py`)은 그 짧은 발화가 감지되는 순간 무조건 재생을 끊는다 — 결과적으로
사용자가 맞장구만 쳐도 퀜이 말을 하다 말고 뚝 끊기는 어색한 경험이 됨.

## 구분 신호 후보

| 신호 | 설명 | 이 프로젝트에서 쓸만한가 |
|---|---|---|
| **지속시간** | backchannel은 대개 200~400ms, 1~2음절. 진짜 barge-in(문장 단위)은 대개 더 김 | 계산 비용 거의 0, VAD가 이미 갖고 있는 정보. **1차 필터로 채택** |
| **어휘** | ASR 결과가 "응"/"어"/"네"/"오"/"헐"/"진짜"/"그렇구나" 같은 정해진 맞장구 단어 목록에 속하는지 | ASR이 이미 끝난 뒤에나 알 수 있음(사후 판정), 프리셋 교체와 무관하게 재사용 가능. **2차 필터로 채택** |
| **운율(pitch/energy contour)** | 맞장구는 대개 짧고 낮은 억양, 진짜 발화 시작은 다른 곡선을 가짐 | 별도 모델/특징 추출 필요, 이 프로젝트 규모에 비해 과함. **채택 안 함** |

## 관련 연구/업계 사례 조사

로컬 설계에 들어가기 전에 실제로 이 문제를 어떻게 풀고 있는지 찾아봤다. 결론부터: 프로덕션
음성 에이전트 업계는 대체로 **음향 신호(지속시간+운율+발화 개시 형태)만으로 학습된 소형
분류기**로 풀고 있고, 텍스트/어휘 기반은 오히려 덜 쓰인다 — 아래 근거 참고. 이 프로젝트
스케일에서는 분류기를 새로 학습시키는 건 스코프 밖이라 여전히 지속시간+어휘 조합을 쓰지만,
아래 조사 결과로 파라미터와 기대치를 조정했다.

- **[LiveKit — Adaptive Interruption Handling](https://livekit.com/blog/adaptive-interruption-handling)**:
  CNN 오디오 인코더로 발화 시작 후 첫 수백 ms의 "파형 모양, 발화 개시(onset)의 세기/날카로움,
  지속시간, 피치/리듬 같은 운율"만 보고 진짜 interruption과 backchannel/기침 등을 구분.
  **어휘(텍스트)는 아예 안 씀** — 오디오만으로 86% precision / 100% recall(500ms 기준
  overlap), 기본 VAD 대비 오탐 51% 감소, 추론 30ms, **중앙값 216ms의 오디오만으로 판단**
  달성. 이 프로젝트가 초안으로 잡았던 `barge_in_confirm_ms: 400`보다 실제 업계 기준은 훨씬
  짧다는 뜻 — 아래 "파라미터 초안"에 반영.
- **[Krisp — Turn-Taking model for Voice AI Agents](https://krisp.ai/blog/turn-taking-for-voice-ai/)**:
  610만 파라미터(65MB) 오디오 전용 CPU 모델로 end-of-turn(발화 종료) 예측은 이미 프로덕션에
  올렸지만, **backchannel 구분은 이 회사조차 아직 "향후 과제"로 남겨둔 상태** — 즉 완벽하게
  풀린 문제가 아니라는 걸 확인. 이 프로젝트의 지속시간+어휘 휴리스틱이 "간단해서 부족한 게
  아니라, 원래 어려운 문제를 간단한 방법으로 근사하는 것"이라는 기대치 설정에 참고.
- **[Kyutai — Moshi](https://kyutai.org/Moshi.pdf)** (RL 후처리 관련 소식,
  [AlphaSignal 요약](https://alphasignal.ai/news/kyutai-uses-reinforcement-learning-to-make-moshi-sound-actually-human)):
  발화자 턴을 아예 명시적으로 나누지 않고 사용자/모델 음성을 병렬 토큰 스트림으로 모델링,
  interruption/backchannel이 학습된 확률적 행동으로 "자연 발생". 강화학습 후처리로 barge-in
  타이밍과 backchannel 타이밍을 동시에 개선. 이 프로젝트의 캐스케이드(ASR→LLM→TTS) 구조와는
  전제 자체가 다른 접근(end-to-end 음성 모델)이라 직접 적용은 스코프 밖이지만, "barge-in과
  backchannel을 하나의 연속적인 판단으로 다뤄야 한다"는 방향성은 참고할 만함.
- **OpenAI — [GPT-Live](https://openai.com/index/continuous-voice-interaction-with-gpt-live/)**
  (3세대 ChatGPT 음성 시스템): Moshi와 같은 결론을 한 번 더 확인해줌 — "turn detector를
  오디오 경로에서 아예 제거"했다고 발표했는데, 이게 가능한 이유는 GPT-Live가 **풀-듀플렉스
  종단간 음성 모델**이라 계속 듣고 계속 말하면서 턴 전환 자체를 모델이 암묵적으로 처리하기
  때문. 이전 세대는 무음 기반 turn detection을 썼는데, 짧은 침묵이나 배경 소음도 "발화
  끝남"으로 오인해서 부자연스러운 타이밍에 끼어드는 문제가 있었다고 함 — 이 프로젝트가 VAD
  임계값을 실측으로 튜닝해야 했던 것과 같은 종류의 문제. **하지만 이건 이 프로젝트(캐스케이드
  ASR→LLM→TTS)에는 바로 적용 불가**: turn detector를 없애려면 ASR/LLM/TTS 세 스테이지를
  하나의 음성-투-음성 모델로 합쳐야 하는데, 그건 지금 프로토타입 스코프를 훨씬 넘어서는
  재설계이자 CM4 같은 온디바이스 타깃에 올리기도 비현실적임. "VAD를 지우자"가 아니라 "VAD가
  필요 없어지려면 아키텍처 자체가 바뀌어야 한다"는 게 정확한 결론.
- **Amazon Science — [Contextual Acoustic Barge-in Classification](https://www.amazon.science/publications/contextual-acoustic-barge-in-classification-for-spoken-dialog-systems)**:
  Alexa도 진짜/가짜 barge-in을 오디오만으로 분류하는 지도학습 모델을 씀 (LSTM 기반, baseline
  대비 F1 4.5% 개선). 여기서도 어휘보다 오디오 신호 우선.
- **Alibaba — [Duplex Conversation](https://arxiv.org/pdf/2205.15060)**: 가짜(false) barge-in을
  4가지로 분류함 — (1) 끼어들 의도가 없는 경우(맞장구, 인사 등 — 이 문서가 다루는 케이스),
  (2) 배경 소음, (3) 에코(로봇이 자기 목소리를 들음 — `talk.py`가 이미 "No echo cancellation"
  으로 문서화해둔 한계), (4) 잘못 잡힌 턴 경계. 이 분류 체계를 빌리면, 이 프로젝트는 지금
  (3)은 알려진 한계로 남겨두고 (1)을 이 문서로 풀려는 것 — 서로 다른 문제라는 걸 명확히
  구분해두는 게 나중에 "barge-in 안 되는 이유"를 디버깅할 때 헷갈리지 않게 해줌.
- **[EdgeAI — 온디바이스 barge-in](https://www.runedge.ai/blog/barge-in-interruption-handling-on-device-voice)**:
  CPU 제약 환경(이 프로젝트의 최종 타깃인 CM4와 같은 조건)에서는 3단 필터를 권장 — ① VAD로
  1차 게이팅(Silero VAD 예시, 1~2MB 모델로 30ms 청크를 1ms 미만에 처리) → ② 부분 ASR
  스트리밍 결과의 endpointing(후행 침묵/구두점) → ③ 부분 transcript의 의미를 보는 소형
  분류기로 "그만—" 대 "어, 어어" 구분. 다만 이건 **스트리밍 ASR을 전제**로 함 — 이 프로젝트의
  ASR(`sense-voice-small`/`vibeasr-bitnet`)은 발화 단위 일괄 처리라 부분 transcript가 없음
  (`vad.py` 문서의 "웨이크워드/풀 스트리밍 ASR" 범위 밖 항목과 직결). 그래서 이 프로젝트는
  ②③을 "발화가 다 끝난 뒤"로 미룰 수밖에 없다 — 아래 2단계 설계의 "사후" 판정이 그 타협.

### 조사 결과가 바꾼 것

1. `barge_in_confirm_ms` 초안을 400ms → **250ms**로 낮춤 (LiveKit의 216ms 중앙값 참고 —
   여전히 실측 전 추정치라는 점은 동일).
2. 어휘 기반 2단계 설계 자체는 유지하되, "이게 최선은 아니고 오디오 전용 분류기가 더 정확한
   업계 표준"이라는 걸 명시 — 나중에 이 프로젝트가 더 커지면 Krisp/LiveKit류 소형 오디오
   분류기 도입을 고려할 만한 지점으로 남겨둠 (아래 "다음 단계"에 추가).
3. "에코로 인한 가짜 barge-in"(`talk.py`가 이미 문서화한 한계)과 "맞장구로 인한 가짜
   barge-in"(이 문서)이 Duplex Conversation 논문 기준으로 서로 다른 범주라는 걸 명확히 함.

## 권장 설계: 지연-정지 + 사후 어휘 판정 (2단계)

### 1단계 — 지연-정지 (실시간, 재생 제어)

`on_speech_start`에서 바로 `sd.stop()`을 부르지 않는다. 대신 발화가 시작된 시점을 기록해두고,
**`BARGE_IN_CONFIRM_MS`(초안 250ms — LiveKit 사례의 216ms 중앙값 참고, 아래 "관련 연구" 절)
동안 계속 말하는 중이면 그때 가서 재생을 끊는다.** backchannel은 대개 이 창 안에서 끝나므로
재생이 안 끊기고, 진짜 끼어들기는 대개 이 창을 넘겨서 자연스럽게 걸린다.

- `vad.py`의 `listen_for_utterance`가 프레임 단위로 이미 "말하는 중" 상태를 추적하고 있으니,
  타이머 로직도 여기 두는 게 자연스럽다. `on_speech_start` 콜백 하나 대신
  `on_speech_start`(UI용, 기존 그대로) + `on_barge_in_confirmed`(신규, 지속시간 조건 충족 시
  1회 호출) 두 개를 받도록 확장.
- `configs/vad.yaml`에 `barge_in_confirm_ms` 필드 추가 (다른 임계값들과 같은 "코드 안 건드리고
  yaml만" 원칙).
- 트레이드오프: 진짜 barge-in의 반응 속도가 400~500ms 늦어짐. AEC 없는 이 프로젝트 환경에서는
  오탐(자기 응답에 스스로 끼어드는 것) 위험을 줄이는 효과도 같이 있어서 나쁘지 않은 교환.

### 2단계 — 사후 어휘 판정 (ASR 결과 기반, turn 처리 여부 결정)

발화가 다 끝나고(무음 감지) ASR이 돌고 나면, 그 결과 텍스트가 backchannel 단어 목록에
속하고 원본 발화 길이도 짧으면(예: `duration_s < 0.6`) **새 대화 턴으로 처리하지 않는다** —
LLM 호출 안 하고, `store.log_turn`도 안 하고, 그냥 로그만 남기고 다음 `listen_for_utterance`로
넘어간다.

- 새 모듈 `src/nobody_flux/backchannel.py`: `BACKCHANNEL_WORDS` 집합(정확 매칭용, ASR이
  구두점/공백을 어떻게 내는지 감안해 normalize 후 비교) + `is_backchannel(text: str, duration_s:
  float) -> bool`. `memory.py`처럼 순수 함수라 단위 테스트하기 쉬움.
- **gray zone**: 1단계에서 이미 재생을 끊었는데(발화가 250ms 넘게 이어져서) 2단계에서
  보니 어휘상 backchannel이었던 경우(예: "그으으래?" 처럼 길게 끈 맞장구) — 이미 끊긴 재생은
  복구 안 함(오디오를 이어붙이는 건 이 프로젝트 스코프 밖). 이런 경우는 그냥 일반 턴으로
  처리한다 — 사용자가 뭔가 반응은 했으니 뭐라도 답하는 게, 아무 반응 없이 침묵하는 것보다
  나음. **알려진 한계로 문서화, 완벽한 해결책 아님.**
- 반대로 1단계를 통과 못 했는데(250ms 안에 끝나서 재생 안 끊음) ASR 결과가 backchannel이
  아니라 진짜 문장이었던 경우 — 이미 재생은 안 끊긴 채로 계속 흐르고 있었을 것이므로, 이번엔
  그냥 일반 턴으로 처리(재생과 겹쳐 들리는 건 감수 — 어차피 backchannel 판정 오류의 대칭적인
  반대쪽 케이스라 완전히 없앨 수 없음).

## 파라미터 초안 (실측 전 추정치, 확정 아님)

```yaml
# configs/vad.yaml에 추가
barge_in_confirm_ms: 250
```

```python
# src/nobody_flux/backchannel.py 초안
BACKCHANNEL_WORDS = {"어", "어어", "응", "으응", "네", "넵", "오", "오오", "헐", "와",
                      "진짜", "정말", "그렇구나", "그래", "맞아", "아하", "음", "아"}
BACKCHANNEL_MAX_DURATION_S = 0.6
```

실제 임계값은 `scripts/_debug_vad_mic.py`를 확장해서(백채널 샘플 여러 개 녹음 → 지속시간
분포 실측) 확정할 것 — vad.yaml의 기존 threshold들도 같은 방식(추측 아니고 실측)으로
정해졌다.

## 구현 위치 요약

- `vad.py`: `listen_for_utterance`에 `on_barge_in_confirmed` 콜백 추가 (지속시간 타이머)
- `configs/vad.yaml`: `barge_in_confirm_ms`
- `src/nobody_flux/backchannel.py` (신규): 어휘 목록 + `is_backchannel()`
- `talk.py`: `on_speech_start`는 UI 로그만 남기도록 남겨두고, `on_barge_in_confirmed`에서
  `sd.stop()` 호출. 턴 처리 직전에 `is_backchannel(user_text, duration_s)` 체크해서 참이면
  `continue`(로그만) — LLM 호출/저장 스킵.

## 검증 계획

1. `scripts/_debug_vad_mic.py`로 backchannel 샘플(어, 응, 오, 진짜? 등) 여러 개와 진짜
   barge-in 시작 문장(전체 문장) 여러 개를 각각 녹음해서 지속시간 분포를 실측 —
   `barge_in_confirm_ms` 기본값을 이 실측치로 확정.
2. `scripts/talk.py`를 마이크로 직접 돌리면서 응답 재생 중에 backchannel/barge-in을 섞어
   말해보고, 재생이 의도대로 (안) 끊기는지 수동 확인 (자동화된 오디오 테스트는 이 프로젝트
   스코프 밖 — run_pipeline.py/benchmark.py도 wav 파일 기반이지 실시간 오디오 상호작용은
   테스트 대상이 아님).

## 다음 단계

- **했음**: 이 설계 문서, 관련 연구/업계 사례 조사, vad.py/backchannel.py/pipeline.py/talk.py
  구현, 단위 테스트 + end-to-end 스모크 테스트
- **다음 단계**: 위 "검증 계획"대로 마이크로 실측(`scripts/_debug_vad_mic.py` 확장) →
  `barge_in_confirm_ms`/`BACKCHANNEL_WORDS`/`BACKCHANNEL_MAX_DURATION_S` 파라미터 확정.
  이 프로젝트 개발 환경(WSL2)에서 마이크 테스트가 불안정할 수 있어(`talk.py` 문서 참고),
  H100 서버(네이티브 Linux)에서 시도할 것.
- **더 나중에(스코프 밖, 참고용)**: 지속시간+어휘 휴리스틱으로도 부족하면, Krisp/LiveKit류
  소형 오디오 전용 분류기(수백만 파라미터, CPU에서 30ms 이내 추론) 도입을 고려. 다만 학습
  데이터 수집·훈련이 필요해서 이 프로젝트 지금 단계(프로토타입 검증)보다는 CM4 실기 검증
  이후, 실사용 데이터가 쌓인 다음이 더 적절한 시점.
- **훨씬 더 나중에(완전히 다른 아키텍처, 스코프 밖)**: OpenAI GPT-Live/Kyutai Moshi처럼
  ASR/LLM/TTS 세 스테이지를 통째로 풀-듀플렉스 종단간 음성 모델 하나로 대체하면 VAD/turn
  detector 자체가 필요 없어짐. 이건 "barge-in 튜닝"이 아니라 이 프로젝트의 근본 아키텍처를
  바꾸는 얘기라 완전히 별개 스코프 — CM4 같은 온디바이스 타깃에 그 정도 크기의 종단간 모델을
  올리는 것도 현실성이 낮음. 참고만 하고 이 문서의 범위(캐스케이드 구조 안에서 VAD 튜닝)는
  유지.
