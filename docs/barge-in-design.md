# barge-in vs backchannel 구분 설계 문서

**상태: 설계만, 구현 없음.** `scripts/talk.py`의 barge-in(`play_async`/`on_speech_start`)은
지금 TEN-VAD가 "말하기 시작함"을 감지하는 즉시 재생을 끊는다. 이 문서는 그중 진짜
끼어들기(barge-in)와 맞장구(backchannel)를 구분해서, 맞장구에는 재생이 안 끊기게 만드는
설계다.

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

## 권장 설계: 지연-정지 + 사후 어휘 판정 (2단계)

### 1단계 — 지연-정지 (실시간, 재생 제어)

`on_speech_start`에서 바로 `sd.stop()`을 부르지 않는다. 대신 발화가 시작된 시점을 기록해두고,
**`BARGE_IN_CONFIRM_MS`(예: 400~450ms) 동안 계속 말하는 중이면 그때 가서 재생을 끊는다.**
backchannel은 대개 이 창 안에서 끝나므로 재생이 안 끊기고, 진짜 끼어들기는 대개 이 창을 넘겨서
자연스럽게 걸린다.

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
- **gray zone**: 1단계에서 이미 재생을 끊었는데(발화가 400ms 넘게 이어져서) 2단계에서
  보니 어휘상 backchannel이었던 경우(예: "그으으래?" 처럼 길게 끈 맞장구) — 이미 끊긴 재생은
  복구 안 함(오디오를 이어붙이는 건 이 프로젝트 스코프 밖). 이런 경우는 그냥 일반 턴으로
  처리한다 — 사용자가 뭔가 반응은 했으니 뭐라도 답하는 게, 아무 반응 없이 침묵하는 것보다
  나음. **알려진 한계로 문서화, 완벽한 해결책 아님.**
- 반대로 1단계를 통과 못 했는데(400ms 안에 끝나서 재생 안 끊음) ASR 결과가 backchannel이
  아니라 진짜 문장이었던 경우 — 이미 재생은 안 끊긴 채로 계속 흐르고 있었을 것이므로, 이번엔
  그냥 일반 턴으로 처리(재생과 겹쳐 들리는 건 감수 — 어차피 backchannel 판정 오류의 대칭적인
  반대쪽 케이스라 완전히 없앨 수 없음).

## 파라미터 초안 (실측 전 추정치, 확정 아님)

```yaml
# configs/vad.yaml에 추가
barge_in_confirm_ms: 400
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

- **했음**: 이 설계 문서
- **다음 단계**: 위 "구현 위치"대로 vad.py/backchannel.py/talk.py 구현 → 마이크로 실측 →
  파라미터 확정
