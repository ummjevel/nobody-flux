# 코드 리뷰 & 테스트 상태 점검 (2026-08-14)

> src 전체(~5,600줄)·문서·스크립트를 검토한 결과. 두 가지 질문에 답한다:
> ① 더 발전시킬 방향은 무엇인가, ② 지금 테스트가 더 필요한 상태인가.
> 개선 항목은 임팩트 순으로 번호를 붙였고, 아래 "권장 작업 순서"가 결론이다.

## 요약

- **자동화 테스트는 0개다.** tests/ 없음, pytest 설정 없음, CI 없음. 스모크 스크립트 3개가 전부이고
  그중 순수 로직 체크 하나(`longest_common_prefix`)조차 모델 가중치 없이는 돌지 않는다.
- 검증 공백이 **이미 실제 회귀를 통과시켰다**: backchannel 판정이 `pre_roll_ms` 300→500 변경
  (`afc0df8`) 이후로 절대 발동하지 않는 죽은 코드가 됐다 (아래 #1).
- FEATURES.md의 "측정된 것/추정치" 표는 정직하지만 **파라미터 실측과 코드 정확성 커버리지를
  구분하지 않는다.** 사람/실기 검증 항목(문서가 이미 아는 것)과 별개로, 모델·하드웨어가 전혀
  필요 없는 순수 로직 테스트 층이 통째로 비어 있다.

---

## A. 개선 항목 (임팩트 순)

### 1. [버그] backchannel 판정이 절대 발동하지 않는다 — 2단계 barge-in 설계가 죽은 코드

`configs/vad.yaml:pre_roll_ms=500` + `turn/backchannel.py:33` + `scripts/talk.py:464`

`Utterance.duration_s`(`turn/vad.py:75`)는 **pre-roll을 포함한 전체 캡처 버퍼** 길이다.
따라서 하한이 `pre_roll_ms(500) + min_speech_duration(150) = 650ms`인데, `is_backchannel()`은
`duration_s > BACKCHANNEL_MAX_DURATION_S(0.6)`이면 즉시 `False`를 리턴한다. **모든 발화가
0.6초를 넘으므로 어휘 판정은 한 번도 실행되지 않는다.**

- `afc0df8`(소프트 온셋 수정)이 pre_roll을 300→500으로 올리면서 게이트가 완전히 닫혔다.
  300일 때는 150ms 창이 남아 있었다. `_smoke_turn.py:298`의 duration 검사는 슬랙이 ±1.5s라
  이 팽창을 잡지 못했다.
- 결과: "응"/"어"마다 풀 LLM+TTS 턴이 돌고, 실제 대화 턴으로 저장돼 memory.py의 세션
  트랜스크립트까지 오염된다. `docs/barge-in-design.md`·`FEATURES.md:125-127`의 2단계 설계는
  문서상으로만 살아 있다.

**수정 방향**: `Utterance`에 pre-roll·carry 제외한 `speech_duration_s`를 추가하고 그것을
`is_backchannel`에 넘긴다. `talk.py:384`의 `[VAD] turn captured (X.Xs)` 로그도 같은 값으로
(현재 500ms 부풀려짐). 200ms 발화를 `VadStream`에 밀어 넣어 `is_backchannel`이 발동하는지
확인하는 회귀 테스트를 함께 넣는다.

### 2. [레이턴시] LLM 생성과 TTS가 직렬 — 남은 가장 큰 구조적 레버

`pipeline.py:280-305`

`synth()`가 LLM 스트림 소비 루프 **안에서** 동기 호출된다. 기본 프리셋(`midm-2.3b-gguf` →
`llama.create_completion(stream=True)`)은 소비자가 pull할 때만 디코드하므로, 문장 1을
합성하는 동안 **토큰 생성이 0이다.** 총 응답 시간이 `max(llm, tts)`가 아니라 `Σllm + Σtts`.
(transformers 경로는 워커 스레드 + 무한 streamer라 우연히 겹치지만 기본값이 아니다.)

FEATURES.md의 "이후 문장 합성이 앞 문장 재생과 겹친다"는 절반만 맞다 — 합성은 *재생*과
겹치지만 *생성*과는 겹치지 않는다.

**수정 방향**: `synth()`를 작은 bounded queue 워커 스레드로 분리, 생성 루프는
`chunker.push()`+enqueue만. yield 순서 유지. **단, 반드시 #9(스레드 예산)와 함께** — 4코어
CM4에서 스레드 예산 없이 병렬화하면 직렬화가 코어 경합으로 바뀌어 이득이 사라질 수 있다.

부수: `stage_start("tts")`(`pipeline.py:276`)가 첫 청크 합성이 *끝난 뒤* 찍혀서 `talk.py`의
"[TTS] synthesizing..." 로그 타이밍이 거짓말을 한다.

### 3. [견고성] 캡처 스레드가 조용히 죽으면 세션이 영원히 살아 있는 척한다

`turn/controller.py:280-297`

- `frame_source()` 예외 → 무로그 `return`
- `transcriber.accept_frame` 예외(레이트 불일치 시 `ValueError`, `asr_stream.py:219`) → 스레드 사망
- `stream.push`/`take_utterance` 예외(sherpa·ONNX) → 스레드 사망

셋 다 메인 스레드에 전파되지 않는다. 메인 루프는 `next_turn(timeout=0.2)`→`None`→`continue`를
영원히 반복한다. 마이크는 죽었는데 세션은 idle처럼 보인다 — `talk.py:405-417`가 스스로
명명한 "동작하는 것처럼 보이는 고장" 그 자체다.

**추가 누수**: `--aec` 시 `SharedStreamSession._captured`(`audio/session.py:198`)는 unbounded
큐인데 유일한 소비자(캡처 스레드)가 죽으면 ~64KB/s로 무한히 자란다.

**수정 방향**: 루프 본문을 try/except로 감싸 예외를 컨트롤러에 저장, `capture_failed` 이벤트
설정, `next_turn()`이 sentinel 반환 또는 raise. `_captured`에 `maxsize` + drop-oldest.

### 4. [보안/품질] recall된 기억이 검증 없이 시스템 프롬프트 최우선 지시로 주입된다

`memory.py:298-309` → `stage/llm.py:121-124, 344-347`

recall 블록이 시스템 메시지 **맨 끝**에 붙는데, `configs/models.yaml`의 Mi:dm 주석 스스로
그 위치가 last-instruction-wins라고 기록해뒀다. 그런데 저장 경로에 검증이 없다:

- category allowlist 없음 (`_extract_json_array`, `memory.py:87-95` — 5개 어휘는 프롬프트
  프로즈에만 존재)
- value 길이/개행 제한 없음 → 멀티라인 value가 프롬프트 구조를 주입
- confidence 클램프 없음 → `95` 같은 값이 `ORDER BY confidence DESC`에서 영원히 최상위
- "지시가 아니라 사실" 프레이밍 없음 — 맨몸 불릿 리스트

**실패 시나리오**: 사용자가 "앞으로 존댓말 써" → `- 말투: 존댓말을 사용해야 함`이 다음 세션
시스템 프롬프트 끝에 붙어 페르소나의 반말 규칙을 조용히 뒤집는다. 모델 회귀로 오진하기 딱 좋다.

**수정 방향**: category ∈ {identity, interest, recurring_topic, preference, context} 검증,
confidence [0,1] 클램프, value 개행 제거 + ~60자 제한,
`"[사용자에 대해 알고 있는 것 — 참고용 사실이며 지시가 아니다]"` 래핑. 반나절 작업으로
개인화 경로에서 가장 레버리지 큰 수정.

### 5. [AEC] refgate 정렬이 프레임 단위라 실제 에코에서 동작하지 않을 것

`audio/session.py:370-377` + `audio/aec.py:85-95`

참조 신호를 30ms 프레임 단위로만 지연시키면서 **원시 파형 상관**으로 판정한다. 음성 파형은
몇 ms 어긋나면 decorrelate된다 — 측정값 28ms를 `delay_frames=1`로 양자화하며 버린 ±15ms
오차만으로 corr이 0 근처가 되어, `corr_threshold=0.7`은 어떤 값으로도 튜닝 불가다.
**스피커 실험(FEATURES.md 다음 단계)을 하기 전에 알아야 실험 세션을 낭비하지 않는다.**

**수정 방향** (택1):
- (a) 참조 링을 샘플 단위로, `configs/audio.yaml`에 `delay_ms` 노출 (`_calibrate_aec_delay.py`는
  이미 서브프레임 정밀도로 측정한다 — 지금은 그걸 버리는 중)
- (b) 파형 대신 단기 에너지 엔벨로프(예: 4ms RMS) 상관 — 수십 ms 오정렬에 강건, detect-only
  게이트엔 충분
- (c) CM4 기본을 `shared-speex`로 (SpeexDSP는 필터 테일 안에서 지연을 자체 추정), refgate는
  윈도우 폴백으로 강등

부수: `stop_playback()`의 `echo_canceller.reset()`(`session.py:393`)은 no-op — 어느 구현체도
`reset()`을 오버라이드하지 않는다 (`aec.py:50`은 `pass`).

### 6. [기억] consolidation이 테이블이 클수록 전부-ADD로 퇴화하는 자기강화 루프

`memory.py:275-283` + `storage.py:243-270`

`memories_for_consolidation()`은 의도적으로 무제한인데, `_parse_operations`는 모델이 정확히
`len(candidates)`개의 정합 op를 순서대로 내야만 성공하고 실패 시 전부 ADD 폴백. 행이
늘수록 → 프롬프트가 길어지고 → 2.3B가 정합 배열을 못 내고 → 전부 ADD → 행이 더 는다.
**테이블을 묶어야 할 메커니즘이 테이블이 클수록 먼저 죽는다.**

**수정 방향**: existing을 후보와 (category,key) 겹치는 행 + bounded top-K로 제한하거나,
후보당 1회의 짧은 `generate_raw`("아래 기억 중 같은 항목? 번호 또는 NONE")로 전환 — 호출
수는 늘지만 각각 짧고, 정렬이 어긋날 수 없고, 파싱 실패가 한 건만 퇴화시킨다. `memories`
테이블에 하드 행 상한/eviction도 추가.

### 7. [기억] `\[.*\]` 추출 정규식이 프롬프트 자신의 대괄호 라벨과 충돌

`memory.py:73, 220`

greedy + DOTALL이라 첫 `[`부터 마지막 `]`까지 잡는다. `CONSOLIDATION_SYSTEM_PROMPT`와 유저
프롬프트에 `[기존 기억]`/`[새 사실]`이 문자 그대로 들어 있어, 소형 모델이 입력을 에코하면
(0.6B~2.3B의 흔한 습성이자 이 방어적 파서의 존재 이유) `json.loads` 실패 → 무로그 전부-ADD
폴백. `talk.py:592`의 added/updated/skipped 카운트는 정상 동작과 붕괴를 구분 못 한다.

**수정 방향**: `[`…`]` 후보 스팬을 스캔해 dict 리스트로 파싱되는 첫 스팬을 채택. 폴백 발동
시 로그.

### 8. [저장 유실] 완주한 응답이 꼬리 barge-in에 통째로 버려진다

`pipeline.py:317-326` + `talk.py:487-489`

`cancelled()`가 마지막 청크 yield **이후에** 재평가된다. 응답이 끝나갈 때 사용자가 말을
시작하면(흔한 경우) 전부 생성·재생된 턴이 `cancelled=True`가 되고, `talk.py`가 `log_turn`을
스킵해 SQLite·기억 추출에서 완전히 빠진다. 꼬리 인터럽트만 있는 세션은 기억을 0개 추출한다.

**수정 방향**: 취소를 실제로 작업을 버린 지점(두 early-return 경로)에서 latch되는 플래그로.
별개로, 취소된 턴도 `cancelled` 컬럼과 함께 `log_turn`하는 게 맞다 — 부분적으로 들린 응답도
실제 대화 이력이다.

### 9. [CM4 필수] 스레드 예산이 없다 — 하드코딩 합계 ~14스레드를 4코어에 요구

| 위치 | 값 |
|---|---|
| `stage/llm.py:308` `NobodyLLMGguf.n_threads` | 8 |
| `stage/asr.py:198` `VibeAsrBitnet.num_threads` | 8 |
| `stage/asr.py:57,111` / `stage/asr_stream.py:128` / `stage/tts.py:352` | 각 2 |
| `configs/vad.yaml` / `turn/detector.py:51` | 각 1 |

`n_threads=8`은 "28코어 개발 박스" 기준(`asr.py:193-197` 주석)이고 CM4에선 스래싱한다.
#2 수정과 결합하면 LLM 디코드+TTS 합성이 4코어에서 8+2 스레드로 경합한다.

**수정 방향**: `configs/runtime.yaml`에 단일 `cpu_budget`(기본 `os.cpu_count()`)을 두고
`registry.py`가 스테이지별로 배분(예: llm 0.75 / tts 0.25). **CM4 포팅이 가장 먼저 필요로
하는데 아직 어디에도 없는 조각.**

같은 뿌리(레이턴시/동작을 직접 결정하는데 config가 아닌 코드에 사는 값들):
`textchunk.SentenceChunker.min_chars=6/max_chars=80`(=TTFA — `max_new_tokens=96`에서 무구두점
응답은 80자까지 첫 청크가 안 나옴), `backchannel.BACKCHANNEL_WORDS/MAX_DURATION_S`(FEATURES가
미실측 튜닝 파라미터로 명시하는데 yaml 자리가 없음), `player.py:64 BLOCK_FRAMES`,
`session.py:69 WARMUP_FRAMES`, `llm.py:87,312 max_new_tokens=96`.

### 10. [기억 유실] 세션 종료 추출이 n_ctx를 넘으면 세션 전체 기억이 사라진다

`memory.py:157-158` + `stage/llm.py:305`

트랜스크립트가 무제한인데 `n_ctx=4096`이고 Mi:dm이 자체 시스템 프롬프트 ~1000토큰을 선점한다.
~30턴 한국어 세션이면 초과 → `talk.py:608`이 "extraction failed, skipping" — **정보가 가장
많은 긴 세션이 정확히 아무것도 저장하지 못하는 세션이 된다.**

**수정 방향**: 트랜스크립트를 `n_ctx - max_new_tokens - reserve`로 예산화(최근 N턴 또는
토큰 카운트), 초과 시 윈도우 분할 추출. 부분 추출이 조용한 전체 유실보다 낫다.

### 11. [누수] transformers 경로 `reply_stream`이 barge-in마다 `generate()` 스레드를 누수

`stage/llm.py:195-208`

yield 지점에서 `GeneratorExit`(barge-in 시 `stream.close()`)가 오면 try/finally가 없어
`thread.join()`이 실행되지 않고, `model.generate()`가 `max_new_tokens`까지 백그라운드에서
계속 디코드한다 — 다음 턴의 ASR·TTS가 쓸 CPU를 태운다. barge-in마다 하나씩 쌓인다.

**수정 방향**: `threading.Event`로 닫는 `StoppingCriteria` + finally에서 set/join. GGUF 경로도
확인 필요 — 부분 소비된 `create_completion` 제너레이터 폐기가 KV 상태를 어중간하게 남겨
`warm_up()`의 prefix-cache 가정과 상호작용할 수 있다.

### 12. [레이턴시] 턴 시작 임계경로에 동기 wav 쓰기

`talk.py:443-445`

`handle_turn` 첫 문장이 최대 ~1.28MB 동기 디스크 쓰기다. CM4 SD카드에선 매 턴 수십~수백 ms.
`--streaming-asr`에선 그 파일이 어디에도 입력되지 않는다("기록용").

**수정 방향**: 단일 슬롯 백그라운드 라이터. 배치 ASR 경로는 어레이가 이미 메모리에 있으니
ASR 계약에 `transcribe_array()`를 추가하면 파일 왕복 자체가 사라진다.

### 13. [잠복 버그] `VoiceActivityDetector`가 docstring과 달리 per-utterance 상태를 공유한다

`turn/vad.py:180-181, 248, 366, 389`

docstring은 "per-utterance 상태는 VadStream에 산다"고 하지만 `VadStream.__init__`이
`config._vad`를 공유한다 — 한 detector에서 나온 두 stream은 sherpa 내부 버퍼·플래그·세그먼트
큐를 공유해 서로를 오염시킨다. 지금은 stream이 하나뿐이라 안 터지지만, docstring이 깨지는
사용법을 적극 권하고 있고 AEC 스피커 작업·캘리브레이션이 두 번째 stream을 원하게 된다.

**수정 방향**: sherpa VAD를 `VadStream.__init__`에서 생성, detector는 순수 config+factory로.

같은 파일의 죽은 코드: `listen_for_utterance`(`vad.py:274-346`, ~75줄 — "캘리브레이션
스크립트가 원한다"고 하지만 아무 스크립트도 안 씀), `VadStream.speaking`(호출자 없음,
게다가 세그먼트 내 무음에서 `_speaking`이 리셋되지 않아 읽으면 거짓말함).

### 14. [잠복 버그] `_AudioRing.append`가 초과 크기 프레임에서 자기 불변식을 깬다

`turn/vad.py:129-136`

`n >= capacity` 분기가 tail을 무조건 인덱스 0에 쓰는데, 링의 계약("절대 위치 p는
`p % capacity`에 산다")은 `(written+n) % capacity == 0`일 때만 성립한다. 어긋나면 `read()`가
크기는 맞고 위상이 틀린 — 즉 셔플된 — 오디오를 에러 없이 반환한다. 오늘은 도달 불가
(capacity ≈ 20.5s vs 30ms 프레임)지만, 이 클래스는 과거 조용한 오디오 손상 버그의 재발
방지용으로 존재한다 (`vad.py:463-469`).

**수정 방향**: 쓰기 정렬(`start = (written+n-capacity) % capacity` + 분할 복사) 또는 그냥
`raise ValueError` — 정당한 호출자가 없다.

### 15. [설계] 로드맵이 필요로 하는 누락 추상화

- **스테이지 Protocol 없음**: `registry._CLASSES`(`registry.py:38-47`)가 공통 베이스 없는 8개
  클래스를 매핑, 계약은 프로즈 표뿐. `AsrStage/LlmStage/TtsStage` Protocol을 정의하고
  `STSPipeline`을 그것으로 어노테이트하면 "프리셋 추가하는 법"이 타입체커가 강제하는 규칙이 된다.
- **표현력 TTS를 표현할 수 없음**: 계약이 `synthesize_audio(text)->(samples,sr)`뿐이라
  per-utterance prosody 슬롯이 없다. 세 백엔드가 각자 컨벤션을 만들기 전에
  `synthesize_audio(text, *, style=None, speed=None)`로 확장. (tts-expressivity 설계가 필요로 함)
- **`ReplyPlayer.is_active()` 의미가 구현체마다 다름**: `StreamPlayer`(`player.py:171`)는
  start()부터 True, `SessionPlayer`(`player.py:289`)는 버퍼 기준(게다가 드레인 한 블록 전에
  False). Protocol docstring(`player.py:100-106`)의 "barge-in 구분" 계약은 아무도 안 쓴다.
  의미 하나를 고르고 docstring 주장 삭제.

### 16. 작지만 구체적인 것들

- `storage.py:13-14, 57-59` — "memories: 아직 아무도 안 씀" docstring이 같은 파일의
  `save_memory`와 모순. 문서 부패.
- `storage.py:84` — pragma 없음. 매 턴 `log_turn`이 메인 스레드 풀 fsync. SD카드 타깃엔
  `journal_mode=WAL; synchronous=NORMAL`이 표준 트레이드.
- `textchunk.py:27 sanitize_for_tts` — 이모지는 거르지만 숫자·로마자는 안 거른다. 페르소나가
  요구하고 모델이 못 하는("`20대`", FEATURES 실측) 한글 숫자 풀어쓰기의 자연스러운 단일
  체크포인트가 여기다. espeak-ng가 한국어 보이스로 `20`을 이상하게 음소화한다.
- `tts.py:68-84` — FreyaTTS/MOSS 경로가 **문장 청크마다** temp파일 write+read+unlink.
  `_freyatts_server.py`가 이미 JSON 라인 프로토콜이니 length-prefixed float32로 디스크 왕복
  제거 가능. (기본 프리셋이 in-process라 우선순위 낮음)
- `tts.py:220-254` / `asr.py:217-255 _ensure_started` — 죽은 서버 재시작 시 이전
  reader/drainer 스레드·파이프 미정리(fd 누수), `close()`의 `kill()` 후 `wait()` 없음(좀비).
- `controller.py:240-264` — `begin_response`/`finish_response`가 `_current_player`와 state를
  따로 설정해 캡처 스레드의 `BARGE_IN_CONFIRMED`와 창이 있다. `barge_in_count`(튜닝 신호로
  문서화됨)가 오염될 수 있다. `_state_lock` 아래에서 함께 설정.
- `registry.py:79-98` — 시작마다 `models.yaml`을 6회 파싱(빌드 3 + `_cli.py` default 3).
  캐시하면 세 스테이지가 서로 다른 파일 내용을 볼 가능성도 제거.

---

## B. 테스트 상태

### 현재 인벤토리

| 스크립트 | 검증하는 것 | 필요 조건 | 자동화 |
|---|---|---|---|
| `_smoke_imports.py` | 24개 모듈 임포트, lazy-import 정책, registry 파싱, AEC backend 선택 | 의존성만. **가중치·하드웨어 불필요** | exit 1. 현재 유일하게 CI 가능한 산출물 |
| `_smoke_turn.py` | prefix 함수 7케이스 / 스트리밍 vs 배치 유사도 ≥0.6 / 합성 프레임 소스로 턴 1개 / transcriber 부착 시 텍스트 비어있지 않음 | **모델 가중치** (마이크는 불필요) | exit 1, ~11s |
| `_smoke_duplex.py` | 자기 재생에 barge-in 안 함 (backend별) | **실제 스피커+마이크** | **장치 없으면 SKIP인데 exit 0** — CI에 넣으면 초록불이 무의미 |

캘리브레이션 3종(`_calibrate_vad_threshold/aec_delay/turn_params`)과
`benchmark.py`/`_ab_persona.py`는 측정 도구지 테스트가 아니다 — 아무것도 assert하지 않는다.

`_smoke_turn.py`의 한계: 순수 로직 체크 1조차 `TEST_WAV.exists()` 조기 종료 뒤에 있어
가중치 없이 실행 불가. 유사도 하한 0.6은 느슨하고, duration 슬랙 ±1.5s는 #1의 pre-roll
팽창이나 `_AudioRing` 랩어라운드 버그를 못 잡는 폭이다. barge-in·취소 스코프·backchannel
스킵·빈 트랜스크립트·멀티턴 큐잉 등 에러/엣지 분기는 전무.

### FEATURES.md 표와의 어긋남

1. 표가 "파라미터 캘리브레이션 실측"과 "스모크 스크립트가 코드 경로를 밟음"을 같은
   실측 라벨로 묶는다. "Phase 3 배선 실측"의 실제 근거는 단일 happy path의 거친 assert 4개다.
2. 코드 정확성 커버리지에 대한 행이 없다 — `순수 로직 단위 테스트 | 없음` 행을 추가할 것.
3. `FEATURES.md:145` "run_pipeline.py는 자동화 테스트용" — 아무 자동화도 호출하지 않고,
   assert 없고, 골든 출력 없고, LLM 샘플링 때문에 비결정적이다. 수동 CLI다.
4. `FEATURES.md:160` "benchmark.py 전반 동작 검증됨" — 과거 수동 관측이지 지속 보증이 아니다.
   프리셋이 내일 깨져도 아무것도 모른다.

### 갭 — 가중치·하드웨어 없이 테스트 가능한데 커버리지 0인 순수 로직 (위험 순)

1. **`turn/controller.py` 상태기계** — 의존성 전부 주입식이라 fake만으로 완전 테스트 가능.
   미테스트 분기 각각이 뚜렷한 사용자 가시 실패에 대응: RESPONDING 외 barge-in no-op(:323),
   `allow_barge_in=False`, `begin_response`의 `_cancel` 클리어(:247 — 회귀하면 모든 응답 즉사),
   RESPONDING 중 SPEECH_STARTED 무시(:307), `_publish`의 finalize→reset 순서, 재생 중 턴
   큐잉/인덱스, frame_source 예외 시 루프 종료.
2. **`textchunk.py`** — 모든 응답의 임계경로. `_find_cut`의 min/max_chars·1차/2차 구두점
   상호작용, `sanitize_for_tts`의 `""` 스킵 시그널.
3. **`memory.py` 파싱/consolidation** — 적대적 0.6B 출력에서 살아남는 것이 존재 이유인 계약.
   `_parse_operations`의 인덱스 산술이 어긋나면 **엉뚱한 기억을 UPDATE = 비가역 데이터 손실**.
4. **`storage.py` 쿼리** — `ORDER BY confidence IS NULL, ...` NULL 반전과
   `ROW_NUMBER() PARTITION BY` dedup. 실패 모드가 "그럴듯하게 틀린 recall"이라 아무도 SQL로
   역추적 못 한다. temp SQLite로 테스트 가능.
5. **`_AudioRing`** — pre-roll 공급자. 같은 자리에서 이미 조용한 손상 버그로 비용을 치렀다.
6. **LocalAgreement 상태기계** — 현재 테스트 불가(`__post_init__`이 무조건 recognizer 생성).
   ~15줄 리팩터링으로 stabilizer를 모델-프리 클래스로 추출하면 "확정분은 줄어들지 않는다"
   불변식을 커버 가능.
7. `backchannel.py` (설계 문서 스스로 "순수 함수라 단위 테스트 쉬움"이라 적어놓고 0개.
   `_normalize`가 전각 구두점 `。！`을 안 거름), `resample.py`(endpoint=False 무드리프트
   불변식), `grace_frames_for_prob`(클램프 3케이스), `pipeline.run_streaming`(fake 3개로
   skipped/cancelled/asr_ms=0 계약).

### 사람/실기만이 검증할 수 있는 것 (문서 판단이 맞음 — 자동화 시도 금지)

실사용 대화 루프(발화 잘림·barge-in 즉시성·맞장구 무시), `barge_in_confirm_ms`·
`BACKCHANNEL_MAX_DURATION_S`·`rule2_*`(이 화자의 라벨 녹음 필요), 기기별 VAD threshold,
AEC 실효성(스피커 셋업 — 단 #5를 먼저 고칠 것), Phase 3 실사용 zipformer 빈 결과(화자/마이크
특성으로 좁혀짐), WASAPI 협상·워밍업 트랜지언트, TTS 자연스러움·페르소나 품질, CM4 레이턴시.

### 권장: 최소 스위트

**목표: ~7파일, 60~80 assert, 가중치·오디오 장치 불필요, 전체 2초 미만.** 이 제약이 핵심 —
가중치가 필요한 스위트는 돌리지 않게 된다.

선행 작업(~30분): `pyproject.toml`에 `[tool.pytest.ini_options]`, `tests/conftest.py`에
sys.path 처리, LocalAgreement stabilizer 추출 리팩터링 1건.

| 파일 | 커버 |
|---|---|
| `test_turn_controller.py` | fake VAD/player/transcriber로 위 분기 전부. **이 한 파일이 나머지 여섯 합보다 리팩터링 안전망 가치가 크다** |
| `test_textchunk.py` | 멀티청크 드레인, min_chars 억제, max_chars 강제 컷, flush, sanitize의 `""` 시그널 |
| `test_memory.py` | `_extract_json_array` 적대 입력 6종, `_dedupe_memories`, `_parse_operations` 실패→ADD, **UPDATE가 올바른 target_id에 도달** |
| `test_storage.py` | temp db로 recent_memories 정렬/dedup/limit, consolidation용 id, update의 created_at 갱신 |
| `test_vad_ring.py` | 랩 seam, 초과 프레임, 오래된 read 클램프, copy-not-view |
| `test_resample.py` | 길이 공식, identity, 빈 입력, 다운믹스, 청크 연결 무드리프트 |
| `test_small_pure.py` | backchannel 경계(0.6s·`<=1`·구두점), longest_common_prefix(스모크에서 이관), stabilizer 단조성, grace 클램프 |

후속: ① `_smoke_turn.py` 체크 1을 pytest로 이관하고 `TEST_WAV` 조기 종료를 체크별 스킵으로,
② CI에 `pytest` + `_smoke_imports.py`만 (**`_smoke_duplex.py`는 넣지 말 것** — SKIP이 exit 0이라
초록불을 무시하는 습관을 가르친다), ③ FEATURES.md 표에 `순수 로직 단위 테스트` 행 추가.

---

## C. 권장 작업 순서

1. **#1 backchannel 버그 수정 + 회귀 테스트** — 반나절. 죽은 기능 부활, 가장 명확한 승리.
2. **최소 pytest 스위트(위 표)** — 이후 모든 리팩터링의 안전망.
3. **#4 기억 recall 검증/프레이밍** — 반나절. 페르소나 오버라이드 통로 차단.
4. **#3 캡처 스레드 실패 전파** — 몇 시간.
5. **#5 refgate 정렬 수정 → 그다음 계획된 사람 검증 + AEC 스피커 실험** — 실험 전에 고쳐야
   실험 세션이 유효하다.
6. **#2 TTS 병렬화 + #9 스레드 예산** — 한 묶음으로, CM4 실측과 함께.

나머지(#6~#16)는 해당 영역을 건드릴 때 같이.
