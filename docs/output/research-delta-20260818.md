# 리서치 델타 (2026-08-18)

> `docs/voice-agent-oss-survey.md`(2026-08-14) **이후**의 변화만 다룬다. survey에 이미 있는
> 항목은 재론하지 않는다. 목적은 survey가 "막혔다"고 닫아둔 결론 세 개
> (한국어 CPU TTS 기성품 없음 / 스트리밍 ASR은 모델 문제 / 3-state 미착수)를 1차 출처로 재검증하는 것.

**표기 규칙** — 이 프로젝트는 `docs/llm-conversational-selection.md`에서 "양자화 레포나 기억에서
라이선스를 가져오지 말 것"을 명시했다. 그 규칙을 출처 등급으로 고정한다:

- **[1차확인]** HF 모델카드 / 레포 LICENSE 파일 / 공식 docs / PyPI 메타데이터 / 논문에서 직접 확인
- **[벤더주장]** 벤더 블로그·발표자료. 검증 안 된 마케팅 수치
- **[추론]** 위 둘에서 우리가 끌어낸 결론. 측정 아님

**§1–6은 문헌 조사고, §7은 실측이다.** 섞지 말 것 — §7의 숫자만 이 박스에서 나온 것이다.

---

## 0. 결론 먼저 — survey의 세 결론은 이렇게 바뀐다

| survey 결론 | 델타 | 근거 등급 |
|---|---|---|
| §0.5 "CPU + 상업 라이선스 한국어 TTS 기성품 없음. **Supertonic은 GPU**" | **틀렸다.** Supertonic은 온디바이스 ONNX가 정체성이고 한국어를 지원하며 sherpa-onnx에 통합돼 있다. 단 모델 가중치가 OpenRAIL-M | [1차확인] |
| §0.5의 대안으로 우리가 세웠던 **Kokoro 한국어** | **불가능하다.** Kokoro에는 한국어 음성이 아예 없다 (`lang_code` ∈ a,b,e,f,h,i,j,p,z) | [1차확인] |
| §7 #4 "스트리밍 ASR — #2886 픽스 기다림" | **막힌 길 확정.** 2025-12-10 오픈, 미해결, 원인 추정은 인코더 export 결함 | [1차확인] |
| §0.1 "Smart Turn v3는 우리에게 정답이었다" | **여전히 맞다. 그리고 대안이 사실상 없다** — 완전 오픈 + 한국어 + CPU 조합은 이것뿐 | [1차확인] |
| (신규) Deepgram Flux를 벤치마크 삼자 | **한국어 미지원.** 레퍼런스로만 유효 | [1차확인] |

**가장 값진 신규 발견**은 turn-taking이 아니라 **투기적 프리필**이다. llama.cpp가 이미
`cache_prompt` + `--cache-reuse`로 그 메커니즘을 제공하고, 이 방식은 LiveKit·Pipecat이
아직 싸우고 있는 이중 발화 버그 클래스를 **구조적으로 제거**한다. §3 참조.

---

## 1. 한국어 CPU TTS

### 1.1 Supertonic — survey 결론을 뒤집는다

- **한국어 지원 [1차확인]** — Supertonic 2 = en/ko/es/pt/fr. Supertonic 3 = 31개어에 한국어 포함.
  sherpa-onnx `scripts/supertonic/gen_calib_configs.py`가 "all 31 Supertonic 3 language codes"를
  다루며 `ko` 포함. ([sherpa docs](https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html),
  [scripts/supertonic](https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/supertonic))
- **sherpa-onnx 통합 [1차확인]** — Supertonic 3 지원은 [PR #3605](https://github.com/k2-fsa/sherpa-onnx/pull/3605),
  **2026-05-13 머지**. `OfflineTtsSupertonicModelConfig` 필드 = duration_predictor / text_encoder /
  vector_estimator / vocoder + `tts.json` + `unicode_indexer.bin`. 합성 시 `--sid` + `--lang`.
  모델 배포본 두 개: `sherpa-onnx-supertonic-tts-int8-2026-03-06`(v2),
  `sherpa-onnx-supertonic-3-tts-int8-2026-05-11`(v3).
- **G2P가 필요 없다 [1차확인]** — 이게 예상 못 한 큰 이득이다. 논문:
  SupertonicTTS는 *"operates directly on raw character-level text and employs cross-attention for
  text-speech alignment, thus eliminating the need for grapheme-to-phoneme (G2P) modules and
  external aligners."* ([arXiv 2503.23108](https://arxiv.org/abs/2503.23108))
  sherpa-onnx 쪽 도구가 이를 뒷받침한다 — `generate_indexer_bin.py`(unicode indexer) +
  `generate_nfkd_table.py`(NFKD 정규화)만 있고 **tokens.txt·음소 목록·espeak·G2P가 아예 없다**.
  → §1.3의 한국어 G2P 지옥을 **통째로 회피**한다.
- **라이선스** — 코드 **MIT**, 가중치 **OpenRAIL-M**. 둘 다 **[1차확인]**으로 승격됐다(§7.2).
  ⚠️ 그리고 함정이 있다: **sherpa-onnx 번들 안의 `LICENSE` 파일은 MIT인데 그건 코드 라이선스다.**
  번들만 읽으면 MIT라고 결론 낸다. 상세와 실제 제한 조항은 §7.2.

> ✅ **§1.1의 미확인 두 건은 §7에서 해소됐다.**
> (a) 버전: **범프가 필요 없다** — 이미 설치된 1.13.4에 `OfflineTtsSupertonicModelConfig`가 있다(§7.1).
> (b) 샘플레이트: **44100이 맞다** — `tts.json`의 `ae.sample_rate`로 확인. sherpa docs 표의 24000이 틀렸다(§7.3).

### 1.2 Kokoro 한국어 — 트랙 D의 전제가 무너졌다

**Kokoro에는 한국어가 없다. [1차확인]**

- `kokoro` PyPI가 문서화한 `lang_code` 값은 **`a,b,e,f,h,i,j,p,z`** — `k`/`ko`가 없다.
  ([kokoro PyPI](https://pypi.org/project/kokoro/))
- `misaki`에 `ko.py`와 `g2pkc/`가 존재하지만 **대응하는 Kokoro 음성이 없다** —
  코드만 있고 목소리가 없다. `misaki/ko.py`는 `from .g2pkc import G2p` 한 줄 래퍼에
  `# TODO: Return List[MToken] instead of None`이 달린 스텁이다.
- 덤으로, **`pip install misaki[ko]`는 깨져 있다 [추론, 확신 높음]** — extras가
  `jamo`,`nltk`만 걸어두는데 `g2pkc.G2p.__init__`이 `import mecab`을 무조건 실행한다.
- sherpa-onnx의 Kokoro도 **영어+중국어만** — Kokoro 페이지가 직접 말한다:
  *"It is a multi-lingual model, but we only add English and Chinese support for it."* [1차확인]

**정정.** 이 계획을 세울 때 나는 "Kokoro가 한국어를 지원한다"를 근거로 트랙 D를 넣었다.
그 출처는 집계 블로그였고, 1차 출처가 반박한다 — 레포가 경고해둔 바로 그 실패 방식이다.
**트랙 D는 전제가 없어졌으므로 폐기를 권고한다.** (§5 참조)

### 1.3 한국어 텍스트 프론트엔드 — 여기가 진짜 지뢰밭

Supertonic이 G2P를 없애줘도 **숫자·영문 확장(TN)은 여전히 우리 몫**이다.
G2P-free ≠ TN-free. sherpa-onnx의 C++ 경로에는 한국어 숫자 확장이 전혀 없어
`"123"`을 넣으면 동작이 정의되지 않는다.

조사 결과 **쓸 수 있는 기성품이 하나도 없다.** 전부 최소 한 축에서 탈락한다:

| 후보 | 라이선스 [1차확인] | 탈락 사유 |
|---|---|---|
| KoG2P | 🚨 **GPL-3.0** | copyleft. 게다가 숫자·영문 확장 기능 자체가 없음 |
| KoNLPy | 🚨 **GPL-3.0+** | copyleft + `JPype1` → **CM4에 JVM** |
| g2pK | Apache-2.0 | 의존성에 konlpy(GPL) + mecab. 2020-08 이후 방치 |
| g2pkc / g2pkk / g2pk2 | Apache-2.0 | konlpy는 뗐지만 **mecab을 `__init__`에서 무조건 import**. 의존성 그래프에서 숨겨져 있을 뿐 |
| num2words | 🚨 **LGPL-2.1** | 기능은 최적합 (`lang_KO.py`에 한자어 기수 + 고유어 서수 `한/두/세`, `스물`). 순수 Python 무의존. LGPL만 문제 |
| kiwipiepy | 🚨 **LGPL-2.1+** | aarch64 wheel 있고 관리 우수하나 G2P·TN 기능 없음 (형태소 분석기) |
| NeMo-text-processing | Apache-2.0 | **한국어 TN·ITN 문법이 양방향으로 존재** — 유일하게 완성도 있는 permissive 자산. 그런데 `pynini`가 **aarch64 wheel 없음** → CM4에서 OpenFst 소스 빌드 = 실질적 차단 |
| jamo | Apache-2.0, 무의존 ✅ | 음절↔자모 분해만. G2P 아님 |

또한 g2pK 계열 전체가 `from nltk.corpus import cmudict`를 하고 이 코퍼스는 **런타임 다운로드**다 —
완전 오프라인 CM4에서는 `nltk_data`를 미리 구워 넣어야 한다.

**권고 [추론]**: 자체 한국어 숫자·영문 확장기를 직접 쓴다(~300줄, 순수 Python, 무의존).
한자어 기수(만/억/조 그룹), 고유어 수사(`한/두/세/네`, `스물`) 단위명사 분기, 소수(`점`),
전화·ID 자릿수 읽기(`0`→`공`), 날짜 불규칙(`육월`→`유월`, `십월`→`시월`),
`원`/`퍼센트`/시·분(시는 고유어·분은 한자어), 라틴 문자표 + 두문자어 + 외래어 사전.
`g2pk2`/`g2pkc`가 Apache-2.0이므로 그들의 `numerals.py`·`english.py` **로직을 읽고 이식**하는 것은
적법하다(출처 표기). NeMo의 한국어 문법은 **x86 박스에서 테스트 오라클로만** 쓰고 CM4에 안 올린다.

현재 방식(LLM 프롬프트로 한글 표기 지시, `persona.py:28-31`)은 **폐기하지 말고 1계층으로 유지**하되,
TTS 입력 경로에 결정론적 가드를 추가한다 — 살아남은 `[0-9A-Za-z%₩$]`를 정규식으로 잡아 확장.
프롬프트 준수는 양자화 모델에서 긴 출력일 때 무너지고, 무너지면 TTS가 숫자를 그대로 읽거나 삼킨다.
그리고 프롬프트 준수는 **테스트할 수 없다**. 가드는 된다.

---

## 2. 턴테이킹 — Smart Turn v3에 사실상 대안이 없다

우리 선택을 검증하려고 벤더·오픈 전수를 훑은 결과, **온디바이스 + 한국어 + EOT** 세 조건을
동시에 만족하는 후보가 셋뿐이고 그중 완전 오픈은 하나다.

| 후보 | 한국어 | CPU/온디바이스 | 가중치·라이선스 | 판정 |
|---|---|---|---|---|
| **pipecat smart-turn-v3** | ✅ 23개어에 한국어 명시 [1차확인] | ✅ "as little as 10ms on some CPUs", `requirements_aarch64.txt` 존재 [1차확인] | ✅ **BSD-2-Clause**, int8 8MB / ~8M param [1차확인] | **우리 선택이 맞다. 대안 없음** |
| livekit/turn-detector | ✅ 14개어에 한국어 [1차확인] | ✅ "CPU-only … CPUExecutionProvider", <500MB RAM [1차확인] | ⚠️ **"LiveKit Model License"(프로프라이어터리)** [1차확인] | 폴백 후보. 단 **텍스트 입력** → 한국어 스트리밍 ASR이 선행 필요 = 연산 두 배 |
| Krisp krisp-viva-tp-v3 | ⚠️ **모순.** VIVA 2.0 블로그는 한국어 포함[벤더주장], SDK docs는 "English only"[1차확인] | ⚠️ C SDK는 armv8a 지원[1차확인], **Python SDK는 aarch64 wheel 없음**[1차확인] | ❌ 프로프라이어터리 `.kef` + 상업 라이선스 | **차단.** `KRISP_VIVA_API_KEY` 필수 → 오프라인 불가 |

부정 결과도 값지다 — 다음은 **EOT 제품이 아예 없다** [1차확인]:
**Silero**(VAD만, v6.2까지 turn-taking 모델 없음), **sherpa-onnx**(학습 EOT 없음 — 규칙 3개 OR:
rule1 2.4s / rule2 1.2s / rule3 20s. 우리가 이미 쓰는 그것), **Picovoice**(EOT 엔진 없음.
게다가 **Cheetah 스트리밍 STT는 한국어 미지원**, Leopard 배치만 한국어), **Rime**(TTS만),
**Ultravox**(speech-LLM, VAD/EOT 컴포넌트 없음).

언어가 안 맞아 탈락: **ESPnet** `Turn_taking_prediction_SWBD`(40ms마다 확률 출력, ROC-AUC 92.0%,
CC-BY-4.0 — 그런데 **영어 전용**, ONNX export 없음), **NVIDIA** `parakeet_realtime_eou_120m-v1`
(p50 160ms/p90 280ms — **영어 전용 + GPU 필수**), **Kyutai** `stt-1b-en_fr`(semantic VAD 있으나
en/fr + GPU, 그리고 **1B STT에서 작은 턴 헤드만 떼낼 수 없다**).

### 2.1 Deepgram Flux — 한국어가 없다

- **`flux-general-multi` 10개어 = en, es, fr, de, hi, ru, pt, ja, it, nl. 한국어 없음. [1차확인]**
  ([language-prompting](https://developers.deepgram.com/docs/flux/language-prompting))
- 파라미터 [1차확인]: `eot_threshold` 0.5–0.9 기본 **0.7**, `eager_eot_threshold` 0.3–0.9
  **기본 미설정(옵트인)**, `eot_timeout_ms` 500–60000 기본 5000.
  하드 제약: `eager_eot_threshold` ≤ `eot_threshold`, 아니면 에러.
- Smart Turn·LiveKit EOU보다 EOT F1이 높다는 주장은 **[벤더주장]** — 벤치마크·데이터셋 미공개.
  독립 재현 근거를 찾지 못했다. 우리 엔드포인터를 갈아치울 근거로는 부족하다.
- 자체 엔드포인팅 지연은 **"p90 (p95) latency of 1 second (1.5 seconds)"** [벤더주장].
  Smart Turn v3의 ~12ms 추론과는 전혀 다른 동작점이다 — **eager 트릭이 존재하는 이유의
  일부가 이 1초 p90을 덮는 것**으로 보인다 [추론].

→ **우리에게**: Flux는 채택 후보가 아니고(한국어 없음, 프로프라이어터리), 계획대로
**아이디어 출처**로만 유효하다. 다만 `eot_threshold` 기본 **0.7**은
우리 `complete_threshold: 0.5`(Smart Turn 스톡값, `detector.py:50`이 "not tuned here"라고 인정)에
대한 유효한 데이터포인트로 남는다 → 스윕 근거.

---

## 3. 투기적 실행 — 이번 조사의 최대 수확

계획의 트랙 C-2를 **설계 변경**해야 한다. 벤더 패턴을 그대로 이식하면 안 된다.

### 3.1 비용 구조가 우리와 정반대다

- Deepgram 1차 문서, 그대로: *"Good for trimming that last 100-200ms of end-to-end latency
  **at the cost of 50-70% more LLM calls**."* [1차확인]
  ([eager-eot](https://developers.deepgram.com/docs/flux/voice-agent-eager-eot))
  임계 0.3–0.5에서 "EagerEndOfTurn 150–250ms earlier" [벤더주장].
- **역산하면 eager 추측의 적중률은 대략 30~50%뿐이다 [추론]** — 턴당 호출이 1.5~1.7배로 늘어난다는
  뜻이므로. 어느 벤더도 적중률을 직접 밝히지 않는다.
- 결정적 차이: **벤더의 비용은 남의 GPU에 대한 청구서고, 우리 비용은 우리의 유일한 CPU다.**
  낭비된 프리필은 CM4에서 돈이 아니라 **가속하려던 그 턴에 지연을 더한다** — 진짜 프리필이
  그 뒤에 줄을 서므로. 벤더의 트레이드오프가 우리에게선 부호가 뒤집힌다 [추론].
- Deepgram 자신의 완화책은 "eager엔 더 작은 모델을 쓰라"인데 [1차확인],
  4GB에 두 번째 가중치를 올릴 수 없으니 **우리에겐 불가**. 우리 대응은 더 작은 모델이 아니라
  **같은 모델로 더 적은 일**이다 — 즉 프리필만.

### 3.2 llama.cpp가 이미 프리필 재사용을 제공한다

**어느 벤더도, 어느 논문도 "토큰을 만들지 않고 KV 캐시만 미리 채우는" 방식을 이름 붙여
설명하지 않는다.** Deepgram은 전체 생성을 권하고, LiveKit은 LLM 생성 + TTS 옵션,
Pipecat은 기능 자체가 없다 [전부 1차확인]. 그런데 llama.cpp에는 이미 있다 [1차확인]:

- `cache_prompt` — *"Re-use KV cache from a previous request if possible. This way the common
  prefix does not have to be re-processed, only the suffix that differs… Default: `true`"*
- `--cache-reuse` — *"min chunk size to attempt reusing from the cache via KV shifting"* (기본 0)
- `-sps` — 슬롯 재사용을 위한 프롬프트 일치율 (기본 0.10)
- `-cram` — 캐시 상한 MiB (**기본 8192 — 4GB CM4에선 말이 안 되는 값이라 반드시 낮춰야 한다**)

**이 방식이 벤더 방식보다 우리에게 구조적으로 우월한 이유:**
1. **취소 안전** — 따뜻해진 캐시는 *틀릴* 수가 없고 쓸모없을 수만 있다. → 아래 3.3의
   이중 발화 버그 클래스가 **아예 발생하지 않는다**
2. **추가 메모리 0** — 같은 슬롯, 같은 KV
3. 두 번째 모델 불필요

또 하나 유리한 점: llama.cpp의 프리픽스 재사용은 **토큰 단위 최장공통접두사**이고
`--cache-reuse`가 분기점 *이후* 청크까지 KV 시프팅으로 재사용한다. vLLM은 **블록 단위**라
한 토큰이 바뀌면 그 블록과 **이후 전부**가 무효화된다 [1차확인].
→ vLLM의 블록 의미론으로 우리 무효화 비용을 추론하면 안 된다.

> ⚠️ 미확인: `n_predict: 0`으로 프리필만 하는 모드가 문서화돼 있지 않다.
> `n_predict: 1`이 안전한 근사이고(우리 `llm.py:425-452` `warm_up()`이 이미 정확히
> 이 트릭을 쓴다 — 토큰 하나만 생성), 슬롯 캐시가 실제로 채워지는지 한 줄 실측이 필요하다.

### 3.3 LiveKit·Pipecat이 실제로 겪은 버그 = 우리 설계 제약

이게 문서보다 값지다. 전부 [1차확인] (공개 이슈):

- **취소는 경쟁 조건이다.** [livekit#4219] `preemptive_generation=True`가 턴당 LLM 요청을
  **두 번** 보내고 둘 다 `cancelled: False`로 보고. 원인: 투기 작업이 **이미 끝난 뒤에** 취소가 도착.
  → **취소에 기대는 설계를 하지 말 것.** 짧은 투기 작업은 취소보다 먼저 끝난다.
  대신 **결과 게이팅**: 투기 결과를 곁버퍼에 넣고, 확정 전사와 일치할 때만 *채택*한다.
  게이팅은 경쟁이 없고 취소는 있다.
- **유령 응답 / 이중 발화.** [livekit-js#1365] 투기 생성이 도구 실행 중 stale 컨텍스트로 시작해
  LLM이 **도구 결과를 환각**하고 사용자가 응답을 두 번 듣는다.
  → 부수효과 작업이 진행 중이면 절대 투기하지 말 것. 변경 중인 컨텍스트를 투기가 보게 하지 말 것.
- **멱등성은 턴 단위여야 한다.** [pipecat#4912] 한 발화 → 여러 생성 → 봇이 여러 번 말함.
  멱등 플래그가 *생성* 단위로 리셋돼서 턴 단위가 아니다. 재현: 문장 중간에 두 번 쉬기.
  → **한국어의 조사·어미 앞 망설임이 정확히 이 트리거다** [추론]. 우리에게 특히 위험.
- **투기는 순수 곁채널이어야 한다.** [livekit#3414, not planned으로 닫힘] 제안된 아키텍처가
  우리가 택할 것과 같다: *"call the large model in advance, cache the result, and only go through
  those node processes when it is confirmed that the preemptive generation data will be used."*
  → 이벤트 발행 없음, 컨텍스트 변경 없음, TTS 없음, 턴으로 로깅 없음. **캐시 쓰기만.**

### 3.4 한국어에서 적중률을 0으로 만드는 함정

**[livekit v1.6.8, PR #6667] "fix(voice): tolerate formatting changes in preemptive transcripts"** [1차확인]
— 대소문자·공백·구두점을 정규화한 뒤 비교하도록 고쳤다. 원인 버그(#5766)는 STT가
확정 시점에 재정규화해서 **모든 투기가 폐기**된 것.

→ **한국어 ASR의 확정 패스는 통상 띄어쓰기를 다시 하고 구두점을 붙인다.**
정규화 없이 비교하면 우리 적중률은 30~50%가 아니라 **거의 0**이다 [추론].
다행히 이 비교에 쓸 정규화는 이미 만들었다 — `metrics.normalize_for_cer`.

구조적 이점 [추론]: 수정이 **꼬리**에 오면 접두사가 살아 프리필이 유효하고,
**앞부분**에 오면 프리필이 파괴된다. 한국어는 어미·조사 쪽 수정이 많아 **우리에게 유리한 방향**이다 —
단 프롬프트 템플릿이 사용자 텍스트를 마지막에 두고 그 뒤에 래퍼 토큰이 없어야 한다.
(우리는 `llm.py:376-401` `_render`로 chat template을 직접 렌더링하므로 확인 가능.)

### 3.5 LiveKit에서 그대로 훔칠 안전장치 [1차확인]

`TurnHandlingOptions` / `PreemptiveGenerationOptions`:
- `preemptive_tts` **기본 False** — *"Deferring TTS reduces unnecessary compute when a user
  interrupts or when the transcript changes"*
- `max_retries` **기본 3**, *"per user turn… counter resets when the turn completes"*
  → **최악의 낭비 CPU를 턴 단위로 바운딩**하는 가장 값싼 안전장치
- `max_speech_duration` 기본 10.0s — *"long utterances are more likely to change"*
- 트리거는 `PREFLIGHT_TRANSCRIPT` 이벤트이고 **재사용 판정은 전사 문자열 동등성 검사**다
  (타이밍 검사가 아니다). 경고: Speechmatics는 이 이벤트를 안 내보내서 **기능이 조용히 죽는다**
  → 우리도 sherpa-onnx에서 "eager 전사" 이벤트를 명시적으로 만들어야 하고,
  단순 interim 가설로는 안 된다.

### 3.6 착수 전 게이트 — 이 측정 없이는 만들지 말 것

투기적 *디코딩*(초안 모델)은 CPU에서 반복적으로 **손실**로 측정됐다:
llama.cpp Metal MTP "net loss (−11% to −24%)" [벤더주장/이슈],
0.5B draft + 7B target on RTX 5060 Ti "0.27×, a nearly 4× regression — **despite perfectly healthy
acceptance rates**" [2차]. llama.cpp #21453 "Speculative Decoding for Low-Latency CPU Inference"는
**벤치마크 0의 빈 연구 아젠다**다 [1차확인].
→ 이건 다른 트레이드(유휴 대역폭 회수)이므로 우리 근거로 인용하면 안 되고, 초안 모델 방식은 채택하지 않는다.

크기 감각 [1차확인 초록] — [arXiv 2511.07425](https://arxiv.org/abs/2511.07425),
Pi 4(**CM4와 같은 Cortex-A72**)·Pi 5·Orange Pi 5 Pro에서 25개 양자화 모델:
*"SBCs can reliably support models up to 1.5B parameters"*. 우리는 2.3B다.
Pi 5(A76, A72보다 빠름) ~20 tok/s prompt-eval, ~7 tok/s generation [2차]
→ 40토큰 한국어 발화 프리필이 Pi 5에서 ~2초, CM4에선 더 나쁘다 [추론].

**게이트: CM4(또는 `NOBODY_CPU_BUDGET=4` 프록시)에서 `llama-bench`로 프리필 tok/s를 먼저 잰다.
전형적 발화의 프리필이 우리가 아끼려는 300–800ms grace 창을 넘으면 이 기능은 자기모순이고,
답은 "만들지 않는다"다.**

---

## 4. 스트리밍 ASR

- **sherpa-onnx #2886은 여전히 오픈** [1차확인] — 2025-12-10 개시, 메인테이너 응답·PR 없음.
  증상은 우리와 동일("recognizer.getResult() always returns empty string").
  이슈 본문의 원인 추정은 인코더의 **PAIT-ONNX-200 경고 → malformed ONNX structure**,
  즉 export/학습 단계 문제이고 런타임 코드 결함이 아니다. 다른 언어(중/불/독/포/러)는 정상.
  → `asr.py:69-72`가 기다리던 픽스는 오지 않는다. **트랙 B의 chunked SenseVoice 선택이 옳다.**
- 참고로 sherpa-onnx의 엔드포인팅은 학습 모델이 아니라 규칙 3개 OR이다 [1차확인]:
  `rule1` 2.4s 무발화 후 무음 / `rule2` 비공백 디코드 후 1.2s 무음 / `rule3` 20s 상한.
  우리 `configs/streaming_asr.yaml`이 rule2를 0.6s로 낮춰둔 것은 이 기본값 대비 의도적 변경이다.

> ⚠️ **이 축은 미완이다.** 담당 조사 에이전트가 세션 한도로 중단됐다. 남은 질문:
> 한국어 스트리밍 대안(Vosk-ko / onnx-asr / FunASR streaming)의 CPU RTF·라이선스,
> chunked 배치 디코드의 선례와 실패 방식, KsponSpeech 전처리가 실제 캡처 실패를 설명하는지.

---

## 5. 계획에 반영할 것

| # | 변경 | 근거 |
|---|---|---|
| 1 | **트랙 D(Kokoro 한국어) 폐기** | Kokoro에 한국어 음성이 없다 [1차확인]. 전제 소멸 |
| 2 | 트랙 A-1의 첫 작업 = **Supertonic 3 포함 sherpa-onnx 정확한 버전 확정** + 샘플레이트 24k/44.1k 불일치 해소 | 미확인 사항 |
| 3 | 트랙 A에서 **G2P 걱정 삭제**, 대신 **한국어 숫자·영문 확장기(자작 ~300줄)를 신규 항목으로 추가** | Supertonic은 character-level [1차확인]이나 TN은 여전히 우리 몫 |
| 4 | 트랙 C-2를 **투기적 생성 → 투기적 프리필(llama.cpp `cache_prompt`)**로 확정. 취소가 아니라 **결과 게이팅**. 비교 전 **정규화 필수**(`metrics.normalize_for_cer` 재사용) | §3.2–3.4 |
| 5 | C-2에 LiveKit 안전장치 이식: `preemptive_tts` 기본 off, **턴당 재시도 상한**, 순수 곁채널, **턴 단위 멱등성** | §3.3, §3.5 |
| 6 | C-2 착수 전 **프리필 tok/s 게이트** 추가 | §3.6 |
| 7 | Deepgram은 벤치마크가 아니라 **아이디어 출처**로 격하. 단 `eot_threshold=0.7`은 스윕 근거로 유지 | 한국어 미지원 + F1 주장은 벤더주장 |
| 8 | `-cram` 기본 8192 MiB는 4GB 타깃에서 **명시적으로 낮춰야 함**을 기록 | [1차확인] |

## 6. 열린 질문의 현재 상태 (2026-08-19 갱신 — 5축 전부 종결)

원래 이 절은 "세션 한도와 WebSearch 예산 소진으로 중단된 축"의 목록이었다. 그 사유는
지났고, 목록은 뒤의 절들이 하나씩 닫았는데도 열린 것처럼 남아 있었다. 실제 상태:

| # | 축 | 상태 |
|---|---|---|
| 1 | 스트리밍 ASR 대안 (Vosk-ko, onnx-asr, FunASR streaming 한국어) | **닫힘 → §9** |
| 2 | 한국어 CPU TTS 잔여 후보 (Piper 한국어, StyleTTS2/VITS/MeloTTS, ZipVoice) | **닫힘 → §11** (MeloTTS-Korean 발견, Piper 한국어는 NC) |
| 3 | 배포 아날로그 + per-core 병목 (OpenLive, ARM 양자화, SBC 재검토) | **닫힘 → §10·§11** |
| 4 | 학술 turn-taking 2025–2026 (Next-Turn, Phoenix-VAD, Easy Turn, MuVAP, VAP) | **닫힘 → §14** — §2 결론 유지, 그리고 VAP 계열 전체 종결 |
| 5 | 부분 가설 불안정성 수치 (Shangguan et al. 2006.01416의 UPSR 표) | **측정으로 대체** — 아래 |

**축 5는 "미완"이 아니라 "쓸 데가 없어짐"이다.** 이 수치를 원한 목적이 투기적 프리필의
적중률 추정이었고, **트랙 C-2는 §8에서 우리 자체 측정으로 기각됐다**(프리필이 턴당 3~16토큰,
절약 ~43ms 대비 비용 ~200ms). 게다가 이 절 자신이 "우리 sherpa 스트림에서 직접 재는 게
빠르다"고 적었고 실제로 그렇게 쟀다(§7: 16개 중 8개 미커밋, 커밋된 내용이 재디코드로 뒤집힘).
논문의 UPSR 표는 이제 어떤 결정의 입력도 아니다. **PDF 추출 실패를 다시 시도하지 않는다.**

> PDF 추출 자체는 이후 해결됐다 — `uv run --no-project --with pypdf`로 임시 의존성을 써서
> CM5 데이터시트 Appendix B를 그대로 뽑았다(§10.5). 이 방법이 있다는 것만 남겨둔다.

---

## 7. 실측 (2026-08-18, Windows CPU, `NOBODY_CPU_BUDGET=4`)

여기서부터는 **문헌이 아니라 우리가 이 박스에서 잰 숫자**다.
도구: `scripts/_ab_tts.py`(신규), 판정 ASR은 sense-voice-small.

### 7.1 sherpa-onnx 범프가 필요 없다 — 트랙 A-1 소멸

`OfflineTtsSupertonicModelConfig`가 **이미 설치된 sherpa-onnx 1.13.4에 있다.**
릴리스 노트가 아니라 introspection으로 확인했고, `unicode_indexer` + `voice_style`
필드 존재가 Supertonic **3** 서명이다. 함께 발견: `OfflineTtsZipvoiceModelConfig`,
`OfflineTtsKittenModelConfig`, `OfflineTtsPocketModelConfig`도 이미 있다.

→ **계획의 트랙 A-1(범프 + 회귀 게이트)은 불필요하다.** 계획에서 가장 위험한 항목
(ASR·VAD·TTS를 한 런타임에 얹은 채로 버전 올리기)이 사라졌다.

### 7.2 라이선스 — [벤더주장] → [1차확인], 그리고 함정 하나

모델 번들 안에 들어 있는 `LICENSE` 파일은 **MIT**다. **이건 가중치 라이선스가 아니다.**
같은 디렉터리의 upstream README가 직접 말한다:
*"The accompanying model is released under the OpenRAIL-M License."*

즉 **sherpa-onnx 재배포본이 OpenRAIL-M 가중치 옆에 MIT LICENSE를 함께 실어놨다.**
번들의 LICENSE만 읽으면 MIT라고 결론 내린다 —
`llm-conversational-selection.md`가 경고한 그 함정의 실물이 우리 `models/`에 들어왔다.

BigScience OpenRAIL-M(2022-08-18) 실제 조항 [1차확인]:
- 상업 이용 **허용**, royalty-free
- 사용 제한을 **하위 계약에 강제 조항으로 전달해야 함** (§4.a)
- 음성 컴패니언에 직접 걸리는 제한 3개:
  **기계 생성 콘텐츠 미고지 금지**(고지 필요) / **동의 없는 사칭·딥페이크 금지** /
  **의료 조언 제공 금지**

`configs/models.yaml`의 `supertonic-3-ko` 주석에 이걸 다 기록했다.

### 7.3 모델 실체

`tts.json` 직접 확인:
- `ae.sample_rate = **44100**` → sherpa docs 표의 24000은 틀렸다. 불일치 해소.
- `text_encoder.n_langs = 0`, `lang_emb_dim = 0` (vector_field도 동일)
  → **언어 조건부가 아예 없다.** 그래서 `generate()`에 `lang` 인자가 없고
  docs의 `--lang`에 대응하는 API도 없다. 언어는 문자에 내재한다.
- `num_speakers = 10`, `style_token_layer.n_style = 50`
- 크기 139MB(int8), 로드 1.3s

### 7.4 명료도·속도 — 비겼고, 속도는 졌다

ASR 라운드트립 CER, 평문 8문장(94 참조자):

| preset | CER | err | RTF 중위 | RTF 최악 | rate | load |
|---|---|---|---|---|---|---|
| sherpa-matcha-ko (현행 기본) | **0.074** | 7 | **0.37** | **0.53** | 22050 | 9.76s |
| supertonic-3-ko (sid=7) | **0.074** | 7 | 0.73 | 0.86 | 44100 | **1.32s** |

- **명료도는 완전 동률**(94자 중 7오류, 둘 다). 이 표본에서 우열을 말할 수 없다.
- **속도는 Matcha가 2배 빠르다.** ⚠️ **CM4 함의**: Supertonic의 최악 RTF 0.86은
  이 28코어 박스를 4코어로 제한한 값이다. CM4의 per-core는 이보다 훨씬 느리므로
  **RTF > 1.0(실시간보다 느림)으로 넘어갈 가능성이 크다.** 스트리밍 재생에 치명적.
  → CM4 실기 측정 없이 Supertonic 채택을 결정할 수 없다.
- 로드는 Supertonic이 7배 빠르다(1.3s vs 9.8s). 단 `warm_up()`으로 숨기는 일회성 비용.
- 화자 스윕(10명): CER 0.025~0.089. **sid=0은 중간, sid=7이 최고** —
  기본값으로 0을 쓰면 조용히 중간 품질을 배포한다. 프리셋에 `speaker_id: 7`로 고정.
  단 **화자 간 CER 차이는 대부분 노이즈다**(6~8문장이면 0.01 ≈ 1자).

### 7.5 숫자·라틴 — 둘 다 못 한다. TN은 선택이 아니다

CER 집계에서 제외했다. SenseVoice가 역정규화를 해서
올바르게 발음된 "세 시 이십 분"을 "3시 20분"으로 적기 때문이다
(두 프리셋이 **똑같이** 이 가짜 감점을 받아서 발견했다).

| 입력 | matcha-ko 전사 | supertonic 전사 |
|---|---|---|
| 스물세 살이라고 했잖아 | 스물세 살이라고 했잖아 ✅ | **스3살**이라고 했잖아 ❌ |
| 지금 3시 20분이야 | 지금 3시 20분이야 | 지금 **셋시** 20분이야 |
| 12,000원이고 30% 할인 | **만 천원**이고 30%센 (값 틀림: 만 이천) | **12000**고 30 (확장 안 함) |
| 와이파이 비밀번호는 ABC123이야 | 와이파이 비밀번호는 **엠비시**야 (ABC≠MBC) | Why Pi Premier Vonon and ABC8 Samia (판독 불능) |

**정정**: 마지막 행을 처음에 "Supertonic이 한글을 영어로 읽는다 = 언어 조건부 부재의 대가"로
읽었다. **틀렸다.** 통제 실험(같은 한국어 문장에 라틴 유무만 변화)에서 한글은 한글로 유지된다:

```
supertonic  "와이파이 비밀번호는 뭐야?"     -> 와이파이 비밀번는 뭐야?
supertonic  "와이파이 비밀번호는 ABC야?"    -> 와이파이의 비밀번호는 A이비씨야.
matcha      "그 가게 이름이 Starbucks라고"  -> 그 가게 이름이 서벅 사고 했지?
supertonic  "그 가게 이름이 Starbucks라고"  -> 그 가게 이름이 Star박카스라고 했지?
```
한글은 안 뒤집힌다. 라틴 **토큰 자체**만 글자 이름으로 어설프게 읽는다.
앞선 판독 불능 출력은 열화된 오디오에 대해 **SenseVoice가 영어로 오인식**한 것이다.

**따라서 TN 확장기(task #8)의 근거는 Supertonic 고유 결함이 아니라 두 프리셋 공통이다.**
Matcha도 `ABC`→`엠비시`, `12,000`→`만 천원`(값 오류)로 틀린다.
espeak-ng의 한국어 숫자 규칙이 있어서 *뭔가는* 하지만 정확하지 않다.

### 7.6 부수 발견 — 테스트가 우리 측정 워크플로에 취약했다

`tests/test_runtime_budget.py`가 `os.cpu_count`를 monkeypatch하는데
`registry.py:95`는 **`NOBODY_CPU_BUDGET`을 먼저** 본다. 그래서
`NOBODY_CPU_BUDGET=4`가 셸에 남아 있으면 이 테스트가 이유 없이 실패한다 —
그리고 CM4 프록시 측정 워크플로가 정확히 그 변수를 설정하라고 한다.
`monkeypatch.delenv`를 autouse fixture로 추가해 고쳤다.

### 7.7 판정 (트랙 A)

**Supertonic은 현행 기본을 대체할 근거가 아직 없다.** 명료도 동률, 속도 2배 열세,
숫자 처리 열세. 남는 장점은 표에 없는 것들이다:

- **화자 10명 vs 1명** — `tts.py:11-15`가 현재 참조 음성을 "user-facing 전에
  교체할 placeholder"로 표시해뒀고, 이건 그 선택지를 열어준다
- ~~**출처·라이선스가 문서화됨**~~ **← 2026-08-18 정정. 이 근거는 틀렸다.**
  `sherpa-matcha-ko`는 "손수 복사한 커뮤니티 체크포인트"가 아니라 **우리가 직접 학습시킨
  자체 모델**이다(목소리 디자인과 학습 데이터를 Qwen3-TTS로 생성). setup 스크립트가 못
  내려받는 이유는 출처 불명이 아니라 **아직 아무 데도 공개하지 않은 자체 산출물**이기 때문이다.
  → 즉 provenance는 후보들의 **장점이 아니라 단점**이다. 자체 모델이 제3자 가중치보다
  통제권이 크고 라이선스 리스크가 낮다. 코퍼스 쪽 질문(Qwen3-TTS 출력물의 학습 이용)도
  **프로젝트 오너 확인으로 종결**됐다(2026-08-18) → 미결 라이선스 항목 없음. §12.1 참조
- **espeak-ng 의존 없음** — G2P 자체가 불필요(character-level).
  ⚠️ **2026-08-19 정정: 이건 라이선스 이득이 아니다.** espeak-ng은 우리가 쓰는
  `sherpa-onnx-c-api.dll`에 **정적 링크**돼 있어 프리셋을 바꿔도 GPL-3.0 노출은
  남는다(ASR·VAD가 같은 DLL을 쓴다). 실질 이득은 **전방 호환성**이다 — upstream이
  2.0.0에서 espeak-ng을 제거할 예정이고 그때 espeak 음소로 학습된 matcha-ko가
  좌초된다. **§13 참조**

→ **비교용 프리셋으로 유지하고 기본은 바꾸지 않는다.** 채택 판단의 남은 입력은
(a) 사람의 청취 — 명료도는 자연스러움이 아니다, (b) **CM4 실기 RTF**.

---

## 8. 트랙 C-2 게이트 결과 — **만들지 않는다**

§3.6에서 "착수 전 프리필 tok/s를 재고, 프리필이 grace 창을 넘으면 이 기능은 자기모순"이라고
게이트를 걸어뒀다. 쟀다. 결론은 **구현하지 않는 것**이고, 이유는 예상과 달랐다.

**측정** — Windows CPU, `NOBODY_CPU_BUDGET=4`(→ llm 스레드 3개), `midm-2.3b-gguf` Q4_K_M:

| 항목 | 값 |
|---|---|
| 정적 프리픽스 (Mi:dm 내장 시스템 프롬프트 + 우리 페르소나 + few-shot) | **1144 토큰** |
| 사용자 발화 | **3~16 토큰** |
| 콜드 첫 호출 | 7.17s (문서의 6.7s와 일치) |
| 웜, 발화만 새로 프리필 | 153ms(3tok) / 208ms(8tok) / 303ms(16tok) |
| 같은 프롬프트 재호출 | **4ms** |

분해하면 **고정 오버헤드 ~117ms + 토큰당 ~11.6ms** (프리필 약 86 tok/s).

**왜 만들지 않는가 — 세 가지가 겹친다.**

1. **이미 해결된 문제다.** 투기적 프리필이 없애려는 비용은 정적 프리픽스 1144토큰인데,
   그건 `llm.py:425-452` `warm_up()`이 이미 세션 시작 시 인사말 뒤에 숨겨서 지불한다.
   턴마다 남는 프리필은 **발화 3~16토큰뿐**이다. llama-cpp-python 0.3.34가
   `generate()`에서 최장공통접두사 KV 재사용을 자동으로 하므로(소스 확인) 별도 배선도 필요 없다.
2. **남은 비용의 대부분이 투기로 제거할 수 없는 고정 오버헤드다.** 117ms 중
   토큰 작업은 3토큰 발화에서 35ms, 16토큰에서 185ms다. 즉 짧은 턴(우리 캡처 셋의 다수)에서는
   **투기로 아낄 수 있는 최대치가 35ms**다.
3. **트랙 B가 부분 전사를 못 준다.** §7의 chunked SenseVoice 실측 —
   최초 커밋이 1.29초, 16개 캡처 중 8개는 쓸 만한 걸 못 커밋, 게다가 커밋 내용이 뒤집힐 수 있음.
   투기의 트리거 자체가 대부분의 턴에서 존재하지 않는다.

**직접 비교 실험**: 부분 전사("오늘 산책 코스 추천해")로 미리 프리필한 뒤 실제 전사
("...줄래?")를 처리하면 **210ms**, 투기 없이 하면 **253ms**.
→ **약 43ms 절약을, 200ms짜리 추가 프리필 호출을 지불해서 얻는다.**
CM4에서는 양쪽이 같이 커지므로 비율은 그대로 나쁘다. 그리고 낭비된 투기 프리필은
CM4의 유일한 CPU를 점유해 **가속하려던 그 턴을 오히려 늦춘다**(§3.1).

> ⚠️ 43ms 수치는 통제가 완벽하지 않다(투기 없는 쪽은 `_llm.reset()` 후 다른 프롬프트로
> 워밍했다). 방향은 분명하지만 정밀한 값으로 인용하지 말 것. 결론을 지탱하는 건
> 그 숫자가 아니라 **"턴당 프리필이 153~303ms이고 그중 117ms가 고정"**이라는 쪽이다.

**그래서 대신 무엇을 하는가.** 아무것도 안 한다 — 이 경로는 이미 최적이다.
LiveKit·Pipecat이 투기적 생성으로 싸우고 있는 문제를 이 레포는 **프리픽스 KV 캐싱으로
이미 풀어놨다**(2026-08 `warm_up()`). 업계가 그 트릭을 안 쓰는 이유는 그들의 LLM이
원격이라 프리픽스 캐시를 프로세스 안에 들고 있을 수 없기 때문이다. 우리는 로컬이라 들고 있다.

**남는 값진 것 하나**: 위 표의 `4ms`. 프롬프트가 완전히 동일하면 프리필이 사실상 공짜다.
즉 진짜 레버는 투기가 아니라 **프리픽스를 안정적으로 유지하는 것**이고, 그건 이미 그렇다.
만약 나중에 프롬프트 앞부분이 턴마다 바뀌는 변경(예: 시간 표시, 동적 메모리 주입 위치)을
넣는다면 이 4ms가 300ms로 돌아온다 — **그게 이 측정이 지켜야 할 불변식이다.**

---

## 9. 한국어 스트리밍 ASR 대안 (§6-1 미완 축 완료)

§4에서 "이 축은 미완"으로 남겨둔 것을 마무리했다. 결론부터:
**CM4에서 우리 1.29초 하한을 이길 수 있는 후보는 딱 하나, Vosk 한국어다.**
그리고 그것은 최종 텍스트용이 아니라 **partial 전용 채널**로만 쓸 수 있다.

### 9.1 가장 값진 것은 모델 목록이 아니라 아키텍처 판정 기준이다

1초 미만 partial에는 **두 조건이 동시에** 필요하다:
**(a) 프레임 동기** — 시각 t까지의 출력이 t 이후 입력으로 바뀌지 않음.
**(b) 상태·캐시 이월** — 청크마다 앞부분을 재계산하지 않음.

| 계열 | (a) | (b) | 판정 |
|---|---|---|---|
| Kaldi online (**Vosk**) | ✅ | ✅ | 격자를 증분 확장 → partial 단조, 증분 비용이 청크 길이 비례. **200~300ms 가능** |
| Streaming transducer (RNN-T, cache-aware FastConformer/Zipformer) | ✅ | ✅ | **이론적 최적**. 한국어 옵션이 깨진 것(#2886) 아니면 CM4에 10배 무거운 것뿐 |
| Streaming CTC | ✅ | 캐시 학습 시 ✅ | 한국어 Conformer-CTC는 cache-aware 미학습 → 2초 청크에서 20% 열화 |
| **비자기회귀 whole-utterance (우리 SenseVoice)** | ❌ | ❌ | 재해석 가능 + 3.0x 증폭 + `min_decode_s+hop_s` 하한. **청킹으로 고칠 수 없는 구조적 문제** |
| AED (Whisper) | ❌ | ❌ | 최악. 디코더까지 자기회귀 |

→ **§7.4에서 측정한 두 결함(1.29초 하한, 내용 뒤집힘)은 튜닝 실패가 아니라
SenseVoice가 (a)·(b)를 둘 다 안 갖췄기 때문이다.** `hop_s`를 줄여도 (a)는 안 생긴다.

### 9.2 후보 정리

| 모델 | 한국어 | 스트리밍 | 크기 | 라이선스 | 판정 |
|---|---|---|---|---|---|
| **vosk-model-small-ko-0.22** | ✅ | ✅ Kaldi online | 82MB | **Apache-2.0** [1차확인] | **유일한 실현 가능 후보** |
| nemotron-3.5-asr-streaming-0.6b | ✅ ko-KR | ✅ **진짜 cache-aware 80ms** | q4_k 409MB | OpenMDW-1.1 | 기술적 최적이나 **CM4 RTF 8~16 추정 → no-go** |
| SungBeom/stt_kr_conformer_ctc_medium | ✅ | ⚠️ 청크만 | 490MB | ⚠️ **태그 apache-2.0이나 Riva 파생** | **라이선스 선결** |
| onnx-asr | ❌ 한국어 모델 0개 | — | — | MIT | 탈락 (로더로만 가치) |
| Fun-ASR-Nano (streaming) | ❌ zh/en/ja | ✅ | 800M | Apache-2.0 | 탈락 |
| Fun-ASR-MLT-Nano | ✅ 31개어 | ❌ offline | 800M | Apache-2.0 | 탈락 |
| Paraformer streaming | ❌ zh/yue | ✅ | — | — | **한국어 Paraformer는 존재하지 않음** |
| Parakeet v3 / Canary v2 | ❌ 유럽어 25개 | ❌ | — | CC-BY-4.0 | 탈락 |
| whisper.cpp tiny/base | ⚠️ 약함 | ❌ AED | 75/142MB | MIT | 탈락 (아래) |

- **#2886은 여전히 오픈**이고 수정·재export·대체 한국어 모델 제안이 **전부 없다** [1차확인].
  sherpa-onnx의 한국어 스트리밍은 **2024-06 이후 2년 넘게 정체**다.
- **Whisper는 개선이 아니다** [1차확인]: Pi 4(=CM4와 동일 A72) NEON 4스레드에서
  인코더 단독 tiny 13.8s / base 30.6s per 30초 창 → RTF ≈0.46 / ≈1.02(디코더 별도).
  메인테이너가 Pi 4 실시간을 낸 조건은 `tiny.en` + `-ac 512` + **step 4~7.5초** —
  즉 **partial 입도가 우리보다 3~6배 나쁘고 영어 전용**이었다.

### 9.3 Vosk 하이브리드 — 구체적 제안

**Vosk = 저지연 partial + 엔드포인팅, SenseVoice = 최종 텍스트.**
LLM에는 SenseVoice 결과만 넘기고, Vosk partial은 UI 표시·바지인 감지·발화 종료 판정에만 쓴다.

받을 것 딱 둘:
`https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip` (82MB, Apache-2.0)
`pip install vosk==0.3.45` (aarch64 wheel 존재)

**왜 최종 텍스트로 못 쓰는가**: WER **28.1** (Zeroth = 낭독체) [1차확인].
실제 마이크 대화체는 더 나쁠 것 [추론]. 그리고 공식 한국어 모델은 이것 하나뿐이며
Alphacephei의 신세대 zipformer 라인(ru/bn/uz)에 **한국어가 없다** [1차확인].
OVOS도 정확도 때문에 Vosk에서 옮겨갔다 [1차확인].

**명시할 트레이드오프**: (i) Kaldi 런타임 + 82MB 모델 + 런타임 ~300MB RAM이
4GB CM4에서 SenseVoice·LLM·TTS와 경합 → **RAM 예산 확인이 선결**.
(ii) ASR 둘을 동시 구동하면 코어 배분이 SenseVoice의 wall RTF 0.21을 악화시킨다.
(iii) **Vosk의 ARM RTF 공식 수치가 1차 출처에 존재하지 않는다** — 첫 작업은 CM4 실측이어야 한다.

### 9.4 하드웨어 로드맵 인자 — A72가 ISA에서 뒤처진다

**Cortex-A72는 ARMv8.0-A라서 DotProd(SDOT/UDOT, v8.2+), i8mm(v8.6),
FP16 산술(v8.2)이 전부 없다** [추론, 근거 강함]. 따라서 q4_k/q8_0 GGUF가
최신 ARM에서 얻는 정수 가속을 CM4는 거의 못 받는다 —
**양자화로 메모리는 줄지만 연산은 별로 안 빨라진다.**

이게 Nemotron 판정을 뒤집는 지점이다. 그 모델은 한국어 CER 7.12~7.59에
진짜 80ms cache-aware 스트리밍, OpenMDW-1.1, arm64 CPU 프리빌드까지 갖춰
**우리 요구사항을 정확히 만족하는 유일한 모델**인데 CM4에선 못 돈다.
**CM5 / RK3588급(Cortex-A76 = ARMv8.2 + DotProd)으로 올라가는 순간 이게 정답이 된다.**
→ "보드를 구할지 타깃을 재검토할지"(FEATURES.md) 결정의 입력으로 기록.

값싼 검증 하나: parakeet.cpp arm64 프리빌드 + q4_k GGUF(409MB)를 CM4에 올려
**30분 안에 실측 가능**하다. 추정이 틀렸다면 판이 바뀐다.

### 9.5 라이선스 규칙 확장 — HF 태그도 1차 출처가 아니다

`llm-conversational-selection.md`는 "양자화 레포나 기억에서 라이선스를 가져오지 말 것"을
규칙으로 갖고 있다. 이번에 **한 단계 더 필요하다는 사례**가 나왔다.

`SungBeom/stt_kr_conformer_ctc_medium`은 HF cardData에 `license: apache-2.0`이라고
적혀 있고 WER 11.51로 매력적이다. 그런데 카드 본문이 스스로 밝히듯 이 모델은
NGC의 **RIVA Conformer ASR Korean 파인튜닝**이고, 그 NGC 페이지는
*"you would be accepting the terms of the Riva license"* — **NVIDIA 독점, 자유 재배포 불가**다.

→ **파생 모델의 HF 라이선스 태그는 원본 출처를 확인하기 전까지 1차 출처로 취급하지 말 것.**
§7.2의 "번들 안 LICENSE가 MIT인데 가중치는 OpenRAIL-M"과 같은 계열의 함정이고,
이번 주에 이 프로젝트가 만난 세 번째 라이선스 함정이다.

---

## 10. CM4 타당성 — **보드를 사지 말고 타깃을 CM5로 올려라**

`docs/FEATURES.md`가 "보드를 구할지 타깃을 재검토할지는 결정 사항"으로 남겨둔 것을
1차 출처로 종결한다. **결론: CM4를 사지 말 것.** 이유가 튜닝으로 회복 불가능한 ISA 결함이다.

### 10.1 A72는 llama.cpp의 ARM 빠른 경로를 **전부** 못 쓴다

| 코어 | 아키텍처 | dotprod | i8mm | FP16 산술 |
|---|---|---|---|---|
| **Cortex-A72 (CM4)** | **Armv8.0-A** | **없음** | **없음** | **없음** |
| Cortex-A76 (CM5/Pi 5) | Armv8.2-A | **있음** | 없음 | 있음 |
| Cortex-A720 (CIX P1) | Armv9.2-A | 있음 | **있음** | 있음 |

[1차확인] GCC `aarch64-cores.def`: `AARCH64_CORE("cortex-a72", …, V8A, (CRC), …)` —
Armv8.0 + CRC뿐. `cortex-a76`은 `V8_2A, (F16, RCPC, DOTPROD)`.
LLVM `AArch64Processors.td`도 동일(`cortex-a72` → `HasV8_0aOps`, DotProd/MatMulInt8 없음).
**i8mm은 어느 Cortex-A7x에도 없다** — i8mm을 원하면 Armv9.2급으로 가야 하고 그건 모듈 폼팩터 포기다.

**결정적 [1차확인]** — `ggml/src/ggml-cpu/ggml-cpu-impl.h`:
dotprod가 없으면 `ggml_vdotq_s32`가 **NEON 명령 6개로 에뮬레이션**된다(SDOT 1개 자리에
`vmull_s8` 2 + `vpaddlq_s16` 2 + `vaddq_s32` 2). 이게 디코드 내부 루프의 실제 비용이다.

그리고 우회로가 없다 [1차확인]:
- `repack.cpp`의 커널 선택 **전체**가 `ggml_cpu_has_dotprod()` 게이트 안 → **NEON-only 분기가 아예 없다.**
  대상 7종(Q4_0, Q4_K, Q5_K, Q6_K, IQ4_NL, MXFP4, Q8_0) 전부 해당. **IQ4_NL로 도망갈 수도 없다.**
- `sgemm.cpp`의 `tinyBLAS_Q0_ARM` 클래스 전체가 `#if defined(__ARM_FEATURE_DOTPROD)` 안.
- `quants.c`의 Q4_K 2-row 배치 경로는 `__ARM_FEATURE_MATMUL_INT8` 필요.

**그 빠른 경로가 주는 실측 이득**[1차확인, llama.cpp PR #5780 / #10541]:
Graviton3에서 Llama-2 7B Q4_0 1스레드 **프리필 2.57x / 디코드 2.12x**,
M2에서 IQ4_NL repack **pp256 3.08x**.
→ **A72는 프리필 약 2.5~3x, 디코드 1.3~2.1x를 구조적으로 잃는다** [추론].

### 10.2 동일 실리콘 실측 — 두 자리 초 응답

[2차, 방법 명시된 llama-bench] Pi 4 8GB, Gemma 4 E2B **Q4_K_M 2.88GiB**:
`pp512 4.06 / tg128 1.68 t/s`. 같은 문서 Pi 5: `31.86 / 6.71`.
→ **프리필 7.8x, 디코드 4.0x 차이.** 클럭비는 1.33x, Geekbench6 싱글코어비는 2.4~2.7x.
**프리필에만 남는 ~3x 잔차가 곧 dotprod repack GEMM이다** [추론] — §10.1 소스 분석과 정확히 일치.

문서 원문: Pi 4는 턴이 진행되며 프리필이 5.3→3.4 t/s로 **열화**하고 원인은
*"memory bandwidth saturation as the KV cache grows"*. Pi 5엔 이 열화가 없다.

**CM4 예측** [추론 — Pi 4B 1.8GHz 기준을 모델 크기 2.01x·클럭 0.83x로 환산]:

| | CM4 (4×A72@1.5GHz, 4T) |
|---|---|
| 디코드 | **2.5~3.5 t/s** |
| 프리필 | **5~9 t/s** (개발박스 86 → **12배 느림**) |
| 세션 시작 warm-up 1144토큰 | **130~230초** |
| 45토큰 응답(한글 ~40자) | **13~18초** |

**대역폭도 이미 천장이다** [1차확인+추론]: Pi 4 tinymembench memcpy 2737 MB/s인데
실측 디코드가 이미 ~5.2 GB/s 가중치 읽기 = 로프에 걸려 있다.
RAM을 8GB로 늘려도, 스레드를 재배분해도 이건 안 움직인다.

### 10.3 죽는 건 LLM 하나다 — 그리고 우리 스레드 예산이 TTS도 죽인다

[1차확인] sherpa-onnx 공식 문서의 **Raspberry Pi 4 실측** Matcha RTF:
`1T 0.941 / 2T 0.561 / 3T 0.451 / 4T 0.411`. **4스레드면 A72에서 실시간이다.**

§7.4에서 "개발박스 RTF 0.37 vs Pi 4 0.411이 거의 같다"는 불일치가 있었는데 **풀렸다 —
스레드 수가 달랐다.** 우리 측정은 `NOBODY_CPU_BUDGET=4`에서 `runtime.yaml`이
tts에 **1스레드**만 주기 때문이다(fraction 0.25 × 4 = 1). 같은 조건으로 다시 재면:

| | 1T | 2T | 3T | 4T |
|---|---|---|---|---|
| 우리 개발박스, matcha-ko | **0.268** | 0.203 | 0.191 | 0.178 |
| Pi 4, matcha-en [1차확인] | **0.941** | 0.561 | 0.451 | 0.411 |

→ **코어당 개발박스가 A72보다 3.5배 빠르다.** 4스레드 대비로는 2.3배.

**그런데 여기서 새 문제가 나온다.** CM4(1.5GHz = Pi 4의 0.83x)에서
**우리 파이프라인이 TTS에 주는 1스레드**면 RTF ≈ 0.941/0.83 = **1.13 — 실시간보다 느리다.**
스트리밍 재생이 따라가지 못한다. Supertonic은 Matcha의 2배 느리므로 1스레드에서 **RTF ~2.3**,
CM4에서 확실히 사용 불가.

→ **`configs/runtime.yaml`의 스테이지 배분(llm 0.75 / tts 0.25 / asr 0.75 = 4코어에 7스레드)은
CM4에서 재검토가 필요하다.** 이건 하드웨어가 아니라 설정 문제이므로 보드 결정과 별개로 유효한 항목이다.
(`cpu_budget: null` 주석이 "CM4 실측 전까지는 고정하지 말 것"이라 해둔 게 바로 이 지점이다.)

### 10.4 보드 비교 — NPU TOPS는 무의미하다

| 보드 | 코어/클럭 | dotprod | i8mm | RAM | NPU 런타임 성숙도 | 가격 | 판정 |
|---|---|---|---|---|---|---|---|
| **CM4** | 4×A72@1.5 | ✗ | ✗ | ~8GB | 없음 | 4GB ~$95–110, 8GB ~$160, **리드타임 10–12주** | **LLM 불가. 이제 싸지도 않다** |
| **CM5 / Pi 5** | 4×**A76**@2.4 | **✓** | ✗ | 2–16GB | 없음 | **from $67.50**, 생산 **2036**까지 | **최적** |
| Radxa CM5 (RK3588S2) | 4×A76 + 4×A55 | ✓ | ✗ | **2–32GB** | 6 TOPS, **부분적** | ~$99–155 | CM4 캐리어 재사용 + RAM 필요 시 |
| Jetson Orin Nano Super 8GB | 6×A78AE + Ampere | ✓ | ✗ | 8GB, **102 GB/s** | **CUDA/TensorRT = 최고 성숙** | $249 | LLM이 여유로워지는 유일한 옵션 |

**NPU 현실 점검** [1차확인]: 우리 세 모델 타입(GGUF LLM / ONNX ASR / ONNX TTS)을
전부 커버하는 NPU 런타임은 **없다.**
- RK3588 RKNN: sherpa-onnx에 **SenseVoice(#2592)·Silero VAD(#2067)** 경로 있음.
  **TTS RKNN 항목은 없다** — Matcha/Kokoro/VITS는 CPU만.
- LLM은 RKLLM 전용(`.rkllm` 포맷, GGUF 아님), **W8A8만 — W4 없음**(2.3B ≈ 2.3GB).
  Radxa 실측 Qwen2.5-1.5B 15.44 t/s.
- ⚠️ `rknn-toolkit2` LICENSE = **"RKNN SDK License"** — Rockchip 제품 호환 목적에만 허용,
  리버스 엔지니어링 금지, **Rockchip이 사유 없이 언제든 해지 가능.** 상업 제품이면 법무 검토 필수.

→ **결정은 "NPU가 있나"가 아니라 "코어당 성능 + dotprod가 있나"로 내려야 한다.**

### 10.5 보드와 무관하게 지금 당장 할 것 — warm-up을 디스크로

**이번 조사에서 가장 값진 구현 항목이다** [1차확인].
`llama_cpp/llama_cpp.py`에 **`llama_state_seq_save_file` / `llama_state_seq_load_file` /
`llama_state_seq_get_data`가 이미 바인딩돼 있다** — 고수준 `Llama`가 안 쓸 뿐이다.

→ 1144토큰 warm-up 프리픽스를 **빌드/최초 부팅 시 한 번 만들어 파일로 저장**하고
매 프로세스 시작 때 로드하면, CM4의 **130~230초 warm-up이 파일 읽기로 바뀐다.**
어떤 보드를 고르든 순이득이다. (`Llama.save_state()`는 쓸 수 없다 — 컨텍스트 전체 블롭이라
1144토큰이 ~215 MiB, 4GB에서 불가.)

참고 [1차확인]: Mi:dm KV는 f16 **192 KiB/token**(48층 × 8 KV헤드 × 128 × 2 × 2B).
`type_k`/`type_v`로 q8_0 양자화하면 96 KiB/token으로 반감되지만
**속도에는 무의미하다** — 디코드는 KV가 아니라 1.43GB 가중치 대역폭에 막혀 있다.
RAM 절약 목적으로만 의미가 있고, 양자화 V는 보통 flash attention을 요구해 CPU 백엔드에서 검증 필요.

### 10.6 부수 정정 — GLaDOS의 600ms는 측정치가 아니다

[1차확인] GLaDOS 저장소에 **전체 시스템의 측정된 레이턴시 벤치마크가 없다.**
저자 진술은 *"Getting round-trip response time under 600 milliseconds is a threshold"* —
**목표 서술**이다. RK3588 언급도 *"Runs on a Rock5b with RK3588 NPU"*뿐 수치 없음.
우리 survey가 이를 "**~600ms 왕복 목표**"로 적은 건 맞지만, 달성된 실측으로 읽히지 않도록
**검증되지 않은 주장**임을 명시한다.

[1차확인] **OpenLive**는 MIT이고 부품 선택이 우리와 거의 동일(Silero VAD + Smart-Turn +
Kokoro/Supertonic)하지만 **런타임이 앱 내 WebGPU**이고 CPU-only/네이티브 ARM 경로가
문서화돼 있지 않다 → **CM4급 CPU엔 그대로 못 쓴다.** 한국어 지원 명시 없음.

### 10.7 권고와 실행 순서

**CM4를 사지 말고 타깃을 CM5(또는 Radxa CM5)로 올린다.** 근거는 §10.1의 ISA 결함(회복 불가),
§10.2의 동일 실리콘 실측(두 자리 초 응답), 그리고 CM4의 가격·리드타임 우위 소멸이다.
마이그레이션 비용은 낮다 — CM5는 동일 55×40mm + 2×100핀이다(단 일부 핀 변경,
composite/2-lane MIPI 제거, 5V/5A 권장 → 캐리어 검증 필요 [2차]).

1. **CM5(또는 Pi 5) 1장으로 실측** — `llama-bench` pp512/tg128 @4T + SenseVoice int8 RTF
   + Matcha RTF. CM4 4GB를 사는 것보다 싸고 정보량이 많다.
2. **보드와 무관하게 즉시**: §10.5의 KV 프리픽스 파일화.
3. **`runtime.yaml` 스테이지 배분 재검토** (§10.3) — TTS 1스레드는 CM4에서 실시간 미달.
4. CM4를 굳이 산다면 **4GB가 아니라 8GB**, 그리고 제품 타깃이 아니라 **바닥 성능 참조기로만.**

### 10.8 남은 미지

- **CM4/CM5 실기 llama.cpp 수치는 웹에 존재하지 않는다.** 전부 Pi 4B(1.8GHz) 역산이고,
  앵커 모델(Gemma E2B 2.88GiB)이 MatFormer 계열이라 dense인 Mi:dm과 활성 가중치 비율이
  다를 수 있어 **디코드 예측이 낙관 쪽으로 틀릴 여지가 있다.**
- **A72에서의 SenseVoice int8 RTF가 확인되지 않았다.** A55(0.175 @4T)와 A76(0.049 @4T)
  사이라는 것만 안다. dotprod 부재가 ONNX Runtime MLAS의 int8 GEMM에 얼마나 걸리는지 미측정.
- Arm 공식 문서(109697) 본문은 리다이렉트로 직접 회수 실패 — 다만 A72 = Armv8.0
  no-dotprod/no-i8mm 결론은 GCC·LLVM 소스로 독립 이중 검증됐다.
- ~~CM5 캐리어 호환성은 2차 출처만 확인~~ → **§10.5에서 데이터시트 원문으로 종결**(2026-08-19).

---

## 11. 잔여 축 통합 (TTS 후보 · 배포 아날로그) + 라이선스 함정 4·5번

두 개의 병렬 스윕이 늦게 돌아왔다. 이미 §9·§10에 있는 내용은 생략하고 **바뀌는 것만** 적는다.

### 11.1 🚨 우리 **기본 ASR**의 가중치가 Apache도 MIT도 아니다 (라이선스 함정 4)

**직접 확인했다.** `models/sense-voice/LICENSE`의 전체 내용:

```
Ref to https://github.com/modelscope/FunASR?tab=readme-ov-file#license
```

라이선스 파일이 아니라 **링크 한 줄**이다. HF 카드도 `license: other`,
`license_name: model-license` → **FunASR Model Open Source License v1.1**(Alibaba)이고,
보고에 따르면 조항 4.2가 *"unjustified denigration"*에 대해 자동 해지된다 [2차].

**이건 비교용 프리셋이 아니라 `configs/models.yaml`의 기본 ASR이다.**
Supertonic의 OpenRAIL-M을 신중히 다뤘는데, 정작 이미 기본값으로 쓰고 있는 스테이지의
가중치 라이선스를 아무도 확인한 적이 없다. **제품화 전 법무 확인 항목.**

> 코드(FunASR)는 MIT여도 **가중치는 별개**다 — §7.2(Supertonic 번들의 MIT LICENSE가
> 코드용이었던 것), §9.5(파생 모델의 HF 태그)와 정확히 같은 계열의 세 번째 변형이다.

### 11.2 Supertonic 재평가 — 우리 측정보다 좋고, 우리가 못 쓰는 레버가 하나 있다

새 1차 정보 둘:
- **한국어 CER 3.26** — Supertonic README의 언어별 표에서 Qwen3-TTS 4.07, VoxCPM2 4.70을
  이긴다 [1차확인]. §7.4에서 우리가 "명료도 동률"로 측정한 것과 모순이 아니다 —
  우리 판정자(SenseVoice ITN)의 해상도가 낮았을 뿐이다.
- **RPi4B(우리와 동일 SoC) RTF ≈0.44** [2차+추론: 141자 문장 4.25초 wall,
  오디오 길이·스레드 수 미공개]. RPi5는 5스텝에서 0.150.

**왜 A72에서 살아남는가** [추론, 근거 있음]: flow-matching은 **비자기회귀**라
처리량이 int8 GEMM에 의존하지 않는다. §10.1의 dotprod 부재가 LLM을 죽이는 이유가
여기엔 적용되지 않는다. (audio.cpp가 Supertonic을 **F32 전용**으로 싣는 것도
int8이 A72급에서 이득이 없다는 약한 증거다.)

**그런데 권고된 레버는 우리 경로에 없다 — 직접 확인했다.**
"스텝 수를 8→5→2로 런타임에 조절"이 Supertonic의 핵심 장점으로 제시됐지만,
`OfflineTtsSupertonicModelConfig`에 **스텝 필드가 없고**(duration_predictor / text_encoder /
tts_json / unicode_indexer / vector_estimator / vocoder / voice_style 뿐),
`tts.json`에도 step 키가 없다. `OfflineTtsConfig`에도 없다.
→ **sherpa-onnx 경로로는 스텝 수를 못 바꾼다.** 그 속도 이득을 원하면
upstream Python SDK(자체 의존성 고정)나 **v2 번들(5스텝 고정, 81MB)** 중 하나여야 한다.
v2가 한국어를 지원하므로(en/ko/es/pt/fr) **v2를 재보는 것이 값싼 실험이다.**

### 11.3 MeloTTS-Korean — OpenRAIL-M을 피할 실제 대안이 하나 있다

§1.3에서 "한국어 CPU TTS 기성품이 없다"고 정리했는데 **하나 놓쳤다** [1차확인]:

**`myshell-ai/MeloTTS-Korean` — 코드와 가중치 모두 MIT.** 한국어 체크포인트가 실제로 배포됨.
VITS2 단일 forward pass, fp32 162MB / **int8 50MB**.
그리고 한국어에 유리한 구조적 특성: **한국어는 BERT prosody 분기를 건너뛴다**
(카드: `bert`/`ja_bert`가 KR에선 zero tensor) → 1024-dim BERT 인코더 비용을 안 낸다.

단점: **어떤 하드웨어에서도 RTF가 공개되지 않았고**, 베이스 레포가 2024-12-24 이후 방치,
화자 1명, 품질은 Supertonic보다 낮을 것, 그리고 g2pkk + mecab-ko-dic G2P 체인을 물고 온다
(§1.3의 mecab 문제로 되돌아간다).

→ **OpenRAIL-M이 법무에서 막히면 이게 답이다.** 그 경우에만 착수.

### 11.4 Piper 한국어는 **존재한다. 단 non-commercial** (라이선스 함정 5)

§1.3에 "Piper 한국어 없음"으로 적었는데 부정확했다.
**`ko_KR-kss` 음성이 존재하고, KSS 데이터셋 유래라 CC-BY-NC-SA-4.0이다** [2차].
즉 "없다"가 아니라 "상업 이용 불가"다. 결론은 같지만 이유가 다르므로 정정한다.

같은 함정의 목록 — **permissive 코드 아래 CC-BY-NC 가중치** [2차]:
F5-TTS, Spark-TTS, OuteTTS 1.0, OpenAudio S1-mini, MMS-TTS-kor, Piper `ko_KR-kss`.

### 11.5 Smart Turn v3.2의 ARM 지연 — 예산에 없던 비용

우리 `configs/turn_detector.yaml`은 "CPU ~12ms"를 전제로 서 있다. 그런데 [2차]:
**AWS c8g.medium(Graviton, 1 vCPU)에서 159ms**, x86 c7a.2xlarge에서 9ms.

우리 스레드 예산은 turn detector에 별도 배분이 없고(`detector.py:63`이
`providers=["CPUExecutionProvider"]`로 고정), **이건 VAD 침묵 뒤에 실행되므로
턴 지연에 그대로 더해진다.** CM4에서 100ms대라면 `barge_in_confirm_ms: 250`,
`endpoint_grace_min_ms: 300`과 같은 자릿수가 되어 설계 전제가 흔들린다.

추가로 [추론]: v3.1이 실제 사람 음성을 얻은 건 영어(88.3→94.7%)와 스페인어뿐이라
**한국어는 여전히 합성 학습일 가능성이 높다.** 언어별 정확도 표는 미공개.

→ **CM4 측정 항목에 Smart Turn 추론 시간을 추가**한다. TTS·ASR만 재면 안 된다.

### 11.6 독립적으로 수렴한 아키텍처 — dual-STT 티어링

§9.3에서 제안한 Vosk 하이브리드(저지연 partial + 정확한 최종)와 **똑같은 구조**를
`RunanywhereAI/RCLI`가 이미 쓰고 있다 [2차]: *"Zipformer streaming + Whisper / Parakeet offline"*.
서로 모르는 두 조사가 같은 답에 도달한 건 설계 검증으로 받아들일 만하다.

함께 훔칠 것 [1차확인, OpenLive 소스]: **end-of-turn 모델을 ASR 디바이스와 분리**한다.
OpenLive는 ASR이 WebGPU에 있어도 Smart-Turn은 **무조건 CPU EP**에서 돌리고,
whisper-tiny의 mel 프론트엔드를 turn 모델의 feature processor로 **재사용**하며,
세션 생성을 try/catch로 감싸 실패 시 `turnSession = null`로 두고 **plain VAD 타임아웃으로
우아하게 강등**한다. 우리 `--endpoint-detect`가 지금 실패 시 어떻게 되는지 확인할 값이 있다.

### 11.7 확정된 사망 목록 (재조사 금지)

[1차/2차 확인] **Kokoro** — 한국어 음성이 **한 번도 배포된 적 없음**, `language:['en']`,
issue #294가 2026-01-08 이후 무응답, 레포 2025-08-06 정지. §1.2 판정 확정.
**ZipVoice** — README가 "Chinese and English"만 명시. 우리 sherpa 1.13.4에
`OfflineTtsZipvoiceModelConfig`가 있어도 한국어 체크포인트가 없다.
⚠️ 함정 기록: ZipVoice README의 언어 바는 **i18n 링크**지 지원 언어가 아니다.
**KittenTTS**(영어 전용), **PocketTTS**(RTF 0.021 GGUF 2코어 — 이 조사 최고의 CPU 수치인데
en/fr/de/pt/it/es, 한국어 없음), **Qwen3-TTS 0.6B**(Apache-2.0 + 좋은 한국어지만
**Pixel 8a에서 RTF 2.06** → 산술적으로 불가), **OmniVoice**(한국어 8,609시간, Apache-2.0이나
0.6B diffusion-LM, H100 벤치만, ONNX export 없음), **TEN Turn Detection**(Qwen2.5-7B, 영/중).

⚠️ **TEN VAD** 주의 [1차확인]: Linux 프리빌드가 **x64 전용**이고 ARM은 Android/iOS만이다.
우리는 sherpa-onnx 경유라 괜찮을 것이나 **CM4에서 확인 필요**.

### 11.8 KsponSpeech — #2886이 고쳐져도 라이선스가 남는다

[1차확인] `sherpa-onnx-streaming-zipformer-korean-2024-06-16` HF 레포에 **라이선스 태그가 없다.**
학습 데이터는 **KsponSpeech**(AI Hub / NIA Korea)이고 취득에 **신청·승인이 필요**하다.
상업 이용 및 모델 재배포 조건을 공개 출처로 확인할 수 없었다 [확인 불가].

→ §9의 "#2886이 막힌 길"에 **두 번째 차단 요인**이 겹친다. 설령 상류가 고쳐도
이 체크포인트를 제품에 넣으려면 KsponSpeech 조건을 먼저 정리해야 한다.
한국 법인이라 신청 자체는 유리한 위치다.

---

## 12. TTS 결론 — 자체 모델을 유지한다 (그리고 §7의 측정법이 틀렸다)

### 12.1 정정 먼저: `sherpa-matcha-ko`는 우리가 학습시킨 모델이다

§7.7에 "손수 복사한 커뮤니티 체크포인트(출처·라이선스 불명)"라고 적었다. **틀렸다.**
직접 학습시킨 자체 모델이고, **목소리 디자인과 학습 데이터를 Qwen3-TTS로 생성**했다.
`matcha-ko-voiceA-ep499-steps10` = voice A / epoch 499 / 10 flow-matching steps.

setup 스크립트가 못 내려받는 이유는 출처가 의심스러워서가 아니라
**아직 공개하지 않은 자체 산출물**이기 때문이다. 이건 결론을 뒤집는다 —
provenance는 대안들의 **장점이 아니라 단점**이다. 자체 가중치가 제3자 것보다
통제권이 크고(교체가 아니라 **재학습**이 가능하다) 사용 제한 조항도 없다.
Supertonic의 OpenRAIL-M에는 있다.

**라이선스는 정리됐다.** 가중치는 자체 소유이고, Qwen3-TTS 출력물의 학습 이용은
**프로젝트 오너가 가능함을 확인**(2026-08-18). 즉 이 프리셋에는 미결 라이선스 항목이
없고 하위 사용 제한도 없다 — 비교 대상이었던 제3자 후보 전부와 반대다.

### 12.2 측정법 결함 — 합성이 프로세스마다 다르다

§7.4의 CER 수치들은 **각 1회 측정이었고, 그건 측정이 아니었다.**
같은 프리셋·같은 문장으로 **0.032 / 0.043 / 0.074**가 나왔다. 원인을 추적했다:

- ASR 판정자는 완전 결정론적이다 (같은 wav → 바이트 동일 전사, 확인)
- **TTS가 프로세스마다 다른 오디오를 낸다** — md5도 다르고 **샘플 수도 다르다**(96312 vs 96214 바이트)
- Matcha는 flow-matching이고 `noise_scale`이 있다. sherpa-onnx가 프로세스 단위로 시드하므로
  한 프로세스 안에서는 재현되지만 프로세스 간에는 안 된다.
  `OfflineTtsConfig`·`generate()`에 **시드 파라미터가 없다** — 고정할 방법이 없다.

→ **94자 세트에서 CER 해상도는 약 ±0.04다.** 그 밴드 안의 비교는 무의미하다.

**이것이 무효화하는 §7의 주장 둘:**
1. "Supertonic 3이 matcha와 명료도 동률(둘 다 0.074)" — 겹치는 분포에서 뽑은 단일 샘플 두 개였다.
2. "화자 10명 중 sid=7이 최고(0.025), sid=2가 최악(0.089)" — **그 순위 전체가 노이즈였다.**
   `configs/models.yaml`의 `speaker_id: 7`은 이제 "임의 선택"으로 주석을 고쳤다.

`scripts/_ab_tts.py`에 `--repeat`(기본 3)을 넣고 **중앙값 + 관측 min-max**를 출력하게 고쳤다.
구간이 겹치면 판별 불가라고 표에 직접 적는다.

### 12.3 반복 측정 결과 — 자체 모델이 이긴다

3회 반복, 8문장, `NOBODY_CPU_BUDGET=4`(tts 1스레드):

| preset | CER 중앙값 | CER min-max | rtf~ | rtfMx | rate | load |
|---|---|---|---|---|---|---|
| **sherpa-matcha-ko** (자체) | **0.043** | **0.021–0.053** | 0.27 | 0.32 | 22050 | 7.8s |
| supertonic-3-ko | 0.064 | 0.064–0.074 | 0.47 | 0.60 | 44100 | 0.8s |
| supertonic-2-ko | 0.074 | 0.064–0.117 | **0.16** | **0.21** | 44100 | 0.7s |

- **matcha-ko의 구간이 두 Supertonic 어느 쪽과도 겹치지 않는다** → 명료도 우위는 **실재한다.**
  §7의 "동률"과 정반대 결론이고, 반복 측정이 있어야 나오는 결론이다.
- **Supertonic 2의 속도 주장은 확인됐다**: v3보다 **2.9배**, matcha보다 **1.7배** 빠르다.
  v3가 flow-matching 스텝을 5→8로 올렸기 때문이고, 그 스텝 수는 sherpa-onnx로 조절 불가
  (§11.2에서 확인). 즉 **v2가 싼 설정에 접근하는 유일한 경로**다.
- v2와 v3의 CER 구간은 겹친다(0.064–0.117 vs 0.064–0.074) → 상류가 보고한
  "CER 3.65 vs 3.26"과 모순은 아니지만 **우리 세트로는 판별 불가**다.
- 로드 시간은 Supertonic이 10배 빠르다(0.7s vs 7.8s). `warm_up()`이 가리는 일회성 비용이다.

### 12.4 결론

**`sherpa-matcha-ko`를 기본값으로 유지한다.** 근거:
① 명료도가 판별 가능한 차이로 가장 좋다 ② 자체 학습이라 통제권이 있고 재학습이 가능하다
③ 0.27 RTF로 이미 예산 안에 있다 ④ 제3자 사용 제한 조항이 없다.

**속도를 위해 품질과 소유권을 내줄 이유가 없다.** CM4에서 병목은 TTS가 아니라 LLM이다(§10.2).

**단 조건부로 v2가 다시 관련해진다.** 우리 스레드 예산이 TTS에 1스레드만 주므로(§10.3)
CM4 환산으로 matcha는 **RTF ≈1.13(실시간 미달)**, supertonic-2는 ≈0.67이다.
즉 CM4 타깃을 유지하면서 스레드 배분을 안 고치는 경우에만 v2가 답이 된다.
**스레드 배분을 고치는 게 먼저다** — matcha에 2스레드면 ≈0.68로 같은 곳에 도달하면서
품질과 소유권을 지킨다. `supertonic-2-ko` 프리셋은 그 선택지를 열어두기 위해 남긴다.

**남은 확인 항목** (라이선스는 §12.1에서 종결됐다): (a) 사람 청취 — CER은 명료도지
자연스러움이 아니다 (b) Supertonic 화자 10명 중 선택은 아직 임의다.

---

## 13. espeak-ng / GPL-3.0 — **라이선스 문제는 모델이 아니라 런타임에 있다** (2026-08-19)

§12에서 "keep matcha-ko"로 닫은 뒤, Supertonic의 장점으로 "espeak-ng 의존 없음"을
들었다(§11 표, §7 목록). **그 근거는 우리가 실제로 배포하는 바이너리를 확인하지 않은
채 쓴 것이었고, 확인해보니 틀렸다.** 아래가 확인 결과다.

### 13.1 espeak-ng은 sherpa-onnx 바이너리에 정적 링크돼 있다 [1차확인]

우리가 설치해 쓰는 휠(`sherpa_onnx 1.13.4`, `sherpa-onnx-c-api.dll`, 4.5MB) 안의 문자열:

```
espeak_ng_Initialize      espeak_ng_SetVoiceByName    espeak_ng_SetPhonemeEvents
espeak_ng_InitializePath  espeak_ng_SetVoiceByFile    espeak_ng_CompileDictionary
espeak_ng_GetSampleRate   espeak_ng_SetParameter      espeak_ng_SetRandSeed
...
?phonemize_eSpeak@piper@@YAXV?$basic_string@...        ← piper-phonemize C++ 심볼
/usr/share/espeak-ng-data                              ← 하드코딩된 폴백 경로
%s/espeak-ng-data
```

스텁이 아니라 **espeak-ng 라이브러리 전체**와 **piper-phonemize**가 컴파일돼 들어 있다.

**그래서 프리셋을 바꿔도 노출은 사라지지 않는다.** 이 프로젝트에서 같은 DLL을
로드하는 것은 TTS만이 아니다 — **SenseVoice ASR, TEN-VAD, streaming ASR이 전부 같은
`sherpa-onnx-c-api.dll` 하나를 쓴다**. matcha-ko를 Supertonic으로 바꾸면 18MB
`espeak-ng-data` 디렉터리는 안 실어도 되지만, **링크된 GPL-3.0 코드는 그대로 남는다.**

> ⚠️ 따라서 §11·§12에 적은 "Supertonic 장점: espeak-ng 의존 없음"은 **데이터 디렉터리
> 수준에서만 참이고 바이너리 수준에서는 거짓**이다. 라이선스 노출을 줄이는 근거로는
> 쓸 수 없다. §13.3의 이유로 대체한다.

### 13.2 upstream이 이 충돌을 인정하고 제거를 예고했다 [1차확인]

sherpa-onnx issue [#3731](https://github.com/k2-fsa/sherpa-onnx/issues/3731)
("Breaking Change: Remove espeak-ng and piper-phonemize dependency", 2026-07-08 개설):

> "Since `espeak-ng` is licensed under GPL, it introduces license constraints that
> are incompatible with the Apache-2.0 license of sherpa-onnx. To keep sherpa-onnx
> fully compatible with Apache-2.0 licensing, we will remove the `espeak-ng`
> dependency."

- **메인테이너 본인들의 판정**이다. "GPL이 걸리는지"는 더 이상 우리 해석 문제가 아니다.
- 제거는 **2.0.0**(메이저)로 예고됐고 **아직 안 나왔다** — PyPI 최신은 `1.13.6`. [1차확인]
- 마이그레이션 경로: `lexicon.txt` 제공, 또는 외부 음소화 후 `GenerationConfig`의
  신규 `tokens` 필드로 직접 전달.
- 즉 지금 우리가 쓰는 1.13.4는 **Apache-2.0을 주장하지만 GPL-3.0 코드를 링크한 상태**다.
  실제 의무 판단(배포 시 소스 제공·라이선스 고지)은 사람이 할 일이고, 여기서는
  사실만 기록한다.

### 13.3 진짜 Supertonic 장점은 라이선스가 아니라 **전방 호환성**이다 [1차확인]

`matcha-ko`의 `tokens.txt`를 열어보면 **espeak IPA 음소 목록**이다:

- 159개 토큰, 앞 14개가 `sherpa-matcha-en`의 것과 **바이트 단위로 동일**
- 꼬리가 IPA 구별기호 — `ʰ ˤ ʦ ̧ ̃ ̪ ̯ ̩ ̝ ̊`
- 모델 자신의 ONNX 메타데이터도 그렇게 말한다:
  `comment: "Korean Matcha-TTS, espeak-ng ko phonemes, icefall tokens.txt ids"`
  (`tts.py:347-350`에 이미 인용돼 있었다)

한글도 자모도 아니다. **matcha-ko는 espeak-ng을 런타임 G2P로 실제로 쓴다.**

→ **2.0.0이 espeak-ng을 제거하면 우리 자체 모델이 1.x에 좌초된다.** 계속 쓰려면
한국어 `lexicon.txt`를 만들거나, 정확히 저 159개 음소를 뱉는 외부 음소화기를 붙여야
한다(그리고 §11이 정리한 대로 한국어 G2P 후보는 전부 GPL·JVM·aarch64 휠 부재 중 하나에
걸린다). Supertonic은 character-level이라 **아무 영향이 없다.**

이게 §12의 "keep matcha-ko" 결론에 붙는 유일한 실질 위험이다. 결론은 유지한다 —
2.0.0은 아직 없고, 우리는 버전을 락하고 있고, 품질·소유권 우위는 그대로다. 다만
**"버전을 올릴 수 없는 이유"가 하나 생겼고**, 그건 기록해야 한다.

### 13.4 `espeak-ng-data`가 없으면 프로세스가 죽는다 (C 레벨, 잡을 수 없음) [1차확인]

배포 이미지에서 18MB 디렉터리를 빼면 어떻게 되는지 실측했다. 조건별로 **프로세스를
분리해서** 재야 한다 — espeak-ng은 프로세스 전역 싱글턴이라 한 프로세스에서 정상 경로를
먼저 초기화하면 뒤이은 잘못된 경로가 그 상태를 재사용해 **거짓 통과**한다(처음 이렇게
재서 "없어도 동일한 오디오가 나온다"는 잘못된 결과를 얻었다).

| 조건 | 결과 |
|---|---|
| `data_dir` 정상 | OK — 54192 samples, 2.46s, peak 0.4286 |
| `data_dir` 없는 경로 | **rc=1, Python 예외 없음.** `Error processing file '/usr/share/espeak-ng-data\phontab': No such file or directory.` |
| `data_dir` 빈 문자열 | **rc=1, Python 예외 없음.** `Error processing file '.\phontab'` |

두 가지가 나온다:

1. **잡을 수 없는 실패다.** `try/except`에 걸리지 않고 프로세스가 그대로 종료된다.
   TTS만 degrade되는 게 아니라 **음성 에이전트 전체가 죽는다**.
2. **하드코딩된 `/usr/share/espeak-ng-data`로 폴백한다.** 리눅스 타깃(CM4/CM5)에서
   시스템 espeak-ng이 깔려 있으면 위 실패가 **에러가 아니라 조용한 성공**이 된다 —
   우리 경로가 아니라 시스템 사전을 쓰고, 버전이 다르면 음소가 달라진다. 개발 박스에는
   `/usr/share`가 없어서 죽지만, 배포 타깃에서는 안 죽고 잘못 발음할 수 있다.

**경로 우선순위를 소스로 확정했다** [1차확인, espeak-ng `src/libespeak-ng/speech.c`
`espeak_ng_InitializePath`] — 이게 위 2번의 범위를 좁힌다:

```c
if (check_data_path(path, 1)) return;                          // 넘긴 경로가 최우선
if (check_data_path(getenv("ESPEAK_DATA_PATH"), 1)) return;     // 그 다음 env
if (check_data_path(getenv("HOME"), 0)) return;                 // 그 다음 $HOME
strcpy(path_home, PATH_ESPEAK_DATA);                            // 마지막이 컴파일 기본값
```

→ **`ESPEAK_DATA_PATH`는 위험이 아니다.** 우리가 넘긴 경로가 유효하면 env var도
레지스트리도 `/usr/share`도 우리를 덮어쓸 수 없다. (검색 요약은 env var가 최우선이라고
했는데 소스가 반박한다 — 요약을 믿지 않은 게 맞았다.)

**단 `check_data_path`는 디렉터리인지만 본다** — `phontab` 유무는 확인하지 않는다.
그래서 *존재하지만 비어 있는* 디렉터리는 이 검사를 통과하고 나중에 phontab에서 죽는다.

→ **조치: 우리가 파이썬에서 먼저 검증한다.** `SherpaMatchaTts._check_data_dir`이
디렉터리 존재 + `phontab`/`phonindex`/`phondata`/`intonations` 4개 파일을 확인하고
`FileNotFoundError`를 던진다. 이제 두 실패 모드 모두 **C 레벨 abort가 아니라 파이썬
예외**로 나오고(프로브 재실행으로 확인: `PY-RAISE`), 리눅스에서의 조용한 오발음도
같이 막힌다. 회귀 테스트 9개(`tests/test_tts_data_dir_guard.py`) — 가중치 불필요,
가드가 `__post_init__` 첫 줄이라 모델을 열기 전에 던지기 때문이다.

### 13.5 고지 의무를 지금은 기계적으로 못 지킨다 [1차확인]

- espeak-ng의 `COPYING`은 **GNU GPL v3**이다.
- 우리 `models/sherpa-matcha-en/`에는 `LICENSE`/`COPYING`/`GPL` 파일이 **재귀 탐색으로도
  하나도 없다**. 있는 건 학습 데이터셋(LJSpeech)만 언급하는 251바이트 `README.md`뿐이다.
- 기기 이미지를 배포하는 순간 GPL-3.0은 라이선스 전문 동봉과 소스 제공을 요구한다.
  **지금 상태로는 못 지킨다.** 그리고 §13.1 때문에 **이건 TTS 프리셋 선택과 무관하다** —
  matcha-ko를 버려도 링크된 espeak-ng이 남는다.

### 13.6 이미 알고 있었던 것과 새로 알게 된 것

레포는 이 방향을 부분적으로 알고 있었다 — `tts-conversational-build-design.md:149`가
"eSpeak-ng: 한국어 규칙 약함 + **GPL-3.0** + C 의존성 → 피하고 싶은 이유 정당"이라 쓰고,
`code-review-20260814.md:262`가 "espeak-ng가 한국어 보이스로 `20`을 이상하게 음소화한다"를
기록했다(§7의 한국어 TN 확장기를 만든 이유 중 하나). **의도는 espeak 회피였는데 실제
배포 상태는 espeak 의존이다** — 그 간극이 이번에 확인된 부분이다.

### 13.7 조치

| # | 항목 | 상태 |
|---|---|---|
| 1 | §11·§12의 "Supertonic = espeak 의존 없음(라이선스 이득)" 주장 정정 | 이 절로 대체 |
| 2 | `tts.py`의 Supertonic docstring에 "런타임은 여전히 링크한다" 명시 | 반영 |
| 3 | `models.yaml`의 `data_dir` 주석에 GPL-3.0 + 2.0.0 좌초 위험 명시 | 반영 |
| 4 | sherpa-onnx 버전 락 사유에 #3731 추가 | 반영 |
| 5 | 배포 시 GPL-3.0 고지·소스 제공 | **부분 해결** — `THIRD-PARTY-NOTICES.md` 신규(라이선스 전문 위치 + 대응 소스 위치 명시). 이미지에 무엇을 어떤 형태로 동봉해야 충분한지는 **여전히 법무 판단** |
| 6 | Pi 이미지에서 시스템 `/usr/share/espeak-ng-data` 폴백 확인 | **해결** — 우선순위를 소스로 확정(넘긴 경로가 최우선)하고 `_check_data_dir` 가드로 막음. 실기 불필요해짐 |
| 7 | 2.0.0 대비 한국어 `lexicon.txt` 경로 조사 | **조사 완료, 착수는 2.0.0 이후** — §13.8 |

### 13.8 2.0.0 마이그레이션 경로 — 조사 완료 (2026-08-19)

`OfflineTtsMatchaModelConfig`에 **`lexicon` 필드가 1.13.4에 이미 있다** [1차확인, 도입부
introspection]. 처음엔 그래서 "지금도 espeak 없이 갈 수 있다"고 봤는데, **설정에 필드가
있는 것과 모델이 그걸 쓰는 것은 별개다.**

**프론트엔드는 설정이 아니라 모델의 ONNX 메타데이터로 선택된다**
[1차확인, `offline-tts-matcha-impl.h` `InitFrontend()`]:

| 메타데이터 | 프론트엔드 | 쓰는 것 |
|---|---|---|
| `is_zh_en` | `MatchaTtsLexicon` | lexicon + data_dir |
| `jieba` | `CharacterLexicon` | **lexicon만** — data_dir 불필요 |
| `has_espeak` | `PiperPhonemizeLexicon` | **data_dir만** — lexicon 무시 |
| 그 외 | — | `SHERPA_ONNX_EXIT(-1)` |

우리 `matcha-ko`의 메타데이터는 `has_espeak: 1`, `jieba: 0` [1차확인, onnxruntime로 직접
읽음] → **`lexicon`을 설정해도 무시된다.** 그리고 `PiperPhonemizeLexicon`은 조건 분기 없이
항상 `CallPhonemizeEspeak`을 부른다 [1차확인].

**2.0.0에서 espeak 분기가 사라지면** `has_espeak` 모델은 마지막 줄로 떨어진다 —
즉 프로세스 종료다. 업스트림 계획 [1차확인, #3731 댓글, csukuangfj 2026-07-15]:

> "We will not delete any files or modify the existing .onnx artifacts within the model
> repository; instead, a lexicon.txt file will be added to every model directory."

**그건 업스트림 모델 레포 이야기다.** `matcha-ko`는 우리 모델이고 그 레포에 없다 →
**lexicon.txt를 우리가 만들어야 한다.**

착수 전 판별해둔 것:

- **입도.** `Lexicon`/`CharacterLexicon`은 입력을 `SplitUtf8`로 **문자 단위**로 쪼개
  조회하고 OOV는 경고 후 **조용히 버린다**(`"OOV %s. Ignore it!"` → `continue`)
  [1차확인, `lexicon.cc`]. 한국어에는 **음절 단위 lexicon이 구조적으로 맞다** —
  한글이 음절 문자이고, 어절 단위로는 교착어라 어휘가 무한하다. 대신 음절 경계를 넘는
  음운 규칙(연음·자음동화: 국물→궁물)을 잃는다.
- **잃을 게 적을 가능성이 높다.** espeak의 한국어는 얇다 — `ko_dict` 47KB
  (`en_dict` 167KB, `cmn_dict` 1.5MB), `lang/ko`는 **51바이트**로 `name`/`language`/
  `pitch`/`intonation` 네 줄뿐이다 [1차확인, 디스크]. 레포가 예전부터 적어둔
  "espeak-ng 한국어 규칙 약함"(`tts-conversational-build-design.md:149`)과
  "`20`을 이상하게 음소화한다"(`code-review-20260814.md:262`)의 정량적 근거다.
- **지금 만드는 게 유리하다.** espeak-ng이 아직 링크돼 있는 동안 음소를 뽑아 lexicon을
  생성해두는 것이 2.0.0 이후보다 쉽다. 단 **2.0.0이 `has_espeak` 모델에 대해 어떤
  lexicon 형식을 기대할지가 아직 확정되지 않았다** — 그게 정해지기 전에 만들면 형식이
  안 맞을 수 있다. **그래서 지금은 만들지 않고, 근거만 확정해둔다.**

**우리 선택지는 셋이고 전부 열려 있다:** (a) `<2`에 머문다(현재), (b) 2.0.0 형식이
확정되면 음절 lexicon을 생성한다, (c) Supertonic으로 간다 — character-level이라 이 문제
자체가 없다. §13.3의 "v3의 진짜 장점은 전방 호환성"이 이 표에서 나온 말이다.

---

## 14. 축 4 완료 — 학술 turn-taking 2025–2026 (2026-08-19)

§6-4가 논문 이름만 적어두고 중단된 축이다. **§2의 "Smart Turn v3에 사실상 대안이 없다"를
검증하는 축이므로, 안 하고 결론을 유지하는 건 근거 없이 유지하는 것이었다.** 했다.

판정 기준은 §2와 같다: **공개 가중치 / 한국어 / CPU 실행 가능 / 라이선스**.

| 후보 | 공개 가중치 | 한국어 | CPU 실행 | 라이선스 | 판정 |
|---|---|---|---|---|---|
| **Easy Turn** (2509.23938) | ✅ [GitHub](https://github.com/ASLP-lab/Easy-Turn) + [HF](https://huggingface.co/ASLP-lab/Easy-Turn), 134★, 2026-01-25 푸시 | ❌ **언급 자체가 없음** | ❌ **850MB, 263ms / 2559MB @ RTX 4090** | ✅ Apache-2.0 | **탈락 — 3중 실격** |
| Phoenix-VAD (2509.20410) | ❌ 없음("requires internal PR approval") | ❌ 영어·중국어 | ❌ Qwen2.5-0.5B, **50ms @ A6000** | — | 탈락 |
| Next-Turn (2606.18094) | ❌ 미기재 | 미기재 | 미기재 | — | **모델은 못 씀. 아이디어는 값짐** |
| Thai EOT (2510.04016) | ❌ "public-ready implementation plan"뿐 | ❌ 태국어 | text-only → 스트리밍 ASR 선행 | — | **레시피가 값짐** |
| MuVAP (2606.16731) | ❌ 미기재 | 미기재 | ❌ **카메라 필요**(face tracks, single camera) | — | 탈락 (구조적) |
| multilingual VAP (2403.06487) | ❌ 미기재 | ❌ **영/중/일. 한국어 명시적 부재** | 미기재 | — | 탈락 |
| FastTurn (2604.01897) — §6에 없던 신규 | ❌ 테스트셋만 공개 | 미기재 | 미기재 | — | 관찰 항목 |

**§2 결론 유지.** 다만 이제 검증된 유지다.

### 14.1 Easy Turn — 유일하게 진짜 후보였고, 숫자가 닫았다

§2의 부정 결과(오픈 EOT 제품이 없다)를 뒤집을 수 있는 유일한 항목이었다. 실제로 공개돼
있고 Apache-2.0이고 활발하다. 실격은 라이선스가 아니라 **규모와 언어**다 [1차확인, README]:

| 모델 | Params(MB) | Latency(ms) | Memory(MB) | ACC_cp | ACC_incp | ACC_bc | ACC_wait |
|---|---|---|---|---|---|---|---|
| Paraformer + TEN Turn Detection | 7220 | 204 | 15419 | 86.67 | 89.3 | – | 91 |
| Smart Turn **V2** | 95 | 27 | 370 | 78.67 | **62** | – | – |
| **Easy Turn** | **850** | **263** | **2559** | 96.33 | 97.67 | 91 | 98 |

**모든 수치가 RTX 4090 기준이다** — 논문 문장: *"All experiments are conducted on a single
NVIDIA RTX 4090 GPU"*. CPU 수치가 아예 없다.

우리 기준으로 환산하면:
- 우리는 **smart-turn-v3.2 int8 8.7MB**를 쓴다. Easy Turn은 **약 98배**다.
- 메모리 2559MB. CM5에서 LLM 가중치만 1.43GB인데 그 위에 2.5GB를 더 얹을 수 없다.
- 한국어는 README·모델카드 어디에도 없고 학습 데이터 예시가 `"lang": "<CN>"`이다.

→ **채택 불가.** 세 이유가 독립적이라 하나가 풀려도 안 된다.

### 14.2 그런데 이 표에서 우리 스택에 대한 경고가 나온다

**Smart Turn V2의 `ACC_incp` = 62%.** "incomplete"(생각 중간의 멈춤) 판정이 **우리 적응형
endpoint grace가 정확히 의존하는 신호다**(`vad.py:277-293`, `grace_frames_for_prob`).
제3자 측정에서 그 판정이 62%라는 것은 가볍게 볼 수 없다.

단서를 정확히 달아둔다: (a) **그들의 테스트셋**이고 자기 모델에 유리하게 구성됐을 수 있다,
(b) **V2이고 우리는 v3.2**다 — v3에서 23개어로 확장하며 무엇이 바뀌었는지 이 표는 말하지 않는다.
그래도 `configs/turn_detector.yaml`의 `complete_threshold: 0.5`가 아직 스톡값이고
`detector.py:50`이 "not tuned here"라고 인정하는 상황과 겹쳐 읽으면,
**§2가 계획해둔 임계값 스윕의 우선순위가 올라간다.**

### 14.3 가장 값진 수확 — 어노테이션 없이 EOT 라벨을 만드는 방법 두 가지

우리 미해결 항목 중 `labels.json` 라벨링과 "한 글자 네의 의미"는 둘 다 **라벨 데이터가
없어서** 막혀 있다. 두 논문이 그걸 우회하는 방법을 준다.

**(a) Next-Turn — 어노테이션이 아예 필요 없다** [1차확인, 초록]:

> "We propose Next-Turn that uses the time-to-next-speech-onset as the training objective,
> where targets are derived directly from speech timestamps and **require no additional
> annotation**."

결과: *"a 25.9% absolute improvement in endpoint accuracy within 320 ms over the strongest
baseline"*, 그리고 *"gains that increase monotonically with increasing pauses"*.

우리에게 직접 맞는다 — `data/sessions/`에 실사용 턴 오디오가 이미 쌓여 있고 타임스탬프가
있다. **사람이 라벨을 붙이지 않아도 학습 타깃을 만들 수 있다는 뜻이다.**

**(b) Thai EOT — 비영어 언어를 자막으로 부트스트랩한다** [1차확인, 초록]:

> "Using transcribed subtitles from the **YODAS corpus** and Thai-specific linguistic cues
> (e.g., sentence-final particles), we formulate EOT as a binary decision over token
> boundaries. … demonstrates that small, fine-tuned models can deliver near-instant EOT
> decisions suitable for on-device agents."

이건 **우리 문제와 구조가 같다** — 비영어, 종결어미 같은 언어별 단서, 온디바이스 목표.
한국어에도 그대로 옮길 수 있는 레시피다(YODAS는 다국어 자막 코퍼스).

단 한계도 같다: **text-only**라 스트리밍 ASR이 선행돼야 하고, §2가 LiveKit turn-detector를
"연산 두 배"로 평가한 이유가 그대로 적용된다. 우리 chunked-SenseVoice는 최초 커밋이
1.29초라(§7) 그 앞단으로 쓰기엔 느리다.

→ **지금 만들지 않는다.** 하지만 "라벨이 없어서 못 한다"가 더는 정확한 서술이 아니다.
Smart Turn 임계값 스윕이 실제 발화를 필요로 하는데, 그 발화에 **사람 라벨 없이** 타깃을
붙이는 경로가 이제 문서화됐다.

### 14.4 우리 4-state 어휘가 독립적으로 검증됐다

트랙 C-1에서 `TurnVerdict`를 만들 때 **TEN의 3-state가 아니라 4개**로 간 이유를
`verdict.py`에 적어뒀다(왜 두 단계를 합칠 수 없는지, 왜 3개가 아니라 4개인지).

Easy Turn이 예측하는 것 [1차확인, 초록]: *"four dialogue turn states: **complete,
incomplete, backchannel, and wait**"*.

우리 것: `FINISHED / UNFINISHED / WAIT / EMPTY`. backchannel이 우리 `WAIT`이고 그들의
`wait`가 우리 `UNFINISHED`에 가깝다는 매핑 차이는 있지만, **네 상태로 쪼갠 판단 자체가
독립적으로 같은 결론에 도달했다.** 설계 판단에 대한 외부 근거로 기록한다.

### 14.5 VAP 계열은 닫힌다

§6이 "한국어 VAP 체크포인트는 발견되지 않았다"고 예비 관찰로 적었던 것을 확정한다
[1차확인, 2403.06487 초록]:

- 다국어 VAP는 **영어·중국어·일본어**이고 **한국어가 없다**
- 그리고 전이가 안 된다: *"a monolingual VAP model trained on one language does not make
  good predictions when applied to other languages"*

→ 일본어 VAP를 한국어에 쓰는 우회로는 **논문 자신이 반박한다.** MuVAP은 카메라가 필요해
오디오 전용 기기에 구조적으로 안 맞는다. **VAP 계열 전체를 닫는다.**

---

## 15. CM5 — 데이터시트 원문 확인 (2026-08-19)

§10이 "CM4를 사지 말고 CM5로 올려라"로 닫았는데, 캐리어 호환성 근거가 2차 출처뿐이었다.
데이터시트 원문(`RP-008180-DS-7-cm5-datasheet.pdf`, Appendix B)을 읽었다. [전부 1차확인]

> 추출 방법을 남겨둔다 — WebFetch의 PDF 텍스트 추출이 이 문서에서 실패했다(구조 메타데이터만
> 나옴). 저장된 PDF에 `uv run --no-project --with pypdf`로 임시 의존성을 붙여 뽑았다.
> `--no-project`가 필요하다: 프로젝트 디렉터리에서 그냥 `uv run`을 쓰면 uv가 `.venv`를
> 건드리려다 실패한다.

### 15.1 §10의 전제가 확인됐다

> "High-performance SoC. Broadcom **BCM2712 quad-core Cortex-A76** (ARMv8) 64-bit processor
> running at **2.4 GHz**."
> "Memory options. Available with 2 GB, 4 GB, 8 GB, or 16 GB **LPDDR4x-4267** SDRAM with ECC"

§10.1의 논지가 그대로 성립한다 — CM4의 A72는 Armv8.0-A로 **dotprod/i8mm이 없어** llama.cpp의
ARM 양자화 빠른 경로를 전부 놓치는데, **A76은 Armv8.2-A라 dotprod을 갖는다.** 클럭도
1.5GHz → 2.4GHz(1.6배)다. 보드를 CM5로 올리라는 권고의 근거가 1차 출처로 확인됐다.

### 15.2 캐리어 호환성 — 폼팩터는 같고 23핀이 다르다

Appendix B Table 14가 CM4↔CM5 핀 차이를 전수 나열한다. 우리에게 의미 있는 것만:

| 핀 | CM4 | CM5 | 우리 영향 |
|---|---|---|---|
| 16 | SYNC_IN | **Fan_tacho** | 팬 제어로 용도 변경 |
| 19 | Ethernet nLED1 | **Fan_PWM** | 이더넷 LED를 쓰면 재배선 |
| 76 | Reserved | VBAT | RTC 배터리. *"constant load of a few uA even if CM5 is powered"* |
| 92 | RUN_PG | **PWR_Button** | Pi 5식 전원 버튼 동작 |
| 94, 96 | AnalogIP1/0 | **CC1/CC2** | **ADC 두 채널이 사라진다** → USB-C PD 협상용 |
| 99 | Global_EN | PMIC_ENABLE | *"No external change"* |
| 100 | nEXTRST | CAM_GPIO1 | 부팅 중 low로 구동돼 nRESET을 흉내 |
| 104, 106 | Reserved | PCIE_DET_nWAKE / PCIE_PWR_EN | 신규 |
| 111 | VDAC_COMP | **VBUS_EN** | USB 3.0 포트 전원 제어 |
| 128–142 | **CAM0** | **USB 3.0 포트** | 카메라 포트가 사라진다 |
| 157–171 | **DSI0** | **USB 3.0 포트** | 디스플레이 포트가 사라진다 |

그 외 [1차확인, B.1.2]: 커넥터 브랜드 변경, PCB가 0.04mm 두꺼움, PCIe CLK가 더는 용량 결합
아님, **HDMI/SDA/SCL/HPD/CEC의 추가 ESD 보호가 CM5에서 제거됨**, CAM1·DSI1이 겸용이 됨.

### 15.3 우리 판정 — 이건 마이그레이션 비용이 아니라 설계 입력이다

**CM4를 안 사기로 이미 결정했으므로**(§10) 우리에게 기존 캐리어가 없다. 즉 23핀 차이는
"고쳐야 할 것"이 아니라 "처음부터 CM5 핀아웃으로 그리면 되는 것"이다.

오디오 전용 기기라는 점이 유리하다 — 가장 큰 변경인 **CAM0(128–142)과 DSI0(157–171)이
USB 3.0으로 바뀐 것**은 카메라도 DSI 디스플레이도 안 쓰는 우리에게 무해하고, 오히려
**USB 3.0 포트 두 개를 얻는다**(USB 마이크에 쓸 수 있다).

주의할 것 둘:

1. **ADC 두 채널이 사라진다**(핀 94/96 → USB-C PD CC). 아날로그 입력으로 뭔가 재려는
   설계(배터리 전압, 아날로그 마이크 레벨)는 다른 방법이 필요하다.
2. **전력 예산** [1차확인, B.3]: *"Power supply designs should accommodate **5 V at up to
   2.5 A**."* 그리고 완화책까지 적혀 있다 — *"If this creates an issue with an existing board
   design, **lowering the CPU clock rate** can reduce the peak power consumption."*
   우리 워크로드가 4코어를 다 쓰는 LLM 디코드라 피크가 실제로 걸릴 쪽이다.
   §10의 스레드 예산 논의(`runtime.yaml`)가 성능만의 문제가 아니라 **전력 문제이기도 하다.**

트랙 길이 변경(B.2)은 무해하다 — *"remain well within tolerances, so no functional impact
is expected."*

→ **§10의 권고를 유지하고, 근거를 2차 → 1차로 승급한다.** 남은 실기 항목은 보드가 생긴
뒤의 측정뿐이다.
