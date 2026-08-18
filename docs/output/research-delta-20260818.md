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

## 6. 아직 열린 질문 (조사 미완)

세션 한도와 WebSearch 예산(200/200) 소진으로 세 축이 중단됐다. 남은 것:

1. **스트리밍 ASR 대안** — Vosk-ko 스트리밍 CPU RTF·한국어 CER, onnx-asr, FunASR streaming 한국어
2. **한국어 CPU TTS 잔여 후보** — Piper 한국어 음성 유무, StyleTTS2/VITS/MeloTTS 한국어 커뮤니티
   체크포인트, ZipVoice 한국어 추가 여부(현재 중/영만)
3. **배포 아날로그 + per-core 병목** — OpenLive 상세, llama.cpp ARM 양자화(A72에 dotprod/i8mm이
   **없다**는 점의 비용), 크로스턴 KV 캐시, CM5/RK3588 등 SBC 재검토,
   Pi급에서 실제 측정된 E2E 왕복 사례
4. **학술 turn-taking 2025–2026** — Next-Turn(2606.18094), Phoenix-VAD, Easy Turn(2509.23938),
   MuVAP(2606.16731), multilingual VAP(2403.06487). **한국어 VAP 체크포인트는 발견되지 않았다**
5. **부분 가설 불안정성 수치** — Shangguan et al.(2006.01416)의 UPSR 표를 PDF에서 추출 실패.
   **적중률의 핵심 입력인데 미정량.** 우리 sherpa 스트림에서 직접 재는 게 빠르다

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
- **출처·라이선스가 문서화됨** — `sherpa-matcha-ko`는 어떤 setup 스크립트도
  내려받을 수 없는 손수 복사한 커뮤니티 체크포인트다(출처·라이선스 불명)
- **espeak-ng 의존 없음** — G2P 자체가 불필요(character-level)

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
   최초 커밋이 1.29초, 16개 캡처 중 7개는 아무것도 커밋 못 함, 게다가 커밋 내용이 뒤집힐 수 있음.
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
