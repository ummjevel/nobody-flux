# 서드파티 고지 (Third-Party Notices)

> `nobody-flux`가 배포 시 함께 싣는 서드파티 구성요소와 그 라이선스. **이 파일은
> 법률 의견이 아니다** — 확인된 사실과, 아직 확인되지 않은 것을 구분해 적은 목록이다.
> 제품화 전 법무 확인이 필요한 항목은 🚨로 표시했다.
>
> 마지막 갱신 2026-08-19.

## 이 파일이 존재하는 이유

이 프로젝트는 라이선스를 각 설정 파일 주석에 흩어서 기록해왔고, 그 방식으로
**같은 종류의 함정을 일곱 번** 만났다. 전부 "코드 라이선스 ≠ 가중치 라이선스"의 변형이다:

1. EXAONE / Kanana — 벤치마크는 좋은데 NC (`configs/models.yaml`)
2. 양자화 레포의 라이선스를 원본으로 착각 (`models.yaml:89-91`)
3. Supertonic 번들 안의 `LICENSE`가 MIT였지만 **가중치용이 아니라 샘플 코드용** (§7.2)
4. 🚨 **기본 ASR**(SenseVoice)의 가중치가 Apache도 MIT도 아님 (§11.1)
5. Piper 한국어는 존재하지만 non-commercial (§11.4)
6. espeak-ng이 sherpa-onnx 바이너리에 **정적 링크**돼 있어 GPL-3.0이 런타임 전체에 걸림 (§13)
7. 🚨 **기본 VAD**(TEN-VAD)가 Apache-2.0이 아니라 "Apache-2.0 + 추가 조건"(Agora 경쟁 금지) (§2.3)

절 번호는 `docs/output/research-delta-20260818.md`를 가리킨다.

**실측된 현재 상태:** 다운로드되는 모델 번들 15개 중 **라이선스 파일을 싣는 것은 3개**이고,
그 중 하나는 URL 한 줄이며 하나는 가중치에 틀린 라이선스다. 즉 번들에 든 파일만으로는
고지 의무를 지킬 수 없다. 이 파일이 그 간극을 메우는 곳이다.

## 확인 등급

- **[디스크]** — 이 기계의 파일을 직접 읽어 확인
- **[1차]** — 업스트림 원본(HF 모델 카드 LICENSE, 레포 COPYING, 릴리스 노트)에서 확인
- **[레포기록]** — 이 레포가 앞서 확인해 기록한 것 (출처 주석 있음)
- **[미확인]** — 확인되지 않음. 추측하지 않았다

---

## 1. 런타임 / 바이너리

| 구성요소 | 라이선스 | 등급 | 비고 |
|---|---|---|---|
| **sherpa-onnx** 1.13.4 | Apache-2.0 | **[디스크]** `sherpa_onnx-1.13.4.dist-info/licenses/LICENSE` | ⚠️ 아래 espeak-ng 항목 참조. 이 휠은 **Apache-2.0 고지만 싣는다** |
| **espeak-ng** | **GPL-3.0** | **[1차]** [COPYING](https://github.com/espeak-ng/espeak-ng/blob/master/COPYING) | 🚨 아래 §1.1 |
| onnxruntime 1.28.0 | MIT | **[디스크]** dist-info | |
| llama-cpp-python 0.3.34 | MIT | **[디스크]** dist-info | |
| numpy 2.5.2 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | **[디스크]** dist-info | 전부 permissive |
| sounddevice 0.5.5 / loguru 0.7.3 / PyYAML 6.0.3 | MIT | **[디스크]** dist-info | |
| soundfile 0.14.0 | BSD-3-Clause | **[디스크]** dist-info | |
| transformers 4.57.1 / sentencepiece 0.2.2 | Apache-2.0 | **[디스크]** dist-info | |

### 1.1 🚨 espeak-ng — GPL-3.0이 두 경로로 들어온다

**경로 A: sherpa-onnx 바이너리에 정적 링크.** `sherpa-onnx-c-api.dll`(4.5MB)에
`espeak_ng_Initialize` / `espeak_ng_InitializePath` / `espeak_ng_SetVoiceByName` 등
espeak-ng API 전체와 piper-phonemize의 `phonemize_eSpeak` C++ 심볼, 하드코딩된
`/usr/share/espeak-ng-data` 폴백 경로가 들어 있다. **[디스크]**

이 DLL은 TTS만 쓰는 게 아니다 — **SenseVoice ASR과 TEN-VAD가 같은 DLL을 로드한다.**
따라서 **TTS 프리셋을 무엇으로 바꿔도 이 노출은 남는다.**

업스트림도 같은 판정을 내렸다 **[1차]**
([k2-fsa/sherpa-onnx#3731](https://github.com/k2-fsa/sherpa-onnx/issues/3731), 2026-07-08):

> "Since `espeak-ng` is licensed under GPL, it introduces license constraints that are
> incompatible with the Apache-2.0 license of sherpa-onnx. To keep sherpa-onnx fully
> compatible with Apache-2.0 licensing, we will remove the `espeak-ng` dependency."

제거는 **2.0.0** 예정이고 아직 출시되지 않았다(PyPI 최신 1.13.6) **[1차]**.
그래서 `pyproject.toml`이 `sherpa-onnx>=1.13.4,<2`로 상한을 걸고 있다 — 이유는 §13.3.

**경로 B: 컴파일된 음소 데이터.** `models/sherpa-matcha-en/espeak-ng-data/`(18MB)는
설치 스크립트가 내려받으며, `ko_dict` 같은 **컴파일된 사전**과 `phondata` / `phontab` /
`phonindex` / `intonations` / `lang/` / `voices/`로 구성된다. 번들에 라이선스 파일은
**재귀 탐색으로도 없다** **[디스크]**.

**대응 소스(corresponding source):** https://github.com/espeak-ng/espeak-ng
컴파일된 `*_dict` 파일의 소스는 그 레포의 `dictsource/`(한국어는 `ko_rules`, `ko_list`)이고,
`phondata`/`phontab`/`phonindex`는 `phsource/`에서 `espeak-ng --compile`로 생성된다.

**아직 안 된 것:** GPL-3.0은 바이너리 형태로 전달할 때 라이선스 전문 동봉과
대응 소스 제공을 요구한다. 위 URL은 **소스가 어디 있는지**를 적은 것이고, 기기 이미지에
무엇을 어떤 형태로 동봉해야 충분한지는 **법무 판단이 필요하다.** 이 프로젝트는 아직
자체 배포 라이선스조차 정하지 않았다(레포에 `LICENSE` 파일 없음 **[디스크]**).

**우리 쪽 의존 실체:** 기본 TTS(`sherpa-matcha-ko`)는 espeak-ng을 런타임 G2P로 실제로
쓴다 — `tokens.txt`가 espeak IPA 음소 목록(159개)이고 ONNX 메타데이터가
`has_espeak: 1`이다 **[디스크]**. §3에 이걸 떼어내는 경로를 적었다.

---

## 2. 모델 가중치 (기본 파이프라인)

| 스테이지 | 모델 | 라이선스 | 등급 |
|---|---|---|---|
| ASR (기본) | SenseVoice-Small int8 | 🚨 **FunASR Model Open Source License v1.1** — Apache도 MIT도 아니다 | **[디스크]** + [레포기록] §11.1 |
| VAD (기본) | TEN-VAD | 🚨 **Apache-2.0 "with additional conditions"** — Agora 경쟁 금지 조항 | **[1차]** §2.3 |
| VAD (대체) | Silero VAD | **MIT**, 추가 조건 없음 | **[1차]** [LICENSE](https://github.com/snakers4/silero-vad/blob/master/LICENSE) + ONNX 메타데이터 |
| 엔드포인트 | Smart Turn v3.2 | BSD-2-Clause (pipecat-ai) | [레포기록] `setup_common.sh:266` |
| LLM (기본) | Mi:dm 2.0 Mini Instruct Q4_K_M | **MIT** | **[1차]** 원본 `K-intelligence/Midm-2.0-Mini-Instruct` (우리가 받는 건 `mykor/...-gguf` 재양자화본 — 규칙 1대로 원본을 확인) |
| LLM (대체) | Qwen3-0.6B Q4_K_M | 원본 `Qwen/Qwen3-0.6B`은 **Apache-2.0**. 우리가 받는 `bartowski/Qwen_Qwen3-0.6B-GGUF`는 **라이선스를 선언하지 않고 LICENSE 파일도 없다**(`base_model`만 명시) | **[1차]** 둘 다 확인. 파생본 선언은 **[확인 불가]** — §2.5 |
| TTS (기본) | `sherpa-matcha-ko` | **자체 소유** — 직접 학습. 음성 디자인·코퍼스를 Qwen3-TTS로 생성, 학습 이용 가부는 프로젝트 오너 확인 완료(2026-08-18) | [레포기록] §12.1 |
| TTS 보코더 | `vocos-22khz-univ` | **Apache-2.0** | **[1차]** §2.4 — ONNX 메타데이터가 출처를 들고 있다 |
| TTS (비교) | Supertonic 3 / 2 | **OpenRAIL-M** (use-restriction 있음) | [1차] [HF LICENSE](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE) |
| TTS (영어 참조) | matcha-icefall-en_US-ljspeech | **가중치 라이선스 명시 없음.** 학습 코드(icefall)는 Apache-2.0, 데이터는 LJSpeech | **[1차]** icefall 레포 + sherpa 문서. 가중치 자체는 **[확인 불가]** |
| ASR (스트리밍, 비활성) | streaming-zipformer-ko | **라이선스 태그 없음.** 학습 데이터 KsponSpeech(AI Hub/NIA)는 신청·승인 필요 | [레포기록] §11.8. 상업 이용·재배포 조건 **[확인 불가]** |

### 2.1 SenseVoice — 기본값인데 가장 불확실하다 🚨

`models/sense-voice/LICENSE`의 **전체 내용** **[디스크]**:

```
Ref to https://github.com/modelscope/FunASR?tab=readme-ov-file#license
```

라이선스 파일이 아니라 링크 한 줄이다. HF 카드는 `license: other`,
`license_name: model-license` → FunASR Model Open Source License v1.1(Alibaba).
**코드(FunASR)가 MIT인 것과 가중치는 별개다.**

이건 비교용 프리셋이 아니라 `configs/models.yaml`의 **기본 ASR**이다. Supertonic의
OpenRAIL-M은 신중히 다뤘는데 이미 기본값으로 쓰고 있는 스테이지는 확인된 적이 없었다.

### 2.2 Supertonic — 번들 안의 LICENSE를 믿으면 안 된다

`models/sherpa-supertonic-3/LICENSE`와 `models/sherpa-supertonic-2/LICENSE`는
**바이트 단위로 동일**하고, 둘 다 `MIT License / Copyright (c) 2025 Supertone Inc.`
20줄이다 **[디스크]**. 이건 Supertone의 **샘플 코드** 라이선스이고 **가중치용이 아니다.**
가중치는 업스트림 HF의 OpenRAIL-M이다 — 확인 URL은 위 표에 있고
`configs/models.yaml:282-288`에도 기록돼 있다.

### 2.3 🚨 TEN-VAD — 우리 **기본 VAD**가 순수 Apache-2.0이 아니다

`ten-vad.onnx`의 ONNX 메타데이터가 라이선스 위치를 직접 들고 있다 **[디스크]**:
`license: https://github.com/TEN-framework/ten-vad/blob/main/LICENSE`.
그 파일의 실제 내용 **[1차]**:

> "The ten-vad is licensed pursuant to the Apache License v2.0 … **with the following
> additional conditions**."
>
> "You may not Deploy the ten-vad in a way that **competes with Agora's offerings**
> and/or that allows others to compete with Agora's offerings"

Apache-2.0 **파생**이지 Apache-2.0이 아니다. 파생물도 같은 조건을 유지해야 하고,
BSD-3-Clause·BSD-2-Clause 코드를 포함한다(레포 `NOTICES`).

**왜 중요한가:** Agora는 실시간 음성·영상 인프라와 대화형 AI를 판다. 우리가 만드는 것이
"Agora의 제공물과 경쟁"하는지는 제품 형태에 달린 판단이고, **VAD는 갈아끼울 수 있는
비교용 프리셋이 아니라 파이프라인 첫 스테이지다.** §2.1의 SenseVoice와 같은 계열 —
기본값으로 쓰는 스테이지의 라이선스를 확인한 적이 없었다. **제품화 전 법무 확인.**

**곁들여 나온 것 (라이선스와 무관하지만 중요):** 같은 메타데이터의
`comment: "It uses 0 as the pitch feature, which may degrade the performance."`
→ sherpa가 export할 때 **pitch feature를 버렸다.** 우리 기본 VAD는 **의도적으로 열화된
빌드**다. VAD 임계값 튜닝(`_calibrate_vad_threshold.py`)의 상한이 여기서 정해진다.

**조치 (2026-08-19):** 기본값을 바꾸지 않았다. 대신 **VAD 엔진을 설정값으로** 만들었다
(`configs/vad.yaml`의 `engine`) — `ten-vad` | `silero-vad`. 이유는 이 판단이 코드 변경이
아니라 사람의 결정이어야 하기 때문이고, 결정이 어느 쪽으로 나든 그때 급하게 코드를 고치는
상황을 피하기 위해서다. Silero 모델은 setup 스크립트가 기본으로 받는다(629KB).

**Silero로 바꿀 때 잃는 것:** `ten-vad` 블록의 `threshold: 0.5`는 이 방 마이크에 대한
**실측값**이고 다른 모델로 이전되지 않는다. 그래서 threshold를 공유 키가 아니라 **엔진
블록 안에** 뒀다. 실패가 비대칭이라는 점이 중요하다 — 임계값이 낮으면 모델이 침묵을 발화로
보고해 **턴이 영원히 끝나지 않는다**(이 레포가 실제로 겪은 실패). 재캘리브레이션이
`_calibrate_vad_threshold.py`이고, 그 스크립트도 이제 yaml의 `engine`을 읽는다 —
안 그러면 silero를 설정해두고 ten-vad를 캘리브레이션해서 그 수치를 보고했을 것이다.

**받은 Silero 체크포인트의 정체** [디스크, ONNX 메타데이터]:
`comment: "This is silero-vad v4 exported to onnx by k2-fsa. Only the 16kHz branch is kept"`
→ **v4**이고 16kHz 전용이다(우리 `SAMPLE_RATE`가 16000이라 문제없음). 입력이 `x: [1, 512]`로
**window_size 512가 그래프에 박혀 있다**(TEN-VAD는 256). 그래서 `window_size`를 두 엔진이
공유하지 않고 각자 sherpa 기본값을 쓰게 뒀다. v5를 원하면 같은 릴리스에
`silero_vad_v5.onnx`(2.2MB)가 있다.

### 2.4 vocos 보코더 — 문서에 없고 파일에 있었다

sherpa-onnx 문서, 릴리스 노트, export PR([#2012](https://github.com/k2-fsa/sherpa-onnx/pull/2012))
어디에도 상류 체크포인트 출처가 없다 **[1차, 전부 확인]**. 그런데 ONNX 파일 자체의
메타데이터에 있었다 **[디스크]**:

```
model_author:    BSC-LT
model_filename:  mel_spec_22khz_univ.onnx
url1:            https://huggingface.co/BSC-LT/vocos-mel-22khz
url2:            https://github.com/gemelo-ai/vocos
```

`BSC-LT/vocos-mel-22khz`의 HF 선언은 **Apache-2.0** **[1차]**.

> **방법 메모:** sherpa-onnx 재배포 모델의 출처를 찾을 때 **ONNX 메타데이터를 먼저 읽어라.**
> 이번에 `vocos`(출처)와 `ten-vad`(라이선스 URL + 열화 경고), `matcha-ko`(`has_espeak`)
> 세 건이 전부 파일 안에서 나왔다. 문서보다 파일이 정확했다.

### 2.5 재양자화 레포는 라이선스를 선언하지 않을 수 있다

규칙 1은 "양자화 레포에서 라이선스를 가져오지 말고 원본을 봐라"인데, 이번에 그 변형을
만났다 — **가져올 것이 아예 없는 경우.**

| 우리가 받는 것 | 파생본 선언 | 원본 | 원본 선언 |
|---|---|---|---|
| `mykor/Midm-2.0-Mini-Instruct-gguf` | **MIT** (+ `LICENSE.txt` 있음) [1차] | `K-intelligence/Midm-2.0-Mini-Instruct` | **MIT** [1차] |
| `bartowski/Qwen_Qwen3-0.6B-GGUF` | **없음** (LICENSE 파일도 없음, `base_model`만) | `Qwen/Qwen3-0.6B` | **Apache-2.0** [1차] |

Mi:dm 쪽은 파생본도 원본과 같은 MIT를 선언하고 LICENSE 파일까지 싣는다 — 이게 정상이다.
Qwen3 쪽이 비어 있다. 원본 라이선스가 파생·재배포를 허용하므로 실질적으로는 그게 따라온다고
보는 게 자연스럽지만, **파생 레포가 아무 선언도 하지 않았다는 사실 자체는 기록해둔다.** 고지문에 무엇을 적을지는
원본을 기준으로 하되, 실제로 배포하는 바이트는 파생본이라는 점이 다르다.

---

## 3. espeak-ng 의존을 떼어내는 경로 (조사 완료, 미실행)

sherpa-onnx가 Matcha의 텍스트 프론트엔드를 **설정이 아니라 모델의 ONNX 메타데이터로**
고른다 **[1차, `offline-tts-matcha-impl.h` `InitFrontend()`]**:

| 메타데이터 | 프론트엔드 | 쓰는 것 |
|---|---|---|
| `is_zh_en` | `MatchaTtsLexicon` | lexicon + data_dir |
| `jieba` | `CharacterLexicon` | **lexicon만** (data_dir 불필요) |
| `has_espeak` | `PiperPhonemizeLexicon` | **data_dir만** (lexicon 무시) |
| 그 외 | — | **`SHERPA_ONNX_EXIT(-1)`** — 프로세스 종료 |

우리 `matcha-ko`는 `has_espeak: 1`, `jieba: 0`이므로 **`lexicon` 필드를 설정해도 무시된다.**
즉 `OfflineTtsMatchaModelConfig`에 `lexicon`이 있다는 것만으로는 마이그레이션이 안 된다.

2.0.0에서 espeak 분기가 사라지면 `has_espeak` 모델은 위 표의 마지막 줄로 떨어진다.
업스트림 계획은 이렇다 **[1차, #3731 댓글, csukuangfj]**:

> "We will not delete any files or modify the existing .onnx artifacts within the model
> repository; instead, a lexicon.txt file will be added to every model directory."

**그런데 그건 업스트림 모델 레포 이야기다.** `matcha-ko`는 우리 모델이고 거기 없으므로
**우리가 직접 lexicon.txt를 만들어야 한다.**

착수 전 남은 판단(형식이 2.0.0에서 확정되기 전에는 결론 불가):

- **입도(granularity).** `Lexicon`/`CharacterLexicon`은 입력을 UTF-8 **문자 단위**로 쪼개
  조회하고, OOV는 경고 후 **조용히 버린다** **[1차, `lexicon.cc`]**. 한국어에는
  음절 단위 lexicon이 구조적으로 맞다(한글은 음절 문자). 대신 음절 경계를 넘는
  음운 규칙(연음·자음동화: 국물→궁물)을 잃는다.
- **잃을 게 얼마나 되는가.** espeak의 한국어는 얇다 — `ko_dict` 47KB
  (영어 167KB, 중국어 1.5MB), `lang/ko`는 51바이트로 `name`/`pitch`/`intonation`뿐
  **[디스크]**. 애초에 espeak이 하는 일이 적다면 잃을 것도 적다.
- **지금 만들 수 있다는 게 중요하다.** espeak-ng이 아직 손에 있는 동안 음소를 뽑아
  lexicon을 생성해두는 것이 2.0.0 이후보다 쉽다.

---

## 4. 갱신 규칙

이 프로젝트가 일곱 번의 함정에서 얻은 규칙:

1. **양자화 레포나 기억에서 라이선스를 가져오지 않는다.** 원본 모델 카드를 읽는다.
2. **번들 안의 `LICENSE`가 가중치 라이선스라고 가정하지 않는다.** 샘플 코드용일 수 있다.
3. **코드 라이선스와 가중치 라이선스를 따로 적는다.** 같은 프로젝트에서도 다르다.
4. **HF 태그도 1차 출처가 아니다** (§9.5) — 파생 모델이 상류 태그를 물려받는다.
5. **ONNX 메타데이터를 먼저 읽는다** (§2.4) — 재배포 모델의 출처·라이선스·열화 경고가
   문서에는 없고 파일에는 있는 경우가 실제로 세 번 있었다.
6. 프리셋을 추가할 때 이 파일에 행을 추가한다. **[미확인]로 적는 것은 허용되고,
   추측해서 적는 것은 안 된다.**
