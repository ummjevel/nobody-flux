# TTS 표현력(paralinguistic) 설계 문서

**상태: 설계/조사 + 프로토타입 진행 중.** 이 문서는 "대화다운 TTS"(한숨·숨소리·웃음 같은
비언어 발성 = non-verbal vocalization, NVV)를 어떻게 확보할지에 대한 전략 문서다. 결정:
단기적으로 CosyVoice2를 프리셋으로 붙여 실측(아래 "Path B"), 중장기적으로 한국어 NVV 데이터
파이프라인(아래 "Path A")을 구축해 갈아끼운다. `configs/models.yaml`의 TTS 프리셋 축(같은
"config로 스왑" 구조)을 그대로 활용하므로, 어느 모델을 쓰든 나머지 파이프라인은 안 바뀐다.

## 문제 정의

- **현재 TTS 상태** (프로젝트 외부, todak-vox/FreyaTTS 쪽 맥락):
  - Qwen3-TTS로 음성을 벌크 생성해 학습시켰더니 화자 5명이 전부 **딱딱한 낭독체**가 됐다.
  - FreyaTTS는 톤·쉼은 자연스럽지만 **발음이 약하고** 추가 학습이 필요한 상태. 아직 사용
    수준 아님.
- **목표**: 실제 대화처럼 들리려면 한숨, 숨 들이쉬기, 웃음, 주저("음...") 같은 NVV가
  필요한데, 이걸 어떻게 확보할지가 핵심 고민.

## 근본 원인 (문헌으로 확인됨)

증류(distillation)로는 표현력을 만들 수 없다. UltraVoice 논문(arXiv:2510.22588)이 정확히
지적한다 — *"텍스트 대화에 TTS를 그냥 씌우면 언어적으론 맞지만 표현적으로는 빈곤해진다
(expressively impoverished)"*. **teacher(Qwen3-TTS)가 안 내는 한숨·웃음은 student가 물려받을
수 없다.** 화자 5명이 다 딱딱한 건 화자 문제가 아니라 teacher의 표현력 천장을 그대로 복제한
결과다. → **Qwen3-TTS 벌크 증류로 표현력을 얻으려는 시도는 구조적으로 막다른 길.** 굳이
합성으로 스케일하려면 표현력 있는 teacher(NVV 태그를 내는 CosyVoice3, 문맥적 NVV를 내는
Sesame CSM)로 증류해야 NVV가 전파된다.

주의: UltraVoice 자체는 NVV 데이터가 **아니다**. emotion/speed/volume/accent 같은 *스타일*
제어 데이터이고, 그것도 전부 TTS 합성본(GPT-4o-audio, CosyVoice, Edge TTS+VC)이다 — 한숨·숨·
웃음은 없다. "UltraVoice 데이터처럼 NVV가 있는 것"이라는 통념은 사실과 다르다.

## 한국어 NVV 데이터 공백 (확인됨)

| | 있음 | 비고 |
|---|---|---|
| 한국어 감정 음성 | KESDy18(ETRI, 성우 30명×4감정), AI Hub 감정셋 | **연기된** 감정·낭독체. 감정엔 쓰나 자발적 NVV 없음. "카테고리 적고 부정 편중" 지적 |
| NVV 토큰 주석 코퍼스 | ❌ 한국어엔 없음 | 전부 중국어(Emilia-NV, NVSpeech170k, MNV-17) / 영어(NonverbalTTS) |

핵심: **연기 감정 음성으론 자연스러운 NVV가 안 나온다.** 진짜 한숨·웃음은 자발적
(spontaneous) 음성(예능·팟캐스트·잡담)에 있다. 한국어 NVV는 **초저자원(ultra low-resource)**.

## Path A — 한국어 NVV 데이터 파이프라인 + 자체 학습 (중장기, 정공법)

이제 막연한 게 아니라 문헌 뒷받침 레시피가 있다:

1. **표현**: NVV를 인라인 한국어 토큰으로 (`[웃음]`, `[한숨]`, `[숨]`) — Emilia-NV
   (nvspeech170k) / Beyond Words(arXiv:2607.01563)의 `[Laughter]`/`[Breathing]` 관례. 학습·
   추론 일관.
2. **구축(2단계, Emilia-NV식)**: ① NVV 풍부한 자발적 한국어 음성 소싱 + 소량 seed 인간
   주석(단어 단위 NVV 라벨) → ② 그걸로 **NVV 인식 ASR(NVASR)** 학습해 대량 미라벨 음성을
   자동 라벨링·스케일.
3. **저자원 부트스트랩(Beyond Words의 핵심 기여)** — 한국어 NVV가 초저자원이라 필수:
   - **2단계 커리큘럼**: 모든 NVV를 일단 generic 토큰 하나로 → 카테고리별 파인튜닝
   - **토큰 간 전이**: 고자원(웃음·숨) → 희귀(울음·기침)
   - **voice-conversion 증강 + 클래스 밸런싱**
4. **학습**: 위 데이터로 TTS 학습/파인튜닝. 대상 후보:
   - **FreyaTTS** 계속 — 자연스러운 톤/쉼(어려운 부분)은 이미 확보, 진짜 한국어 음성으로
     학습하니 **발음 문제도 동시 해결**. NVV는 위 토큰으로 얹음.
   - 또는 **CosyVoice/CSM 파인튜닝** — 이미 표현력 있는 베이스에 한국어 데이터 주입.

효과: 발음(진짜 한국어) + 표현력(진짜 NVV)을 한 번에. 무겁지만 어느 최종 모델을 고르든
데이터 파이프라인은 공용 자산.

## Path B — 이미 표현력 있는 모델 갖다 쓰기 (단기, 지금 실측)

| 모델 | NVV | 한국어 | 라이선스/크기 | 비고 |
|---|---|---|---|---|
| **CosyVoice2-0.5B** | `[laughter]`/`[breath]` 태그 | zero-shot Korean | Apache-2.0, 0.5B, 스트리밍 | 지금 프리셋화해서 실측할 대상 |
| **CosyVoice3** | 태그 + "웃으면서 말하기"·강조, 5000h instruction | zero-shot Korean | 작음 | 더 성숙, 공개 범위 확인 필요 |
| **Sesame CSM 1B** | 문맥적 숨·웃음(태그 아님) | 주로 영어, **파인튜닝 필요** | 오픈, 1B, Mimi codec | 가장 대화체답지만 한국어=데이터 필요 → Path A와 합류 |
| **Dia2 / Chatterbox** | 웃음·기침·감정강조 | 영어 중심 | 오픈 | 참고 |

**좁혀진 실증 질문 하나**: CosyVoice2/3의 `[laughter]`/`[breath]` 태그가 **한국어 문맥에서도
실제로 먹히는가?** (태그는 주로 중/영으로 학습됨). 되면 → 학습 없이 표현력 있는 한국어 TTS를
확보(당장 "쓸 수준" 문턱 판별). 안 되면 → Path A로.

## 평가

- **NVBench / NVV-SuperBench**(arXiv:2604.16211): NVV 45종 taxonomy, controllability·placement·
  perceptual salience 다축 평가 — "NVV가 실제로 자연스럽게 들리는가/의도한 위치에 오는가"
  측정용. 한국어엔 그대로 안 맞을 수 있으나 평가 축(제어성/배치/현저성)은 차용 가능.
- 당장은 사람 귀로 판정(합성 샘플 몇 개에 `[웃음]`/`[한숨]` 넣어보고 자연스러운지) + 이
  프로젝트 `scripts/benchmark.py`로 latency 비교.

## 결정 & 시퀀싱

1. **지금**: CosyVoice2-0.5B를 TTS 프리셋으로 붙여 한국어 발음 + NVV 태그 실측(Path B). 이게
   이 문서의 모든 가설을 가장 싸게 판별한다. (task: CosyVoice2 프로토타입)
2. **CosyVoice2가 쓸 수준이면**: 프로젝트 기본 TTS 후보로 두고, Path A는 "더 자연스러운
   자체 모델"을 위한 병행 트랙으로.
3. **부족하면**: Path A 착수 — 자발적 한국어 음성 소싱 → seed 주석 → 한국어 NVASR(Beyond
   Words 저자원 전략) → FreyaTTS(또는 CosyVoice/CSM 파인튜닝) 학습. 어느 경우든 Qwen3-TTS
   증류는 접는다.

## 참고 문헌

- UltraVoice (arXiv:2510.22588) — 증류 stiffness 진단 / 스타일 제어 합성 데이터
- Emilia-NV · NVSpeech170k (nvspeech170k.github.io) — 중국어 NVV 코퍼스 + 2단계 구축 파이프라인
- NonverbalTTS (arXiv:2507.13155) — 영어 NVV 코퍼스
- Beyond Words: NVV in ASR (arXiv:2607.01563) — 인라인 NVV 토큰 + 저자원 부트스트랩 전략
- MNV-17 (arXiv:2509.18196) — 중국어 performative NVV
- CosyVoice3 (arXiv:2505.17589) · CosyVoice2-0.5B (HF: FunAudioLLM/CosyVoice2-0.5B) — 태그 제어 NVV + 한국어 zero-shot
- Sesame CSM-1B (HF: sesame/csm-1b) · Speechmatics 파인튜닝 가이드 — 문맥적 NVV, 파인튜닝
- KESDy18 (ETRI) — 한국어 연기 감정 음성
- NVBench / NVV-SuperBench (arXiv:2604.16211) — NVV 평가 벤치마크
- Toward Natural Emotional TTS w/ Fine-Grained NVV Control (arXiv:2605.25504)
