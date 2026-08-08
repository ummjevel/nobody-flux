# TTS 표현력(paralinguistic) 설계 문서

**상태: 설계/조사.** 이 문서는 "대화다운 TTS"(한숨·숨소리·웃음 같은 비언어 발성 =
non-verbal vocalization, NVV)를 **온디바이스(CM4, GPU 없음) 전용**이라는 이 저장소의 목적
안에서 어떻게 확보할지에 대한 전략 문서다. **핵심 방향(제약 반영)**: 표현력 있는 대형 GPU TTS
(CosyVoice2 등)를 디바이스에 올리는 게 아니라, 디바이스엔 경량 CPU 모델(현재 `sherpa-matcha-ko`,
후보 Piper/Kokoro-ONNX/FreyaTTS)을 유지하고, 거기에 NVV를 넣기 위해 한국어 NVV 데이터
파이프라인(Path A)을 만든다. 대형 표현 모델은 디바이스가 아니라 **서버측 데이터 생성
teacher/레퍼런스**로만 쓴다(아래 "지배적 제약"). CosyVoice2 한국어 NVV 실측은 서버(H100)
작업으로 보류. `configs/models.yaml`의 TTS 프리셋 축을 그대로 쓰므로 디바이스 모델을 갈아끼워도
나머지 파이프라인은 안 바뀐다.

## 문제 정의

- **현재 TTS 상태** (프로젝트 외부, todak-vox/FreyaTTS 쪽 맥락):
  - Qwen3-TTS로 음성을 벌크 생성해 학습시켰더니 화자 5명이 전부 **딱딱한 낭독체**가 됐다.
  - FreyaTTS는 톤·쉼은 자연스럽지만 **발음이 약하고** 추가 학습이 필요한 상태. 아직 사용
    수준 아님.
- **목표**: 실제 대화처럼 들리려면 한숨, 숨 들이쉬기, 웃음, 주저("음...") 같은 NVV가
  필요한데, 이걸 어떻게 확보할지가 핵심 고민.

## 지배적 제약: 온디바이스 전용, GPU 없음

**이 저장소(nobody-flux)는 CM4급 온디바이스 CPU 전용 타깃이다.** 이게 TTS 선택을 지배한다.
디바이스에 실제로 올릴 TTS는 **CPU에서 실시간(또는 그에 준하게) 도는 경량 모델**이어야 한다
— 지금 기본값 `sherpa-matcha-ko`(ONNX, in-process, GPU 불필요)가 바로 그 이유로 선택됐다.
FreyaTTS/MOSS-TTS-Nano는 GPU 지향이라 비교용 프리셋으로만 남아있다.

따라서 표현력 있는 대형 신경망 TTS(CosyVoice2/3, Sesame CSM, Dia 등 — torch/GPU/스트리밍
지향, 0.5B~1B+)는 **디바이스 TTS 후보가 아니다.** CPU에서 너무 느리고 무겁다. 이들의 역할은
둘 중 하나로만 한정한다:
1. **품질 상한 레퍼런스** — "표현력 있는 한국어 TTS가 어디까지 가능한가"를 서버에서 확인.
2. **데이터 생성 teacher(서버측)** — 아래 근본 원인 참고. 표현력 있는 teacher로 NVV 풍부한
   한국어 학습 데이터를 서버에서 생성하고, 그걸로 **경량 CPU 학생 모델**을 학습(증류).
   teacher는 GPU/서버에서 돌고, 디바이스에 나가는 건 어디까지나 작은 CPU 모델.

즉 디바이스 TTS 축(Piper/Matcha/VITS/Kokoro-ONNX급 경량 CPU 모델)과 표현력 획득 수단(서버측
teacher/데이터)을 분리해서 봐야 한다.

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
4. **학습 대상 = 경량 CPU 모델** (위 "지배적 제약"): 디바이스에 나가는 학생은 반드시
   Piper/Matcha/VITS급 CPU-경량 TTS. 표현력 있는 대형 모델을 디바이스에 올리는 게 아니라,
   그런 모델이 만든/주석한 데이터로 작은 모델을 학습한다.
   - **FreyaTTS** 계속 — 이미 경량 방향이고 자연스러운 톤/쉼(어려운 부분)은 확보. 진짜
     한국어 음성으로 학습하니 **발음 문제도 동시 해결**. NVV는 위 인라인 토큰으로 얹음.
   - teacher는 서버측: 표현력 있는 모델(CosyVoice3/CSM, GPU)로 NVV 풍부한 한국어 데이터를
     **생성**하거나, 실제 음성에 NVV를 **자동 주석(NVASR)**. teacher는 절대 디바이스에 안 나감.

효과: 발음(진짜 한국어) + 표현력(진짜 NVV)을, **CPU에서 도는 작은 모델**에. 무겁지만 어느
최종 모델을 고르든 데이터 파이프라인은 공용 자산.

## Path B — 표현력 있는 대형 모델: 디바이스용 아님, 서버측 레퍼런스/teacher로만

**중요(제약 반영)**: 아래 모델들은 GPU 지향이라 **CM4 no-GPU 디바이스 TTS 후보가 아니다.**
CosyVoice2도 이 로컬 박스(RTX 5090 sm_120 vs torch cu121)에선 GPU가 안 먹고 CPU에서 느리다 —
디바이스는 더 열악하다. 그래서 이들은 (1) 품질 상한 레퍼런스, (2) 서버측 데이터 생성 teacher
로만 본다. 디바이스 TTS 프리셋으로 넣지 않는다.

| 모델 | NVV | 한국어 | 크기/성격 | 이 프로젝트에서의 역할 |
|---|---|---|---|---|
| **CosyVoice2-0.5B** | `[laughter]`/`[breath]` 태그 | zero-shot Korean | 0.5B, GPU/스트리밍 | 서버측 teacher/레퍼런스 (디바이스X) |
| **CosyVoice3** | 태그 + "웃으면서 말하기"·강조 | zero-shot Korean | GPU 지향 | 상동, 더 성숙 |
| **Sesame CSM 1B** | 문맥적 숨·웃음 | 주로 영어, 파인튜닝 필요 | 1B, GPU | 상동, 한국어=데이터 필요 |
| **Dia2 / Chatterbox** | 웃음·기침·감정강조 | 영어 중심 | GPU | 참고만 |

**남은 실증 질문(서버측)**: CosyVoice2/3의 `[laughter]`/`[breath]` 태그가 **한국어 문맥에서도
실제로 먹히는가?** (태그는 주로 중/영으로 학습됨). 되면 → 디바이스 배포용은 아니지만, Path A의
**데이터 생성 teacher**로 쓸 수 있음(NVV 풍부한 한국어 학습셋 생성). 안 되면 → teacher 후보에서
제외. 어느 쪽이든 이 실측은 **서버(H100)에서** 하는 게 맞다(로컬 RTX 5090 sm_120 불가). 이
저장소(디바이스 전용)엔 프리셋으로 넣지 않는다.

## 평가

- **NVBench / NVV-SuperBench**(arXiv:2604.16211): NVV 45종 taxonomy, controllability·placement·
  perceptual salience 다축 평가 — "NVV가 실제로 자연스럽게 들리는가/의도한 위치에 오는가"
  측정용. 한국어엔 그대로 안 맞을 수 있으나 평가 축(제어성/배치/현저성)은 차용 가능.
- 당장은 사람 귀로 판정(합성 샘플 몇 개에 `[웃음]`/`[한숨]` 넣어보고 자연스러운지) + 이
  프로젝트 `scripts/benchmark.py`로 latency 비교.

## 결정 & 시퀀싱 (온디바이스 제약 반영)

디바이스 TTS는 경량 CPU 모델(현재 `sherpa-matcha-ko`) 축을 유지한다. 표현력은 "디바이스에
큰 모델을 올리는" 문제가 아니라 "작은 CPU 모델에 어떻게 NVV를 넣느냐" 문제로 본다.

구체적인 모델 지형·표현력 기법·아키텍처·데이터 전략과 "FreyaTTS에 얹기 vs 새로 만들기" 판단은
후속 리서치 문서 `docs/tts-small-expressive-research.md` 참고. 요지: **FreyaTTS(flow-matching)에
얹는 게 최저 위험 정답**(이미 자연스러운 운율 확보 + flow field 위 표현력은 가산적 + CPU/ONNX/
스트리밍 친화). 레시피 = 인라인 NVV 토큰(NVSpeech 패턴) + 경량 조건화(GST + FastSpeech2 variance
adaptor) + CosyVoice3 teacher로 데이터 생성 + NVV-ASR 부트스트랩 + 커리큘럼 학습, 발음은 G2P로
별개 트랙. StyleTTS2 신규 구축은 학습 불안정·스트리밍 없음으로 비권장.

1. **보류(현재)**: CosyVoice2 한국어 NVV 실측은 서버(H100) 작업으로 미룸 — 로컬 RTX 5090
   불가 + 이 저장소(디바이스 전용)엔 대형 GPU TTS를 프리셋으로 넣지 않기로. (실측 시엔 서버에서
   `git clone --recursive FunAudioLLM/CosyVoice` → `inference_cross_lingual('...[laughter]...',
   prompt_16k)` → 24kHz 출력; wetext 사용이라 pynini 불필요, tensorrt/deepspeed는 추론에 불필요.)
2. **디바이스 TTS 후보 정리(no-GPU)**: 경량 CPU TTS 안에서 품질/표현력 비교 — 현재
   `sherpa-matcha-ko`, 그리고 후보로 Piper(초경량), Kokoro-82M-ONNX(품질↑, CPU 가능), FreyaTTS
   (톤 자연스러움). NVV는 아직 이들 중 누구도 기본 지원 안 함 → Path A 데이터가 있어야 얹힘.
3. **Path A(정공법, 온디바이스에 맞음)**: 자발적 한국어 음성 소싱 → seed 주석 → 한국어
   NVASR(Beyond Words 저자원 전략)로 스케일 → **경량 CPU TTS(FreyaTTS/Matcha급)** 를 NVV 인라인
   토큰으로 학습. 필요하면 서버측 표현력 teacher(CosyVoice3/CSM)로 데이터 생성만 거들되,
   나가는 모델은 어디까지나 작은 CPU 모델. Qwen3-TTS 증류는 접는다.

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
