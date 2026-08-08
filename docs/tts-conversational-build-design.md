# 대화체 표현 소형 TTS 자체 구축 설계

`docs/tts-small-expressive-research.md`(모델·기법 서베이)의 후속. 여기선 좁은 질문 하나에
답한다: **"온디바이스(CM4, CPU)에 적합하면서 대화체로 표현력 좋은 TTS를 직접 만든다면, 어떤
아키텍처·기법·데이터로 가야 하나?"** — Matcha류의 단정적 운율이 대화에 부적합하다는 판단에서
출발.

## 1. 왜 Matcha류가 "단정적"인가 (구조적 진단)

밋밋함은 취향 문제가 아니라 학습 목표의 결과다.

- **Over-smoothing**: NAR TTS가 MAE/MSE 회귀로 mel을 예측하면, 같은 문장의 **여러 자연스러운
  읽기(다봉분포)** 를 하나의 **평균**으로 뭉갠다 → 흐릿하고 단정적. (Revisiting Over-Smoothness
  in TTS, arXiv:2202.13066)
- **결정론적 duration/pitch**: FastSpeech2식 variance adaptor나 Matcha의 결정론적 정렬은 리듬을
  균일하게 만든다. "사람이 상황마다 다르게 말하는" 변동성을 못 낸다.

→ 즉 "단정적 운율"은 **(a) 회귀 목표 + (b) 결정론적 운율 + (c) 낭독체 데이터** 세 개가 겹친
결과. 대화체는 이 셋을 다 뒤집어야 한다.

**중요한 뉘앙스 — flat = flow-matching 탓이 아니다.** FreyaTTS도 flow-matching이지만 밋밋하지
않다(표현력 있는 Qwen3-TTS에서 증류돼 자연스러운 톤/쉼 확보). 즉 vanilla Matcha가 밋밋한 건
아키텍처가 아니라 **낭독 데이터 + 결정론적 샘플링** 때문. 이 구분이 아래 선택을 가른다.

## 2. 밋밋함을 푸는 레버 — 실측으로 재정렬

> **실측 반영(중요)**: 스토캐스틱 duration predictor(VITS SDP)를 켜고 테스트했으나 대화체
> 표현력이 안 나왔다. 이건 문헌과도 일치한다 — VITS2 ablation에서 SDP vs 결정론적은 **MOS
> 0.14 차이(미미)**, Casanova 등은 SDP가 **부자연스러운 길이 → 발음 불명확**을 유발한다고 보고,
> 많은 후속 연구가 제어력 위해 **SDP를 다시 결정론적 예측기로 교체**했다. → **stochastic
> duration은 답이 아니다.** 이유: (1) 타이밍(리듬) 한 축만 흔들 뿐 피치·에너지·전달 스타일은
> 그대로 회귀로 뭉개짐, (2) "랜덤 분산 ≠ 맥락에 맞는 표현", (3) **결정타 — stochastic은 학습된
> 분포에서 샘플할 뿐, 밋밋한 데이터로 학습했으면 분포 자체가 밋밋해서 샘플해도 밋밋하다.**

그래서 레버를 영향력 순으로 다시 세운다:

| 순위 | 레버 | 원리 | 왜 이게 진짜 | CPU 비용 |
|---|---|---|---|---|
| **1** | **표현력 있는 데이터** | 자발적·대화·감정 음성으로 학습 | 학습 분포에 없는 표현력은 어떤 트릭으로도 못 만듦. stochastic이 안 된 근본 원인 | (데이터 작업) |
| **2** | **운율을 조건부 분포로 모델링** | 음소단위 prosody latent을 **AR/flow prior**로 예측(arXiv:2211.01327), 또는 style-diffusion(StyleTTS2), DDPM prosody(2305.16749) | 회귀(평균 뭉갬) 대신 다봉분포를 실제로 학습. 여기서 도움되는 "AR"은 *오디오 토큰 AR*(비쌈)이 아니라 *prosody latent AR*(작고 쌈) | 낮음~중 |
| **3** | **문맥(대화 이력) 조건화** | 이전 턴 텍스트/음향 → 현재 운율 조건 (FCTalker 2210.15360, M²-CTTS, CSM 사상) | "무작위 다양"이 아니라 "상황 적절" 운율 = 대화체의 정의 | 낮음 |
| **4** | **자발적 스타일 모델링** | spontaneous 병목/전이(SponTTS 2311.07179), filled-pause 예측기(AdaSpeech3) | 멈춤·비유창성의 자연스러움 | 낮음 |
| **5** | **NVV 인라인 토큰** | `[웃음]`/`[한숨]` vocab 확장(NVSpeech) | 비언어 발성 | 0 |
| — | ~~스토캐스틱 duration/pitch~~ | ~~VITS SDP, 스토캐스틱 F0~~ | **강등: 미미하고 때로 해로움(위 실측)** — 켜도 되지만 표현력의 해법 아님 | 0 |

핵심 통찰(수정): **대화체 표현력은 stochastic 트릭이 아니라 (1) 데이터 + (2) 운율의 조건부
분포를 제대로 모델링하는 것에서 나온다.** 아무리 좋은 아키텍처·샘플링도 낭독 데이터로 학습하면
낭독체다.

## 2b. 정직한 한계 (인정하고 가기)

오늘 CPU-only에서 CSM/Dia/Orpheus급의 "완전히 살아있는" 대화체는 어렵다 — 그 느낌은 **오디오
토큰 전체를 문맥 조건으로 autoregressive하게 모델링**하는 데서 오고, 그건 GPU(Orpheus-3B는 Q4
에서도 ~8GB VRAM)다. CM4에서 현실적으로 노릴 수 있는 건:
- (a) **표현력 있는 데이터 + 운율-latent prior + 문맥 조건화**로 소형 모델의 표현 상한을 최대한
  끌어올리기 (야심찬 자체 구축, 불확실하지만 "내 것"). — 아래 권장.
- (b) 디바이스는 "좋지만 CSM급은 아닌" TTS로 두고, 완전한 표현은 포기하거나 서버 연결 시에만.
- (c) 하이브리드: 디바이스 기본 + 서버(큰 AR) 표현 — 단, 이 저장소의 "완전 로컬" 목표와 상충.

## 3. 권장 아키텍처 — 데이터·운율분포 중심으로 재정렬

스토캐스틱이 답이 아니게 되면서 "VITS냐 flow냐"의 무게가 줄었다. 진짜 차별점은 **어느 베이스가
(1) 표현력 있는 데이터로 학습됐고 (2) 운율의 조건부 분포(prosody-latent prior)를 제대로 모델링
하며 (3) 문맥 조건화를 붙이기 쉬운가**이다. 아키텍처 자체보다 이 세 조건이 밋밋함을 가른다.

공통 코어(어느 베이스든 동일):
- **표현력 있는 자발적 대화 한국어 데이터** (1순위 — 없으면 무엇도 안 됨).
- **prosody-latent prior**: 음소단위 운율 latent을 AR/flow prior로 예측(arXiv:2211.01327) 또는
  style-diffusion. "평균 뭉갬"을 실제 다봉분포 모델링으로 대체. (오디오토큰 AR 아님 → CPU 가능)
- **대화 이력 조건화** + **NVV 토큰**.

베이스 후보:
- **FreyaTTS(flow) 유지**: 이미 비-flat(발음이 약점). 위 코어(데이터+prosody prior+문맥+NVV)를
  얹고 발음(G2P) 별개 트랙. **최저 위험** — 자연 운율을 다시 안 쌓아도 됨.
- **StyleTTS2-lite**: 운율 분포 모델링(style-diffusion)+pretrained SLM으로 표현력 상한 최고. 단
  학습 불안정·스트리밍 없음·디퓨전 지연 → CM4 위험 큼.
- **VITS2/Style-BERT-VITS2 새로**: CPU 검증 최강(Piper)·한국어 생태계 강함. 단 스토캐스틱은 이제
  셀링포인트 아님 → 여기에도 prosody-latent prior를 별도로 얹어야 표현력 나옴. 자연 운율 처음부터.

## 4. 판단

- **스토캐스틱 내장이 더 이상 VITS2 선택 이유가 못 됨**(실측·문헌). 그래서 "밋밋함 잡으려 VITS2"는
  약해졌다.
- FreyaTTS는 이미 비-flat이고 진짜 약점은 발음 → **밋밋함을 이유로 FreyaTTS를 버릴 근거 없음.**
- **결정 요인은 베이스가 아니라 공통 코어**(표현 데이터 + prosody-latent prior + 문맥 + NVV). 이게
  "내가 만드는 것"의 실질이자 자산.

**권장**: 베이스는 **FreyaTTS 유지**(최저 위험)하고, 노력은 전부 공통 코어에 — 특히 **① 표현력
있는 한국어 대화 데이터 확보, ② prosody-latent prior 도입**에 집중. 새 아키텍처(VITS2)로 갈아타도
같은 코어를 해야 하므로, 자연 운율을 이미 가진 FreyaTTS 위에서 코어를 검증하는 게 낭비 없다.
StyleTTS2 신규 구축은 CM4 리스크로 비권장.

## 5. 최소 구축 레시피 (베이스: FreyaTTS 유지 권장, 순서 = 영향력 순)

1. **데이터 먼저**: 표현력 있는 자발적 한국어 대화 음성 확보 — AI-Hub 대화/감정, CoreaSpeech 700h,
   방송/팟캐스트/잡담. NVV 인식 ASR 부트스트랩으로 `[웃음]`/`[한숨]` 자동 태깅 + 감정/스타일 라벨.
   부족분만 CosyVoice3 서버 teacher로 NVV·감정 데이터 생성 + VC/pitch 증강. **이게 1순위 — 데이터
   없이는 아래가 다 무의미.** 소규모라도 먼저 확보해 "밋밋한 분포" 문제부터 깬다.
2. **prosody-latent prior 도입**: 운율(피치/에너지/타이밍)을 회귀로 예측하지 말고, 음소단위 운율
   latent을 **AR 또는 flow prior로 예측**(arXiv:2211.01327)해 다봉분포를 학습. 이게 stochastic
   duration보다 훨씬 본질적인 "밋밋함 해소". 추론은 작은 prior 샘플이라 CPU 가능.
3. **문맥 조건화**: 이전 N턴(텍스트, 선택 음향)을 작은 인코더로 요약해 현재 운율에 조건
   (FCTalker/M²-CTTS). "상황 적절" 운율 = 대화체.
4. **NVV 토큰**: `[웃음]`/`[한숨]`/`[숨]`을 음소셋에 추가 (NVSpeech 패턴). 1의 데이터로 학습.
5. **발음(별개 트랙)**: 한국어 G2P/자소(JAMO) 견고화 — FreyaTTS의 실제 약점.
6. **평가**: 운율 다양성 지표(arXiv:2509.19928) + NV-Bench(2603.15352) + 사람 청취(A/B). `benchmark.py`
   로 CPU latency. **stochastic on/off가 아니라 "데이터+prior 넣기 전/후"로 A/B** 해야 실제 효과 보임.

## 5b. 실측 교훈 & 학습/프론트엔드 노트 (세션 정리)

이 프로젝트에서 실제로 부딪혀 확인한 것들. **모델을 갈아타기 전에 이걸 먼저 보라.**

**표현력·밋밋함**
- Qwen3-TTS 벌크 증류 → 화자 5명 다 딱딱. **표현력은 증류로 못 만듦**(teacher 천장 복제).
- VITS **stochastic duration 켜도 대화체 안 됨**(실측). 문헌 일치(VITS2 ablation MOS 0.14, SDP
  발음 부작용 → 후속 연구 결정론 회귀). → stochastic은 답 아님.
- 진짜 원인 = 회귀손실 평균뭉갬 + 결정론적 운율 + 낭독 데이터. 진짜 레버 순위 = **데이터 >
  운율 조건부 분포(prosody-latent prior) > 문맥 조건화.** stochastic 아님.
- **flat ≠ flow-matching.** FreyaTTS(flow)는 안 밋밋(발음이 약점). vanilla Matcha가 밋밋한 건
  데이터/결정론 탓.

**음질 디버깅 (금속성/먹먹)**
- ZipVoice from-scratch/AI-Hub(24kHz, 500h+) → 잡음/먹먹/금속성. **데이터 양 문제 아님**(500h 충분).
- 금속성 = acoustic↔보코더↔SR 체인 문제. **copy-synthesis(GT→피처→보코더→wav)로 분기**:
  금속성이면 보코더/피처설정, 깨끗하면 acoustic. **acoustic 파인튠 전에 이걸 먼저 하라**(보코더가
  원인이면 파인튠해도 안 고쳐짐).
- 보코더 피처설정(n_mels/n_fft/hop/fmin/fmax/sr)을 acoustic과 **완전 일치** 확인.
- from-scratch보다 **사전학습 fine-tune이 금속성에 강함**(깨끗한 음향 렌더링이 언어 불문 전이).

**파인튠·프리트레인 재사용**
- **프리트레인 못 쓰는 거 아님.** 웨이트 대부분(인코더+flow 디코더=음향 능력)이 vocab 무관하게
  그대로 로드됨 — 그게 금속성 해결의 핵심 부분.
- vocab 확장 = 리사이즈 **append**(기존 행 복사 + 새 행만 init). 한국어 토큰은 **맨 뒤에 append**
  (중간 삽입 금지, ID 밀림 방지). TTS는 **출력 vocab 없음**(연속 음향 출력) → 입력 임베딩만 손봄.
- 입력 스킴을 바꿔도(음소→자모/바이트) **음향 디코더는 전이** → from-scratch 회귀 아님. 텍스트
  인코더만 재학습.

**멀티링구얼**
- 유지하려면 **replay**(영/중/한 혼합 학습), 혼합 **비율** 관리가 곧 "비율 깨짐" 대응. 한국어 단일
  목표면 영/중 망각은 **손해가 아니라 이득**.
- ZipVoice 베이스가 이미 영/중 멀티 → 한국어 얹으며 영/중 replay가 "처음부터 멀티"보다 쉬움.

**프론트엔드 (eSpeak 회피)**
- eSpeak-ng: 한국어 규칙 약함 + **GPL-3.0** + C 의존성 → 피하고 싶은 이유 정당.
- 대안: **g2pK→jamo**(발음↑·무의존성·라이선스 깨끗, **추천**) / 자모만(무의존성, 규칙은 데이터로)
  / byte·BPE(멀티링구얼 균일·G2P 제거, 데이터 더 필요).
- ZipVoice는 언어별 토크나이저 교체 구조(`EspeakTokenizer` 등) → 한국어 토크나이저 새로 꽂기 자연스러움.

**최근 참고 (2026)**
- **ZipVoice/ZipVoice-Dialog**(k2-fsa, 123M flow, base는 CPU near-RT, 우리 sherpa-onnx와 같은 생태계):
  대화 레시피 = monologue 사전학습 → dialogue 파인튠 커리큘럼 + speaker-turn 임베딩 + `[S1][S2]`
  interleaved. (한국어·NVV 없음, dialog 벤치는 GPU → 드롭인 아니라 **레시피 각색용**.)
- **SwanVoice**(2605.30993): monologue+dialogue 표현 zero-shot 장문. **dots.tts**(2606.07080, 2B,
  서버): 의미계획(LLM)↔음향렌더(flow head) 분리. **Voxtral**(390M flow, prompt에서 운율 추론).
  **Kitten**(24MB, 범주 감정 보이스).

**현재 권장 (종합)**
- 베이스: **FreyaTTS 유지**(비-flat, 최저 위험). 노력은 공통 코어에 — ① 표현력 있는 한국어 대화
  데이터 ② prosody-latent prior ③ 문맥 조건화 ④ NVV 인라인 토큰. **발음은 g2pK로 별개 트랙.**
- 큰 표현 모델(CosyVoice3/CSM/Dia/Orpheus)은 디바이스 아님 → **서버 데이터 생성 teacher**로만.
- 다음 착수 후보: (a) copy-synthesis로 금속성 원인 분기, (b) 표현력 있는 한국어 대화 데이터 소싱,
  (c) g2pK→jamo 프론트엔드 PoC, (d) prosody-latent prior 설계.

## 6. 참고 문헌

- 진단: [Over-Smoothness in TTS(2202.13066)](https://arxiv.org/pdf/2202.13066) · [운율 다양성 지표(2509.19928)](https://arxiv.org/html/2509.19928v3)
- stochastic 한계(실측 뒷받침): [VITS2 ablation(2307.16430)](https://arxiv.org/pdf/2307.16430) (SDP vs 결정론 MOS 0.14) · SDP 부작용(Casanova 등) → 후속 연구 결정론 회귀
- **운율 조건부 분포 모델링(핵심)**: [음소단위 prosody latent AR/flow prior(2211.01327)](https://arxiv.org/pdf/2211.01327) · [DiffStyleTTS(2412.03388)](https://arxiv.org/pdf/2412.03388) · [DDPM prosody(2305.16749)](https://arxiv.org/pdf/2305.16749) · [Apple hierarchical prosody NAR](https://machinelearning.apple.com/research/hierarchical-prosody-modeling) · [Stochastic pitch/Glow-TTS(2305.17724)](https://arxiv.org/pdf/2305.17724)
- AR vs NAR·CPU 한계: [저자원 표현 NAR(Amazon 2106.12896)](https://arxiv.org/pdf/2106.12896) · [ZipVoice-Dialog NAR flow(2507.09318)](https://arxiv.org/pdf/2507.09318) · Orpheus-3B Q4 ~8GB VRAM(GPU 전용, CM4 불가)
- 대화 문맥: [FCTalker(2210.15360)](https://arxiv.org/pdf/2210.15360) · [RADKA-CSS(2501.06467)](https://arxiv.org/pdf/2501.06467) · [Conversational E2E TTS(2005.10438)](https://arxiv.org/pdf/2005.10438) · [Sesame CSM 블로그](https://www.sesame.com/blog/crossing-the-uncanny-valley-of-voice)
- 자발적 스타일: [SponTTS(2311.07179)](https://arxiv.org/pdf/2311.07179) · [ChatTTS](https://github.com/2noise/ChatTTS)
- 베이스: [Style-BERT-VITS2 표현 평가(2505.17320)](https://arxiv.org/html/2505.17320v1) · [MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean)
- NVV/데이터: [NVSpeech(2508.04195)](https://arxiv.org/abs/2508.04195) · [CosyVoice3(2505.17589)](https://arxiv.org/html/2505.17589v1) · [CoreaSpeech(NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/121811)
