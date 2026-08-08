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

## 2. 밋밋함을 푸는 레버 (전부 CPU에서 쌈)

| 레버 | 원리 | CPU 비용 | 대화체 기여 |
|---|---|---|---|
| **스토캐스틱 duration 예측** | flow 기반 확률적 길이 (VITS 내장). 결정론적 대비 인간다운·다양한 리듬 | 거의 0(추론 시 샘플) | 리듬 변동성 — 큼 |
| **스토캐스틱 pitch 예측** | F0도 확률적으로 (Glow-TTS 다양성 논문, arXiv:2305.17724) | 낮음 | 억양 변동성 |
| **운율 latent 샘플링** | 운율을 회귀 대신 분포에서 샘플(소형 VAE/flow/DDPM prosody predictor; DiffStyleTTS 2412.03388, DDPM prosody 2305.16749) | 낮음~중 | 표현 다양성 — 큼 |
| **문맥(대화 이력) 조건화** | 현재 발화 운율을 이전 턴 텍스트/음향에 조건 (FCTalker 2210.15360, M²-CTTS, CSM 사상) | 낮음(작은 인코더) | **대화 적절성 — 핵심** |
| **자발적 스타일 모델링** | spontaneous 스타일 병목/전이(SponTTS 2311.07179), filled-pause 예측기(AdaSpeech3) | 낮음 | 자연스러운 멈춤·비유창성 |
| **NVV 인라인 토큰** | `[웃음]`/`[한숨]` vocab 확장(NVSpeech 패턴) | 0 | 비언어 발성 |

핵심 통찰 두 가지:
1. **VITS는 스토캐스틱 duration predictor를 구조적으로 갖고 있다** — Matcha/FastSpeech2가 밋밋한
   바로 그 지점을 VITS는 설계상 회피. 그리고 VITS = Piper = CPU 실시간 검증됨.
2. **대화체의 진짜 결정 요인은 아키텍처보다 데이터(자발적 대화 음성) + 문맥 조건화.** 아무리 좋은
   아키텍처도 낭독 데이터로 학습하면 낭독체가 된다.

## 3. 권장 아키텍처 — 두 갈래

### 갈래 A: VITS2-family 새로 구축 (스토캐스틱이 구조적)

- **베이스**: VITS2 / Style-BERT-VITS2 계열. 스토캐스틱 duration + flow prior가 밋밋함을 구조적으로
  방지. Piper(VITS)가 CM4급에서 실시간이니 CPU 적합성 검증됨. 한국어 생태계도 강함(MeloTTS-Korean,
  Style-BERT-VITS2 KO).
- **얹기**: (a) 스토캐스틱 pitch, (b) GST/스타일 임베딩(감정), (c) **대화 이력 인코더**(이전 턴 →
  현재 운율 조건화), (d) NVV 토큰.
- **장점**: 밋밋함 방지가 구조 내장 + CPU 최강 검증 + ONNX 성숙. **단점**: FreyaTTS가 이미 이룬
  자연스러운 운율을 처음부터 다시 쌓아야 함.

### 갈래 B: FreyaTTS(flow-matching) 유지·확장

- FreyaTTS는 **이미 밋밋하지 않다**(자연 톤/쉼 확보). 그러면 밋밋함 레버 중 필요한 건 스토캐스틱
  운율 latent 정도이고, 나머지는 문맥 조건화 + NVV 토큰 + 발음 개선.
- **장점**: 가장 어려운 자연 운율을 이미 가짐 → 최저 위험. **단점**: flow 위에 스토캐스틱 다양성·
  문맥성을 얹는 건 VITS의 "공짜 스토캐스틱"보다 손이 더 감. 발음 문제도 병행 해결 필요.

## 4. 어느 갈래? (판단)

- **밋밋함만 놓고 보면** VITS2가 구조적으로 유리(스토캐스틱 내장). 사용자의 "Matcha 단정적"
  불만을 아키텍처 레벨에서 직접 해소.
- **그러나 FreyaTTS는 이미 비-flat**이라, "밋밋함 해결"만을 이유로 FreyaTTS를 버리는 건 근거 약함.
  FreyaTTS의 진짜 약점은 밋밋함이 아니라 **발음**이다.
- **결정 요인은 결국 공통**: 자발적 대화 데이터 + 문맥 조건화 + NVV 토큰 — 이건 어느 베이스든
  똑같이 해야 한다. 이 데이터/조건화 작업이 "내가 만드는 모델"의 실질 노력이자 자산.

**권장**:
- 소유·학습 가치를 중시하고 밋밋함을 구조로 확실히 잡고 싶다 → **갈래 A(VITS2-family)로 새로
  구축.** Piper/Style-BERT-VITS2를 스캐폴드로, 위 얹기 4종 + 자발적 대화 데이터. 아키텍처가 대화체를
  구조적으로 밀어줌.
- 최단 경로로 쓸만한 결과를 원한다 → **갈래 B(FreyaTTS)** + 문맥 조건화 + NVV + 발음 개선.
- 어느 쪽이든 **자발적 대화 한국어 데이터 + NVV 토큰 파이프라인**이 공통 선행 자산 → 여기부터
  시작하는 게 낭비 없음.

## 5. 최소 구축 레시피 (갈래 A 기준, 갈래 B도 3~5 공유)

1. **베이스 스캐폴드**: Style-BERT-VITS2(또는 MeloTTS-Korean) 한국어 체크포인트로 시작 —
   스토캐스틱 duration + 스타일 임베딩 이미 있음. ONNX-CPU 경로 확인.
2. **문맥 조건화 추가**: 이전 N턴(텍스트, 선택적으로 음향)을 작은 인코더로 요약해 현재 발화
   조건에 주입 (FCTalker/M²-CTTS 식). 대화체의 핵심.
3. **NVV 토큰**: `[웃음]`/`[한숨]`/`[숨]`을 음소셋에 추가 (NVSpeech 패턴).
4. **데이터**: 자발적 한국어 대화 음성(AI-Hub, CoreaSpeech 700h) → NVV 인식 ASR 부트스트랩으로
   자동 태깅 + 감정/스타일 라벨. 부족분은 CosyVoice3 서버 teacher로 NVV·감정 데이터 생성 +
   VC/pitch 증강. **중립→표현 커리큘럼.**
5. **발음(별개 트랙)**: 한국어 G2P/자소(JAMO) 처리 견고화 — FreyaTTS에도 공통 적용.
6. **평가**: 운율 다양성 지표(arXiv:2509.19928) + NV-Bench(2603.15352) + 사람 청취. `benchmark.py`로
   CPU latency.

## 6. 참고 문헌

- 진단: [Over-Smoothness in TTS(2202.13066)](https://arxiv.org/pdf/2202.13066) · [운율 다양성 지표(2509.19928)](https://arxiv.org/html/2509.19928v3)
- 스토캐스틱 운율: [VITS(stochastic duration)] · [Stochastic pitch/Glow-TTS(2305.17724)](https://arxiv.org/pdf/2305.17724) · [DiffStyleTTS(2412.03388)](https://arxiv.org/pdf/2412.03388) · [DDPM prosody(2305.16749)](https://arxiv.org/pdf/2305.16749) · [Apple hierarchical prosody NAR](https://machinelearning.apple.com/research/hierarchical-prosody-modeling)
- 대화 문맥: [FCTalker(2210.15360)](https://arxiv.org/pdf/2210.15360) · [RADKA-CSS(2501.06467)](https://arxiv.org/pdf/2501.06467) · [Conversational E2E TTS(2005.10438)](https://arxiv.org/pdf/2005.10438) · [Sesame CSM 블로그](https://www.sesame.com/blog/crossing-the-uncanny-valley-of-voice)
- 자발적 스타일: [SponTTS(2311.07179)](https://arxiv.org/pdf/2311.07179) · [ChatTTS](https://github.com/2noise/ChatTTS)
- 베이스: [Style-BERT-VITS2 표현 평가(2505.17320)](https://arxiv.org/html/2505.17320v1) · [MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean)
- NVV/데이터: [NVSpeech(2508.04195)](https://arxiv.org/abs/2508.04195) · [CosyVoice3(2505.17589)](https://arxiv.org/html/2505.17589v1) · [CoreaSpeech(NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/121811)
