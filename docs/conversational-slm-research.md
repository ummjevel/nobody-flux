# Conversational SLM 리서치 — 지식을 버리고 대화에 파라미터를 쓰는 0.5~2B

> 2026-08. 질문: **지식/reasoning을 포기하고 자연스러운 일상 대화 능력에 파라미터를 집중한
> 0.5~2B 모델은 실제 Voice Agent에서 어디까지 가능한가.**
> 선행 문서: [`llm-conversational-selection.md`](llm-conversational-selection.md) (현 기본
> midm-2.3b 선정 기록), [`FEATURES.md`](FEATURES.md) (파이프라인 현황). 타깃은 CM4(4코어
> Cortex-A72, GPU 없음), 언어는 한국어, 페르소나는 반말 캐주얼("퀜", `persona.py`).
>
> 표기 원칙: **[실측]** = 이 프로젝트에서 잰 것, **[출처]** = 외부 문서에서 읽은 것,
> **[추정]** = 둘 다 아닌 것. 라이선스는 전부 원본 모델 카드/라이선스 파일 기준이며
> 양자화 저장소 표기는 신뢰하지 않는다(선행 문서에서 이미 한 번 데였다).

## 결론 먼저

1. **방향 자체는 학술적으로 이미 증명됐다.** SODA/COSMO 연구가 정확히 이 가설을 검증했다:
   대형 모델(teacher)로 일상 대화 150만 개를 생성해 소형 모델에 distill했더니, 그 소형 모델
   (COSMO, 3B급)이 자연스러움·일관성에서 훨씬 큰 대화 모델들을 이기고 **때로는 사람이 쓴
   골드 응답보다도 선호**됐다 [출처: [SODA, EMNLP 2023](https://arxiv.org/abs/2212.10465)].
   대화 능력은 지식과 분리 가능하고, 분리하면 소형에 들어간다. **단, 전부 영어다.**
   한국어 + 반말 + 음성 제약(TTS로 나감) 조합의 기성 모델은 2026-08 현재 존재하지 않는다 —
   이 조합은 직접 만들어야 하고, 만드는 방법(distillation SFT)은 위 연구가 그대로 레시피다.

2. **지형도가 이 프로젝트의 선정 문서 이후로 크게 움직였다.** 세 가지가 새로 왔다:
   - **Kanana-2 1.3B/3B** (2026-07, 카카오) — 한국어 특화 경량이 드디어 갱신됐지만,
     라이선스(KananaOpenLicense)가 **"온디바이스 제품 임베드는 별도 상업 라이선스"를 명시**한다.
     이 프로젝트가 하려는 게 정확히 그것이라, 성능 상한 참고용 이상이 못 된다(§1).
   - **Qwen3.5 Small 0.8B/2B** (2026-03, Apache-2.0 계열) — 0.6B→0.8B 자리의 세대 교체 후보.
   - **LFM2.5 230M~2.6B** (2026-01~08) — 라즈베리파이급을 명시 타깃으로 하지만, LFM2가 이
     페르소나에서 실패한 전적이 있어 [실측] 재검증 대상이지 기본 후보가 아니다.
   - Mi:dm은 서버용 32B(Mi:dm K 2.5 Pro)만 갱신됐고 **소형 공개 최신은 여전히 2.0 Mini 2.3B**다.

3. **"어디까지 가능한가"의 현재 답: 프롬프트만으로는 2B 부근이 하한이고, SFT가 그 하한을
   내릴 수 있는지가 미검증 핵심 질문이다.** 이 프로젝트 실측으로 0.6B는 프롬프트+few-shot으로도
   붕괴(위반 5/18, 되묻기 33%, 앵무새/문법 붕괴)했고 2.3B는 깨끗했다(0/18)
   [실측: 선정 문서]. 그러나 이건 **범용 instruct 모델에 프롬프트를 얹은 결과**다. COSMO의
   증거는 대화 특화 SFT가 이 그림을 바꿀 수 있다고 말한다 — 0.6~1.3B에 수천~수만 턴의
   페르소나 대화를 SFT했을 때 2.3B 프롬프트 수준에 도달하는지가 실험할 가치가 가장 큰
   단일 질문이다(§5).

4. **CM4 예산이 이 리서치의 존재 이유다.** Pi 4/CM4(Cortex-A72)에서 Q4 1B급은 한 자릿수
   초반 tok/s, 2.3B는 1~2 tok/s 수준으로 보인다 [추정: Pi 5 실측치에서 역산, §5.5].
   40자 응답 ≈ 30~60토큰이므로 2.3B는 첫 문장까지도 수 초가 걸릴 수 있다. **즉 midm-2.3b를
   CM4에 그대로 들고 가는 그림은 위태롭고, "0.6~1.3B에 대화를 밀어넣는" 이 문서의 주제가
   플랜 B가 아니라 사실상 메인 경로일 가능성이 높다.** (CM4 실기 실측이 선행 과제인 이유.)

5. **duplex 네이티브(Moshi/PersonaPlex)는 대화의 '타이밍'에서 앞서 있지만 전부 7B+GPU다.**
   CM4에서는 불가능하고, 이 프로젝트는 그들이 모델 내부에서 푸는 4축(pause/turn-taking/
   backchannel/interruption)을 시스템 레벨(turn/ 패키지)에서 이미 풀고 있다. 텍스트 SLM
   경로 유지가 옳다(§4).

---

## 1. 온디바이스 SLM 지형도 (0.3B~4B, 2026-08)

### 1.1 범용(다국어) 모델

| 모델 | 크기 | Q4 메모리* | 한국어 | 라이선스 | GGUF | 비고 |
|---|---|---|---|---|---|---|
| Qwen3 | 0.6B/1.7B/4B | 0.4/1.1/2.5GB | 공식 119개 언어, 실측 사용 가능 | **Apache-2.0** | 공식 | 현 프리셋. thinking 스위치 [실측] |
| **Qwen3.5 Small** | 0.8B/2B/4B/9B | ~0.5/1.2/2.4GB | 미실측 — Qwen3 계보 | Apache-2.0 (보도) | 커뮤니티([lmstudio](https://huggingface.co/lmstudio-community/Qwen3.5-0.8B-GGUF)) | 2026-03. Gated DeltaNet 하이브리드 어텐션(3:1 linear:full), 262K ctx, 멀티모달 [출처: [MarkTechPost](https://www.marktechpost.com/2026/03/02/alibaba-just-released-qwen-3-5-small-models-a-family-of-0-8b-to-9b-parameters-built-for-on-device-applications/), [Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models)] |
| Gemma 3 | 270M/1B | 0.2/0.7GB | 다국어이나 한국어 비중 낮음 [추정] | Gemma Terms + gated | 공식 | 270M은 분류/추출용 초경량 [출처: [Google](https://ai.google.dev/gemma/docs/releases)] |
| Gemma 3n | E2B/E4B | 실효 ~2/3GB | 위와 동일 | Gemma Terms | 공식 | raw 5B/8B를 **PLE 캐싱**(레이어별 임베딩을 CPU/스토리지로 분리)으로 2B/4B 메모리에 구겨넣음 [출처: [Google 개발자 블로그](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)] |
| Gemma 4 | E2B/E4B/26B/31B | E2B ~2GB | 미실측 | **Apache-2.0 (보도·릴리스 노트)** — 채택 전 모델 카드 직접 확인 필수 | 확인 필요 | 2026-03-31 [출처: [Google releases](https://ai.google.dev/gemma/docs/releases), [aurigait](https://aurigait.com/blog/gemma-4-features-benchmarks-guide/)]. Gemma가 Apache로 왔다면 라이선스 지형이 바뀐 것 |
| Llama 3.2 | 1B/3B | 0.7/1.9GB | **공식 지원 8개 언어에 한국어 없음** | Llama Community License | 공식 | 한국어 후보로는 약함 [출처: 모델 카드] |
| SmolLM2/3 | 135M~3B | ~0.1~1.9GB | **미지원**(유럽 6개 언어) | Apache-2.0 | 공식 | 학습 전 과정 공개는 참고 가치 [출처: [TinyWeights](https://tinyweights.dev/posts/best-small-language-models-2026/)] |
| Phi-4-mini | 3.8B | ~2.3GB | 다국어이나 어시스턴트/추론 지향 | MIT | 공식 | 크기·성격 모두 이 프로젝트와 반대 방향 [출처: [HF 카드](https://huggingface.co/microsoft/Phi-4-mini-instruct)] |
| LFM2 | 350M/700M/1.2B | 0.2/0.4/0.7GB | 8개 언어에 한국어 포함 | LFM Open License(매출 상한) | 공식 | **[실측] 이 페르소나에 부적합** — 350M 마크다운 남발, 700M 장황·이모지(`models.yaml` 주석) |
| **LFM2.5** | 230M/1.2B/2.6B/VL-3B | 0.15~1.6GB | VL은 한국어 명시, 텍스트는 확인 필요 | 확인 필요(LFM 계열) | 공식(전 모델) | 2026-01~08. 2.6B가 "라즈베리파이에서 에이전트" 명시 타깃 [출처: [Liquid AI](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai), [VentureBeat](https://venturebeat.com/technology/no-cloud-no-gpus-no-problem-liquid-ais-new-model-lfm2-5-2-6b-brings-powerful-ai-agents-to-devices-as-small-as-a-raspberry-pi), [LFM2.5-230M](https://www.liquid.ai/blog/lfm2-5-230m)] |

\* Q4_K_M GGUF 파일 크기 기준 근사(파라미터 수 × ~0.6byte) + KV 캐시/런타임 별도. [추정]

### 1.2 한국어 특화 모델 — 라이선스가 여전히 1차 필터다

| 모델 | 크기 | 라이선스 | 온디바이스 상업 배포 | 비고 |
|---|---|---|---|---|
| **KT Mi:dm 2.0 Mini** | 2.3B | **MIT** | ✅ 무제한 | 현 기본. Base에서 width pruning + 다단계 distillation으로 만든 온디바이스 목적 모델 [출처: [모델 카드](https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct), [기술 보고서](https://arxiv.org/abs/2601.09066)]. 2026 신작 Mi:dm K 2.5 Pro는 32B 서버용 — 소형 갱신 없음 [출처: [arXiv](https://arxiv.org/abs/2603.18788)] |
| **Kakao Kanana-2** | **1.3B/3B** (+30B MoE) | **KananaOpenLicense** | ⚠️ **별도 상업 라이선스 필요** | 2026-07 공개, 한국어 벤치에서 Qwen3-1.7B 크게 앞섬(KoSimpleQA 32.5 vs 13.4), 공식 GGUF 있음 [출처: [모델 카드](https://huggingface.co/kakaocorp/kanana-2-3b-instruct)]. **LICENSE 파일 직접 확인 결과**: 자사 서비스 이용은 자유이나 ①제3자 API 제공 ②SI/온프레미스 재판매 ③**온디바이스 제품 임베드**는 카카오와 별도 계약 + "Powered by Kanana" 표기 의무. 이 프로젝트는 정확히 ③에 해당 |
| LG EXAONE 4.0 | **1.2B**/32B | EXAONE License 1.2-**NC** | ❌ 상업은 별도 계약 | 2025-07. 1.2B는 크기·목적이 딱인데 라이선스가 그대로 막는다 — 3.5 때와 동일한 결론 [출처: [모델 카드](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B), 공식 GGUF 있음] |
| Naver HyperCLOVA X SEED | 0.5B/1.5B/3B | 자체 라이선스 + gated | ⚠️ 확인 불가(선정 문서와 동일) | 2025-12에 32B THINK/8B Omni 추가 — 소형 라인 변화 없음 [출처: [나무위키 HyperCLOVA](https://namu.wiki/w/HyperCLOVA)] |
| SKT A.X-4.0-Light | 7B | Apache-2.0 | ✅ (크기 초과) | 타깃 밖 |

**시사점 세 개:**

- **"한국어 특화 + 온디바이스 상업 배포 가능"의 답은 여전히 Mi:dm 2.0 Mini 하나다.**
  Kanana-2는 성능이 매력적이지만 온디바이스 임베드 조항 때문에 계약 없이는 배포 불가 —
  다만 카카오가 계약 창구를 열어놨으므로, 제품화 단계에서 협상 가능한 옵션으로는 유효하다.
- **자유 라이선스 축은 Qwen 계열이 굳혔다.** Qwen3 0.6B/1.7B(현 프리셋) 위에 Qwen3.5
  0.8B/2B가 얹혔고, Gemma 4가 정말 Apache라면 선택지가 하나 더 는다.
- **0.5B 미만(Gemma 3 270M, LFM2.5-230M)은 이 용도에 안 맞는다.** 이 급은 분류·추출·
  도구 호출 같은 좁은 태스크용으로 포지셔닝돼 있고 [출처: Liquid AI, Google 블로그],
  자유 대화는 명시적 비타깃이다. 대화 특화 SFT의 하한 실험 대상으로도 0.5B~를 권장.

## 2. "대화 특화 소형 모델" 선행 연구 — 이 가설은 새 것이 아니다

### 2.1 Conversational distillation: SODA → COSMO가 원형

- **SODA** [출처: [arXiv:2212.10465](https://arxiv.org/abs/2212.10465), EMNLP 2023]:
  상식 그래프의 사회적 상식을 시드로 InstructGPT에서 **150만 개 일상 대화를 생성(distill)**.
  사람 평가에서 기존 사람-작성 코퍼스(DailyDialog 등)보다 **더 일관되고 구체적이고
  자연스럽다**고 판정됐다 — teacher 생성 데이터가 사람 데이터보다 나을 수 있다는 것 자체가 발견.
- **COSMO**(SODA로 학습한 소형 대화 모델): 미학습 데이터셋에서 GODEL·BlenderBot·Koala·
  Vicuna보다 자연스럽고, **때로는 골드 응답보다 선호**. 지식·추론 벤치마크는 애초에 안 쟀다 —
  이 프로젝트의 문제의식("벤치마크가 아니라 다시 대화하고 싶은가")과 정확히 같은 평가 철학.
- **PersonalityChat** [출처: [arXiv:2401.07363](https://arxiv.org/html/2401.07363)]:
  페르소나 사실 + 성격 특성까지 넣은 distillation으로 소형(GODEL-small급) 개인화 대화 모델을
  학습 — "페르소나 일관성도 distill 가능"의 직접 증거.
- 유사 계열로 PLACES(프롬프트 기반 대화 합성) 등이 있다 [출처: PersonalityChat 관련 연구 절].

**이 프로젝트에의 번역**: teacher(대형 한국어 모델)로 "퀜" 페르소나 대화를 수천~수만 턴
생성 → 0.6~2.3B에 SFT. 선정 문서가 "현실적인 경로 1"로 이미 지목한 것과 같고, TTS의
teacher/student 구도(`tts-expressivity-design.md`)와도 같은 패턴이다.

### 2.2 BlenderBot 계열의 교훈 (2020~2022)

BlenderBot은 "대화 스킬 블렌딩(공감·페르소나·지식)"으로 2.7B가 9B급과 겨루는 걸 보였지만,
retrieval 없이는 **지식 질문에 자신 있게 환각**했고, 세대가 갈수록 검색·안전 모듈이 붙으며
무거워졌다 [출처: [BlenderBot 3, arXiv:2208.03188](https://arxiv.org/abs/2208.03188); 요지는
지식 컷오프 이전 일반 지식]. 교훈 두 개:

1. **소형 대화 모델의 지식 환각은 못 고치는 게 아니라 회피하는 것이다.** "모르면 모른다고
   말한다"가 페르소나 규칙(`persona.py`)에 이미 있는 건 우연이 아니라 필수다. 학습 데이터에
   "지식 질문 → 솔직한 회피 + 되묻기" 패턴을 충분히 넣어야 한다.
2. **안전은 크기로 해결 안 된다.** 건강/법률/돈 회피 규칙도 데이터로 들어가야 한다.

### 2.3 페르소나 일관성 — 최근(2025~26) 연구

- **Persona-Aware Contrastive Learning** [출처: [ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.1344/), [arXiv:2503.17662](https://arxiv.org/abs/2503.17662)]: 라벨 없이 role consistency를 올리는 대조 학습.
- **Multi-turn RL로 페르소나 일관성** [출처: [arXiv:2511.00222](https://arxiv.org/abs/2511.00222)]: 멀티턴 RL로 페르소나 불일치 55%+ 감소.
- **PersonaArena** [출처: [arXiv:2605.17044](https://arxiv.org/pdf/2605.17044)]: 동적 시뮬레이션 기반 role-play 평가.

공통 발견은 "페르소나 드리프트는 멀티턴에서 발생한다"는 것 — 단발 QA로는 안 잡힌다.
현 `_ab_persona.py`가 단발 입력 기반이라는 한계와 정확히 맞물린다(§5.2에서 멀티턴 확장).

### 2.4 응답 길이 제어

명시적 길이 지시는 SOTA 모델도 자주 어긴다 [출처: [Characterizing LLM response lengths,
arXiv:2506.08686](https://arxiv.org/pdf/2506.08686)]. 접근은 세 갈래: ①프롬프트 지시(현행,
소형에서 불안정) ②선호 학습에 길이 제약 위반을 패자로 넣는 **LIFT** ③학습 데이터의 길이
분포 자체를 목표 분포(이 프로젝트는 30~60자)로 맞추기. 소형 모델 SFT에서는 ③이 가장 싸고
확실하다 [추정 — 단, SFT가 데이터 분포를 모방한다는 일반 원리에 근거].

### 2.5 한국어 일상대화 데이터

AI Hub [일상대화](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=543) /
[감성대화](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271) /
[SNS 멀티턴](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71694) 등이 있으나
①승인 절차 ②톤이 페르소나와 다름(선정 문서와 동일한 판단) ③국내 IP/기관 요건이 있다.
**주 데이터는 distillation으로 만들고, AI Hub는 사람 대화의 화제 분포·턴 길이 분포를
참조하는 통계적 앵커로만 쓰는 것을 권장** — 예: "실제 한국어 일상대화의 중앙값 발화 길이"를
teacher 생성 데이터의 길이 분포 검증에 사용.

## 3. Voice-first 대화 능력의 정의와 평가

### 3.1 왜 MT-Bench류가 아닌가 — 이번엔 외부 근거가 생겼다

선정 문서의 판단(벤치마크 점수 ≠ 대화 상대 품질)을 2025~26 평가 연구가 뒷받침한다:

- **VoiceBench** [출처: [arXiv:2410.17196](https://arxiv.org/abs/2410.17196)]: LLM 기반 음성
  어시스턴트 평가 — 단 "어시스턴트" 태스크 중심이라 컴패니언 평가로는 부분 적용.
- **Audio MultiChallenge** [출처: [arXiv:2512.14865](https://arxiv.org/pdf/2512.14865)]:
  멀티턴 음성 대화의 자연스러운 상호작용(수정 발화, 화제 전환 등)을 평가 — 텍스트 벤치가
  못 보는 축을 명시적으로 분리.
- **SDiaReward** [출처: [arXiv:2603.14889](https://arxiv.org/pdf/2603.14889), ACL 2026]:
  음성 대화 보상 모델. 핵심 개념이 **colloquialness gap** — "대본/문어체 대화"와 "자연스러운
  구어 대화"를 구분하는 축을 보상으로 모델링. 이 프로젝트가 EXAONE에서 관측한 실패(문어체
  장황함)가 학계에서 정식 평가축이 됐다는 뜻.
- **한국어**: [LogicKor](https://huggingface.co/blog/amphora/navigating-ko-llm-research-2)가
  한국어 MT-Bench 격이지만 추론·수학·코딩 중심이라 **이 용도에 부적합**. 음성 쪽은
  KVoiceBench 등 한국어 SpeechLM 벤치마크가 나오기 시작했다 [출처:
  [arXiv:2605.27984](https://arxiv.org/html/2605.27984)] — 단 이는 음성 입출력 모델용이고
  텍스트 LLM 스테이지 평가에는 직접 안 맞는다. **한국어 컴패니언 대화 평가는 기성품이 없다.
  직접 만들어야 하고, 이미 절반(기계 지표)은 있다.**

### 3.2 평가축 제안 — 기존 `_ab_persona.py` 위에 쌓기

| 축 | 측정 방법 | 상태 |
|---|---|---|
| 규칙 위반(반말/이모지/숫자·로마자/앵무새) | 정규식, `_ab_persona.py` | **있음** [실측 도구] |
| 되묻기(턴 반환율) | 휴리스틱, 동일 | **있음** |
| 판박이(distinct rate) | 동일 | **있음** |
| 길이 분포(30~60자 target) | 동일(중앙값) → 분포로 확장 | 확장 필요 |
| 반복(멀티턴 내 n-gram self-repetition) | distinct-n을 세션 단위로 | 신규 — 판박이의 멀티턴 일반화 |
| **문맥 일관성**(3턴 전 언급을 기억하는가) | LLM-as-judge | 신규 |
| **페르소나 일관성**(멀티턴 드리프트) | LLM-as-judge + 규칙(이름/나이/말투) | 신규 |
| follow-up 적절성(되물음이 문맥에 맞는가) | LLM-as-judge | 신규 — 되묻기"율"만으로는 기계적 되물음을 못 거름 |
| 감정 적절성(위로/축하/맞장구 매칭) | LLM-as-judge | 신규 |
| ttfb / tok/s / 메모리 | `benchmark.py`, `_ab_persona.py`, `storage.py` turns | **있음** |

### 3.3 LLM-as-judge 설계

- **pairwise 비교가 절대 점수보다 신뢰도가 높다** [출처: [MT-Bench/Chatbot Arena,
  arXiv:2306.05685](https://arxiv.org/html/2306.05685v4)]. A/B 응답을 무작위 순서로 제시하고
  스왑해서 두 번 물어 position bias를 상쇄.
- 축별 루브릭을 분리한다(자연스러움 따로, 페르소나 따로) — 뭉뚱그린 "품질" 점수는 verbosity
  bias(긴 답 선호)에 오염된다 [출처: 동일 논문]. **이 프로젝트에서는 verbosity가 감점이므로
  루브릭에 "음성으로 나가는 대화다, 길면 나쁘다"를 명시해야 한다.**
- **judge와 teacher를 분리한다.** teacher가 만든 데이터로 학습한 모델을 같은 모델이 채점하면
  self-preference bias로 낙관 오차가 생긴다 [출처: LLM-as-judge 서베이,
  [Awesome-LLMs-as-Judges](https://github.com/CSHaitao/Awesome-LLMs-as-Judges)].
- judge는 한국어 반말의 미묘함(존댓말 혼입, 어색한 조사)을 읽어야 하므로 한국어 강한 대형
  모델로. 기계 지표(§3.2 상단)가 먼저 싸게 거르고, judge는 통과분만 채점 — 비용 통제.
- **최종 게이트는 여전히 사람이다.** 자동 평가는 "아닌 것"을 거를 뿐(선정 문서의 원칙 유지).
  judge 점수와 사람 선호의 상관을 소량(20~30 페어)으로 먼저 확인하고 나서 judge를 신뢰할 것.

## 4. Turn-taking / duplex 네이티브 모델 동향 — 그리고 왜 우리는 텍스트 SLM인가

### 4.1 동향 (2024→2026)

- **Moshi** (Kyutai, 2024-09): 7B Helium 기반, 사용자/시스템 오디오를 병렬 스트림으로 동시
  모델링. 이론 지연 160ms, L4 GPU에서 실효 ~200ms [출처: [Moshi 논문](https://kyutai.org/Moshi.pdf),
  [GitHub](https://github.com/kyutai-labs/moshi)]. 끼어들기 정확도는 62% 수준 [출처: 아래 비교].
- **PersonaPlex** (NVIDIA, 2026-02): Moshi 아키텍처 확장 7B. 시스템 프롬프트로 페르소나·목소리
  전환, speaker-switch 70ms, 사용자 끼어들기 성공률 100%(Moshi 60.6%) [출처:
  [GitHub](https://github.com/NVIDIA/personaplex), [분석](https://www.kunalganglani.com/blog/nvidia-personaplex-full-duplex-voice-ai)].
  로컬 구동 보고는 **8GB VRAM GPU** 기준 [출처: [MakeUseOf](https://www.makeuseof.com/nvidia-personaplex-local-speech-model-8gb-vram/)].
- **후속 연구가 4축으로 수렴**: pause handling / turn-taking / backchanneling / user
  interruption을 축별 보상으로 RL 후처리 [출처: [arXiv:2607.07148](https://arxiv.org/pdf/2607.07148)].
  전체 조망은 full-duplex SLM 서베이 [출처: [arXiv:2509.14515](https://arxiv.org/pdf/2509.14515)].

### 4.2 트레이드오프 — 이 프로젝트 구조와의 비교

| | duplex 네이티브 (Moshi/PersonaPlex) | 이 프로젝트 (cascade + 시스템 턴 관리) |
|---|---|---|
| 턴 타이밍 품질 | 모델이 학습으로 획득 — 상한이 높다 | VAD+Smart Turn+상태기계 — 규칙 기반, 튜닝 가능 |
| 하드웨어 | 7B + GPU(≥8GB VRAM) | **CM4 CPU에서 성립하는 유일한 경로** |
| 언어 | 영어 중심(한국어 duplex 공개 모델 없음, 2026-08) | ASR/LLM/TTS 각각 한국어 최적 선택 가능 |
| 모델 교체 | 통짜 — LLM만 갈아끼우기 불가 | 스테이지별 독립 스왑(`models.yaml`) |
| backchannel/barge-in | 모델 내부(음향적으로 자연) | `turn/` 패키지에서 명시적 구현 [실측 완료 부분 있음] |
| 실패 모드 | 불투명(왜 끼어들었는지 모름) | 상태기계라 디버깅 가능 |

**결론: 텍스트 SLM 경로 유지.** duplex의 교훈은 가져온다 — 그들이 보상으로 최적화하는 4축이
곧 이 프로젝트 `turn/` 패키지의 명세이고, 평가축(§3)에도 넣을 수 있다(예: backchannel에
응답을 끊지 않는지). duplex 네이티브는 "서버급 하드웨어가 허용되는 미래 제품"에서 재평가.
nobody-stream(서버 S2S) 쪽 관심사이지 nobody-flux(엣지)의 현재 관심사가 아니다.

## 5. 실험 설계 제안 — "어느 크기부터 대화가 되는가"

### 5.1 두 개의 질문으로 분해

- **Q1 (크기 사다리, 프롬프트만)**: 동일 프롬프트·동일 대화셋에서 크기별 대화 품질 곡선은
  어떻게 생겼는가? 어디서 꺾이는가?
- **Q2 (SFT 효과)**: 대화 특화 SFT가 그 곡선을 왼쪽(작은 쪽)으로 얼마나 미는가?
  구체적으로 **"0.6~1.3B + SFT ≥ 2.3B + 프롬프트"가 성립하는가?** 성립하면 CM4 예산 문제가
  풀리고, 프리필 토큰(현재 매 턴 재전송하는 페르소나+few-shot)도 가중치로 흡수돼 레이턴시가
  이중으로 준다.

### 5.2 후보 사다리

| 단계 | 모델 | 역할 | 라이선스 |
|---|---|---|---|
| 0.6B | qwen3-0.6b-gguf (기존 프리셋) | 하한 앵커 — 프롬프트만으론 붕괴 [실측] | ✅ |
| 0.8B | Qwen3.5-0.8B (신규 프리셋) | 0.6B의 세대 교체 후보 | ✅(확인) |
| 1.3B | Kanana-2-1.3B | **참고 상한**(한국어 특화 이 크기의 잠재력) — 배포 불가* | ⚠️ |
| 1.7B | qwen3-1.7b-gguf (기존 프리셋) | 자유 라이선스 중간 앵커 | ✅ |
| 2B | Qwen3.5-2B | 신형 2B | ✅(확인) |
| 2.3B | midm-2.3b-gguf (현 기본) | **기준선** — 이걸 못 이기면 의미 없음 | ✅ |
| 3B | Kanana-2-3B / Gemma 4 E2B | 상한 참고 | ⚠️/확인 |

\* Kanana-2는 실험(내부 평가)은 라이선스상 가능하나 온디바이스 제품 배포는 별도 계약(§1.2).
"1.3B 한국어 특화가 얼마나 하는지"의 데이터 포인트로만 쓴다 — Kanana/EXAONE을 참고 상한으로
쓰던 선정 문서의 방식 그대로.

### 5.3 평가 프로토콜

1. **대화셋 고정이 먼저다**(선정 문서 "평가를 먼저 고정한다"의 확장).
   - 현 `_ab_persona.py`의 단발 6입력 → **멀티턴 시나리오 20~30개 × 4~6턴**으로 확장.
     시나리오는 `storage.py`의 turns 테이블에 쌓인 실사용 턴에서 추출(실제 실패 사례 우선 —
     기존 regression-set 철학 유지) + 감정/화제전환/지식질문 회피 등 §3.2 축별 커버.
   - 멀티턴 스크립트의 사용자 측 발화는 고정하고(모델 응답과 무관하게 진행 가능한 시나리오로
     설계), 모델 응답만 바꿔가며 비교 — 재현성 확보.
2. **3단 평가**: ①기계 지표(§3.2 상단, 확장된 `_ab_persona.py`) → ②LLM-as-judge pairwise
   (§3.3, 기준선 midm-2.3b 대비 승률) → ③사람 대화(최종 후보 2개만, `talk.py`로 직접).
3. **운영 지표 동시 측정**: 같은 실행에서 ttfb/총 시간(이미 측정), 피크 메모리(신규 —
   `llama.cpp` 로그 또는 psutil), 개발 박스 CPU와 CM4 실기 양쪽.
4. **결과 해석 규칙**: Q1 곡선에서 "기계 지표 통과 + judge 승률 40% 이상"이 나오는 최소
   크기가 "프롬프트만으로 대화가 되는 하한". Q2는 SFT 모델이 같은 기준을 통과하는 최소 크기.

### 5.4 학습 파이프라인 (Q2)

1. **teacher 선정.** 조건: 한국어 반말 자연스러움 + 출력 데이터의 학습 이용 허용.
   - 후보 A — 상용 API 대형 모델: 품질 최상. **각 제공사 약관의 '출력물로 모델 학습' 조항을
     계약 관점에서 확인하고 문서에 기록할 것**(제공사별로 다르고 자주 바뀐다 — 여기 적어봤자
     낡는다).
   - 후보 B — 오픈 가중치 대형(Qwen3 235B/32B 등 Apache 계열): 약관 문제 없음, RTX 5090
     또는 H100 서버에서 셀프호스트. 품질이 A보다 낮으면 A로 초안 → B로 증량하는 혼합도 가능.
   - 후보 C — EXAONE/HyperCLOVA 등 NC 모델을 teacher로: **금지.** NC 모델 출력으로 상업 모델을
     학습하는 건 라이선스 위반 소지가 크다.
   - teacher ≠ judge (§3.3).
2. **데이터 생성.** SODA 방식의 한국어·페르소나 번안:
   - 시드 = 화제(일상 화제 목록) × 감정 상태 × 시나리오 골격. AI Hub 통계로 화제·길이 분포 검증(§2.5).
   - teacher에게 "퀜" 페르소나 시스템 프롬프트 + 음성 제약(숫자 한글 표기, 이모지 금지,
     30~60자, 되묻기)을 주고 **양쪽 화자 모두 생성**(사용자 발화도 합성 — ASR 오인식 섞인
     변형도 일부 포함하면 강건성에 도움 [추정]).
   - 규모: 5천 턴에서 시작 → 2만 턴까지 증량 실험. 브랜드 보이스류 스타일 튜닝은 500~2,000
     예시로 충분하다는 통설 [출처: [2026 LoRA/QLoRA 가이드](https://pub.towardsai.net/fine-tuning-llms-in-2026-lora-qlora-unsloth-and-everything-in-between-929eaf94aea2)]이나,
     멀티턴 대화 태도까지 넣으려면 그보다 많이 필요하다고 본다 [추정].
   - **생성 데이터도 기계 지표로 필터링한다** — 규칙 위반 턴이 학습 데이터에 들어가면 그대로 배운다.
3. **학습.**
   - 1차: **QLoRA r=16** (선정 문서 계획 그대로). 2.3B 이하는 RTX 5090 32GB에서 여유.
   - 0.6~0.8B는 **full fine-tune도 병행 비교** — 이 크기는 full FT도 단일 GPU로 싸고,
     스타일이 아니라 "대화 태도" 같은 넓은 변화는 LoRA rank가 병목일 수 있다 [추정].
   - 멀티턴 학습 형식: 세션 단위 샘플(마지막 assistant 턴만 loss) — 페르소나 드리프트가
     멀티턴에서 생기므로(§2.3) 단발 QA 형식으로 만들면 안 된다.
   - 파국적 망각 체크: SFT 후 안전 회피(건강/법률/돈)와 "모른다" 응답이 살아있는지를
     평가셋에 포함.
4. **반복.** 평가(§5.3) → 실패 유형을 시드에 추가 → 재생성 → 재학습. `_ab_persona.py`의
   "실패에서 온 regression set" 철학을 데이터 생성 루프에 그대로 적용.

### 5.5 CM4 추론 예산 [추정 — 실측으로 대체할 것]

CM4 = BCM2711, 4×Cortex-A72 @1.5GHz (Pi 4와 동일 실리콘). 공개 실측은 Pi 5(Cortex-A76,
2~3배 빠름) 위주라 역산치만 제시한다:

| 모델(Q4_K_M) | Pi 5 보고치 | CM4 역산 [추정] | 40자(≈45tok) 응답 생성 시간 |
|---|---|---|---|
| 0.6~1.1B | 5~10 tok/s | **2~5 tok/s** | 9~23s… 단 스트리밍이라 첫 문장(~15tok)은 3~8s |
| 1.7~2.3B | 3~6 tok/s | **1~3 tok/s** | 첫 문장 5~15s — 대화로 성립 안 할 가능성 높음 |
| 3B | 4~6 tok/s(Pi 5) | ~1 tok/s | 사실상 불가 |

[출처: Pi 5 수치 — [Stratosphere](https://www.stratosphereips.org/blog/2025/6/5/how-well-do-llms-perform-on-a-raspberry-pi-5), [sitepoint](https://www.sitepoint.com/llms-raspberry-pi-edge/); CM4 역산은 A72/A76 배율 기반 추정]

이 표가 맞다면 **CM4에서 대화가 성립하는 크기는 0.6~1.3B뿐이고, Q2(SFT로 소형을 끌어올리기)가
성공해야만 CM4 제품이 성립한다.** 프리필도 잊지 말 것: 페르소나+few-shot+히스토리를 매 턴
프리필하는 현 구조는 CM4에서 그 자체로 수 초다 — SFT로 프롬프트를 가중치에 흡수하는 것이
품질 이전에 레이턴시 대책이다. (완화 수단: llama.cpp 프롬프트 캐시/세션 KV 재사용, 4스레드
고정, Q4_0 온라인 리팩(ARM 최적화) 등 — CM4 실측 때 함께 확인.)

### 5.6 성공 기준 (미리 고정)

- **품질**: SFT 소형(≤1.3B)이 midm-2.3b+프롬프트 대비 judge 승률 ≥45% **그리고** 기계 지표
  위반 0, 되묻기 ≥60%, 판박이 ≥95%, 길이 중앙값 30~60자.
- **속도**: CM4 실기에서 첫 문장 TTS 투입까지(llm ttfb + 첫 문장 완성) ≤2.5s [추정 목표 —
  실사용 체감으로 조정].
- **둘 다 통과 못 하면**: 크기를 한 단계 올리거나(1.7B), CM4를 포기하고 타깃 보드를 올리는
  결정을 데이터로 한다.

## 다음 단계

즉시 (실험 준비):

- [ ] **CM4 실기에서 현 프리셋 Q4 tok/s 실측** — §5.5의 추정 표를 실측으로 교체. 이 숫자가
      이 문서 전체의 분기점이다 (FEATURES.md의 기존 다음 단계와 동일 항목).
- [ ] Qwen3.5-0.8B/2B GGUF 프리셋 추가(`models.yaml`) 후 `_ab_persona.py` 통과시켜보기 —
      한국어 능력이 Qwen3 대비 좋아졌는지 먼저 확인. 라이선스는 모델 카드에서 직접 재확인.
- [ ] Kanana-2-1.3B/3B 내부 평가용 프리셋 추가(배포 불가 표기 필수 — models.yaml 주석에
      KananaOpenLicense 온디바이스 조항 명기).
- [ ] `_ab_persona.py`를 멀티턴 시나리오로 확장(§5.3) + 피크 메모리 측정 추가.

이후 (Q1 → Q2 순서):

- [ ] Q1: 사다리 전체를 확장된 프로토콜로 1회 완주 — "프롬프트만의 하한" 곡선 확보.
- [ ] LLM-as-judge 하네스(pairwise, 스왑, 축별 루브릭) 구축 + 사람 선호와의 상관 검증(20~30 페어).
- [ ] teacher 선정(약관 확인 기록 포함) → 5천 턴 파일럿 데이터셋 생성 → qwen3-0.6b(또는
      0.8b) QLoRA 1차 — "SFT 소형 vs midm-2.3b 프롬프트" 첫 비교.
- [ ] 결과에 따라: 성공 시 증량(2만 턴)+full FT 비교, 실패 시 1.3B급으로 상향 재시도.

하지 말 것 (선행 문서의 결론 유지 + 이번 리서치로 보강):

- 지식을 넣으려는 파인튜닝 — BlenderBot의 교훈(§2.2) 그대로, 환각만 는다.
- NC 모델(EXAONE/HyperCLOVA/Kanana 구버전)을 teacher로 쓰는 것 — 라이선스 오염(§5.4).
- duplex 네이티브로의 전환 — 7B+GPU는 CM4에서 성립 불가, 서버 레포(nobody-stream) 관심사(§4.2).
- 양자화 저장소의 라이선스 표기를 믿는 것 — Kanana-2도 보도("Apache-2.0 적용")와 실제
  LICENSE 파일(KananaOpenLicense, 온디바이스 별도 계약)이 달랐다. **원본 LICENSE 파일만 믿는다.**
