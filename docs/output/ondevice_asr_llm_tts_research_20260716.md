# 온디바이스 ASR+LLM+TTS 모델 리서치 — 최종 필터링

작성: 2026-07-16
필터 조건: **① 완전 오픈 라이선스만 (Apache-2.0/MIT/BSD/CC-BY류, NC 계열 전부 제외) ② 한국어 지원 확인된 것만**
타깃 하드웨어: 온디바이스 (4GB RAM, Cortex-A72 쿼드코어 1.5GHz, GPU 없음)
용도: 클라우드 STS(Gemini Live/OpenAI Realtime)를 로컬 파이프라인으로 완전 대체

---

## 0. 하드웨어 전제

타깃 온디바이스 하드웨어는 Raspberry Pi 4와 동급 CPU(Cortex-A72). 실측 벤치마크 기준 1B급 LLM이 4~12 tok/s. 페르소나 응답을 1~4문장(약 30~80 토큰)으로 강제하고 있어도 응답 생성에 5~20초가 걸릴 수 있음 — Gemini Live 평균 TTFA 797ms와 자릿수가 다름. 메모리도 ASR+LLM+TTS+기존 wakeword/화자인증 모델을 합치면 4GB 중 2.5~3.5GB를 쉽게 씀. 현재 Docker `mem: 512MB` 제한은 이 프로젝트에서 반드시 상향 필요. **모델 선정보다 온디바이스 실측 PoC가 선행되어야 함.**

---

## 1. ASR (로컬 STT)

| 모델 | 방식 | 라이선스 | 한국어 | 비고 |
|---|---|---|---|---|
| **SenseVoice-Small** (FunAudioLLM/Alibaba) | non-autoregressive, VAD-트리거 | Apache 2.0 | ✅ zh/en/yue/ja/ko 공식 지원 | Whisper-Large 대비 15배 빠름, INT8 양자화 ~234MB |
| **Vosk `vosk-model-small-ko-0.22`** (82MB) | Kaldi 하이브리드, 네이티브 스트리밍 | Apache 2.0 | ✅ 한국어 전용 모델 | 저장소 `vosk/`에 이미 다운로드/래퍼 존재 — 즉시 시도 가능 |
| **sherpa-onnx streaming Zipformer (한국어)** | 스트리밍 transducer, KsponSpeech 학습 | Apache 2.0 | ✅ 한국어 전용 | ⚠️ [GitHub #2886](https://github.com/k2-fsa/sherpa-onnx/issues/2886)에 "빈 transcription" 버그 리포트(2025-12) — 직접 검증 필요 |

**1차 추천**: SenseVoice-Small + Vosk small-ko를 온디바이스에서 나란히 실측 비교.

---

## 2. LLM (로컬 대화 엔진)

| 모델 | 파라미터 | 라이선스 | 한국어 | 판정 |
|---|---|---|---|---|
| EXAONE 4.0 1.2B (LG) | 1.2B, 온디바이스 전용 | ❌ EXAONE License 1.2-NC | 한국어 최우선 특화 | **제외** — 상업 이용 명시적 금지 |
| Kanana Nano 2.1B (Kakao) | 2.1B | ❌ CC-BY-NC-4.0 | 한국어 특화 | **제외** |
| Llama 3.2 1B (Meta) | 1B | ⚠️ Llama Community License(클릭스루+사용제한) | 공식 지원언어 미포함(en/de/fr/it/pt/hi/es/th) | **제외** — 완전 오픈 기준 미달 + 한국어 비공식 |
| Qwen2.5-Omni-3B | 3B | ❌ Qwen 전용 라이선스(3B/72B만 비-Apache) | 확인 안됨 | **제외** |
| **Qwen3-0.6B** | 0.6B | ✅ Apache 2.0 | 119개 언어 중 포함(전용 튜닝 아님) | **통과** — Q4_K_M ~1GB, Pi4급에서 구동 확인 |
| **Gemma 4 E2B** (Google) | raw ~5~6B / effective ~2B | ✅ Apache 2.0 (2026-04 전환) | 140+ 언어 네이티브 | **통과** — 오디오 입력 멀티모달 지원(ASR fused 가능), 단 온디바이스 메모리 실측 필요(Q4 기준 raw 파라미터 전량 로드 시 ~3~3.5GB) |

**결론**: 한국어 품질 최상위 모델(EXAONE/Kanana)은 전부 비상업 라이선스라 제외. 완전 오픈 중에서는 **Qwen3-0.6B**(안전한 기본값)와 **Gemma 4 E2B**(오디오 fused 가능성, 메모리 리스크)가 유력.

---

## 3. TTS (로컬 음성 합성)

| 모델 | 크기 | 라이선스 | 한국어 | 판정 |
|---|---|---|---|---|
| MMS-TTS 한국어 (facebook/mms-tts-kor) | VITS 기반 | ❌ CC-BY-NC 4.0 | ✅ | **제외** |
| Piper 한국어 (neurlang/piper-onnx-kss-korean) | 63.5MB | ❌ CC-BY-NC-SA-4.0 (KSS 데이터셋 상속) | ✅ | **제외** |
| **Kokoro-82M** | 82M | ✅ Apache 2.0 | ✅ 공식 지원 언어에 포함 | **통과** |
| **Kyutai Pocket TTS** | 100M | ✅ CC-BY-4.0 | ✅ 2026-05 다국어 업데이트로 추가 확인 | **통과** — CPU 실시간 전용 설계 |
| **MeloTTS-Korean** | VITS/VITS2/Bert-VITS2 기반 | ✅ MIT | ✅ 한국어 전용 체크포인트 | **통과** — BERT 프론트엔드라 상대적으로 무거울 수 있음 |
| **MOSS-TTS-Nano** (OpenMOSS/Fudan) | 100M | ✅ Apache 2.0 | ✅ 20개 언어 중 명시적 포함 | **통과, 최유력** — "4-core CPU, zero GPU 실시간" 공식 스펙이 타깃 온디바이스 쿼드코어 사양과 정확히 부합, 48kHz stereo, zero-shot voice cloning |
| **FreyaTTS (Korean fork, 이번 세션에서 직접 준비)** | 183M | ✅ Apache 2.0 (코드+AudioVAE2 둘 다) | 원본은 터키어 전용 → 이번 세션에서 한국어로 이식 준비 완료 (가중치는 미학습) | 아래 4절 참조 |

**갱신**: MOSS-TTS-Nano가 이번 대화 중 새로 발견됨 — 100M, Apache 2.0, 4코어 CPU 실시간, 한국어 명시 지원. 지금까지 나온 후보 중 **온디바이스 스펙(쿼드코어, GPU 없음)과 가장 정확히 일치하는 공식 사양**을 가진 유일한 모델. TTS 1순위 실측 대상으로 격상 권장.

---

## 4. Fused / S2S 모델 (요청에 따라 조사, cascade 아닌 통합형)

| 모델 | 유형 | 라이선스 | 한국어 | 온디바이스 판정 |
|---|---|---|---|---|
| Moshi (Kyutai) | 완전 S2S, full-duplex | ✅ CC-BY-4.0 | ❌ 미지원(en/fr만) | **제외** — 라이선스는 통과해도 한국어 없음 + 7B 백본 |
| Ultravox | ASR-LLM fused | ⚠️ 명시 안됨, Llama/GLM 백본 상속 가능성 | 불명확 | **제외** — 8B~355B 백본, 온디바이스 논외 |
| Qwen2.5-Omni-3B | 진짜 S2S | ❌ Qwen 전용 라이선스 | 불명확 | **제외** |
| Qwen3-Omni-30B-A3B | 진짜 S2S | ✅ Apache 2.0 | ✅ 한국어 음성 입출력 지원 | **제외** — MoE라도 전체 30B가 메모리 상주 필요, 온디바이스 완전 불가 |
| MiniCPM-o 2.6/4.5 | 진짜 Any-to-Any | ✅ Apache 2.0 | 텍스트/비전만 한국어 확인, **음성 출력은 en/zh bilingual만 명시** | **제외** — 음성 쪽 한국어 미확인 + 8~9B 크기 |
| **Gemma 4 E2B** | ASR-LLM fused (오디오 in → 텍스트 out) | ✅ Apache 2.0 | ✅ 140+ 언어 | **유일한 생존 후보** — 단 메모리 실측 필요 |

**결론**: fused/S2S 카테고리는 한국어+완전오픈+온디바이스 세 조건을 동시에 만족하는 모델이 **Gemma 4 E2B(오디오→텍스트만) 하나뿐**. 완전 S2S(오디오→오디오)는 현재 시점 온디바이스급에서 전멸 — 전부 7B 이상 백본이 필요하다는 게 업계 공통 패턴.

---

## 5. FreyaTTS 한국어 이식 (이번 세션 실제 작업)

`https://github.com/freyavoiceai/FreyaTTS` (183M, Apache-2.0, 터키어 전용 char-level TTS)를 리서치 과정에서 발견. 구조 검토 결과:

- **AudioVAE2**는 OpenBMB VoxCPM2에서 그대로 가져온 **언어 독립적 범용 코덱** (Apache-2.0) — 재학습 불필요
- 텍스트 인코딩이 **캐릭터 레벨(phonemizer/G2P 없음)** — 한글은 표기가 발음에 가까운 문자 체계라 이 방식에 원래 적합
- 심볼 vocab이 JSON 파일 하나(`char_vocab.json`)라 교체가 trivial

→ `https://github.com/ummjevel/FreyaTTS.git`를 클론해 `/mnt/c/Users/zoey/dev/FreyaTTS`에서 한국어 이식 완료:
- `freyatts/hangul.py`: 한글 자모 분해/조합 (순수 유니코드 연산, 초성19+중성21+종성전용11=자모 51개)
- `freyatts/char_vocab.json`: 127 심볼(특수2+구두점12+숫자10+라틴52+자모51)로 재생성
- `freyatts/pipeline.py`: 한자어/고유수사 숫자 읽기, 시각 표현("아홉시 이십팔분" 규칙), 영문 약어 한글 표기 변환
- `training/build_manifest_ko.py`: 원시 (wav, transcript) 코퍼스 → 학습용 manifest.jsonl 빌드 스크립트 신규 작성
- README/MODEL_CARD/eval 프롬프트까지 한국어로 업데이트, **단 실제 한국어 가중치는 미학습 상태**(학습 코퍼스 확보가 다음 단계)

**코퍼스 라이선스 가이드** (README에 명시):
| 코퍼스 | 라이선스 | 비고 |
|---|---|---|
| KSS | ❌ CC BY-NC-SA 4.0 | 가장 흔히 쓰이지만 상업용 금지 |
| Zeroth-Korean | ✅ CC BY 4.0 | 51.6h, 105화자(다화자) |
| Common Voice Korean | ✅ CC0 | 크라우드소싱, 품질 편차 |
| 자체 성우 녹음 | ✅ 완전 소유 | 가장 안전한 상업 경로 |

> **참고**: 이 리서치 세션 도중 같은 클론 경로(`/mnt/c/Users/zoey/dev/FreyaTTS`)에 내가 만들지 않은 변경(`requirements.txt`에 `uroman` 추가, `training/extract_short_segments.py`의 로마자 변환 로직을 `uroman` 기반으로 교체, `training/run_extract_short_ko.sh` 신규)이 발견됨 — 다른 세션이 동시에 같은 저장소를 작업 중일 가능성이 있어 사용자에게 별도 확인 필요.

---

## 6. 종합 추천 스택

| 구성 | 1순위 | 비고 |
|---|---|---|
| ASR | SenseVoice-Small | Vosk small-ko를 폴백으로 비교 |
| LLM | Qwen3-0.6B | Gemma 4 E2B(오디오 fused)와 비교 실측 |
| TTS | **MOSS-TTS-Nano** | 온디바이스 스펙과 정확히 일치하는 유일한 공식 사양. FreyaTTS(한국어 이식판, 가중치 학습 필요)와 실측 비교 |
| ASR+LLM 통합 | Gemma 4 E2B | 별도 TTS만 붙이면 캐스케이드 단계 하나 제거 가능 — 메모리 실측 최우선 |
| 완전 S2S | 없음 | 현재 온디바이스급에서 실전 후보 부재 |

**다음 단계**: 온디바이스에서 ① MOSS-TTS-Nano RTF, ② SenseVoice-Small + Qwen3-0.6B 캐스케이드 latency, ③ Gemma 4 E2B 오디오 입력 메모리/속도 세 가지를 우선 실측.

---

## Sources

- [Raspberry Pi 5 LLM Benchmarks 2026 — Local AI Master](https://localaimaster.com/blog/llm-raspberry-pi-5)
- [Running LLMs on Raspberry Pi 5 — TinyWeights](https://tinyweights.dev/posts/run-llms-raspberry-pi-5/)
- [An Evaluation of LLMs Inference on Popular SBCs (arXiv)](https://arxiv.org/html/2511.07425v1)
- [k2-fsa sherpa-onnx-streaming-zipformer-korean](https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16)
- [Korean streaming models empty transcription — Issue #2886](https://github.com/k2-fsa/sherpa-onnx/issues/2886)
- [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
- [SenseVoice vs Whisper benchmark](https://whispernotes.app/blog/sensevoice-fastest-cjk-transcription)
- [Vosk models list — alphacep/vosk-space](https://github.com/alphacep/vosk-space/blob/master/models.md)
- [LGAI-EXAONE/EXAONE-4.0-1.2B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B)
- [EXAONE-3.0 LICENSE](https://github.com/LG-AI-EXAONE/EXAONE-3.0/blob/main/LICENSE)
- [kakaocorp/kanana-nano-2.1b-instruct](https://huggingface.co/kakaocorp/kanana-nano-2.1b-instruct)
- [Kanana GitHub](https://github.com/kakao/kanana)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen 3 Full Lineup Guide 2026](https://baeseokjae.github.io/posts/qwen-3-full-lineup-guide-2026/)
- [Qwen2.5-3B switch to Apache 2.0 commit (shows 3B was previously non-Apache)](https://huggingface.co/Qwen/Qwen2.5-3B/commit/839823b867963f5234bed65a7af47fccee77b2ad)
- [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- [Google releases Gemma 4 under Apache 2.0 — VentureBeat](https://venturebeat.com/technology/google-releases-gemma-4-under-apache-2-0-and-that-license-change-may-matter-more-than-benchmarks)
- [9to5google: Gemma 4 Apache 2.0](https://9to5google.com/2026/04/02/google-gemma-4/)
- [Gemma 4 model card — Google AI](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 audio encoder E2B/E4B — MindStudio](https://www.mindstudio.ai/blog/gemma-4-audio-encoder-e2b-e4b-speech-recognition)
- [Gemma 4 E2B vs E4B edge models — MindStudio](https://www.mindstudio.ai/blog/gemma-4-e2b-vs-e4b-edge-models-audio-vision-phone)
- [Gemma 3n E2B specs — apxml](https://apxml.com/models/gemma-3n-e2b-it)
- [Moshi — Kyutai GitHub](https://github.com/kyutai-labs/moshi)
- [Moshi CC-BY-4.0 license — Local AI Master](https://localaimaster.com/blog/moshi-realtime-speech-guide)
- [Ultravox GitHub](https://github.com/fixie-ai/ultravox)
- [fixie-ai/ultravox-v0_7-glm-4_6 (355B backbone)](https://huggingface.co/fixie-ai/ultravox-v0_7-glm-4_6)
- [OpenMOSS/MOSS-TTS-Nano GitHub](https://github.com/OpenMOSS/MOSS-TTS-Nano)
- [OpenMOSS/MOSS-TTS GitHub](https://github.com/OpenMOSS/MOSS-TTS)
- [MOSS-TTS LICENSE (Apache-2.0)](https://github.com/OpenMOSS/MOSS-TTS/blob/main/LICENSE)
- [openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6)
- [openbmb/MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- [OpenBMB MiniCPM-o 2.6 announcement — MarkTechPost](https://www.marktechpost.com/2025/01/14/openbmb-just-released-minicpm-o-2-6-a-new-8b-parameters-any-to-any-multimodal-model-that-can-understand-vision-speech-and-language-and-runs-on-edge-devices/)
- [Pocket TTS now supports six languages incl. Korean — Kyutai blog](https://kyutai.org/blog/2026-05-04-pocket-tts-multilingual/)
- [Pocket TTS: CPU-only TTS — Kyutai blog](https://kyutai.org/blog/2026-01-13-pocket-tts/)
- [kyutai/tts-1.6b-en_fr license cc-by-4.0](https://huggingface.co/kyutai/tts-1.6b-en_fr)
- [Kokoro TTS languages incl. Korean — Soniqo guide](https://soniqo.audio/guides/kokoro)
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- [myshell-ai/MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean)
- [MeloTTS GitHub](https://github.com/myshell-ai/MeloTTS)
- [facebook/mms-tts-kor (CC-BY-NC 4.0)](https://huggingface.co/facebook/mms-tts-kor)
- [neurlang/piper-onnx-kss-korean (CC-BY-NC-SA-4.0)](https://huggingface.co/neurlang/piper-onnx-kss-korean)
- [Zeroth Korean — OpenSLR](https://openslr.org/40/)
- [torchaudio MMS_FA multilingual forced alignment docs](https://docs.pytorch.org/audio/2.9.0/tutorials/forced_alignment_for_multilingual_data_tutorial.html)
- [FreyaTTS upstream — freyavoiceai/FreyaTTS](https://github.com/freyavoiceai/FreyaTTS)
- [FreyaTTS technical report (arXiv:2607.09530)](https://arxiv.org/abs/2607.09530)
