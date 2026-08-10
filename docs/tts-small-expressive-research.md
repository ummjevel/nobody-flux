# 작은데 표현력 좋은 TTS — 모델·기법·아키텍처 리서치 (참고용)

> **TTS 문서 3부작 — 이 문서 = ② 서베이.**
> ① `tts-expressivity-design.md` — 전략(온디바이스 제약·Path A/B·한국어 NVV 데이터 공백)
> ② `tts-small-expressive-research.md` — 소형 표현력 TTS 모델·기법 서베이 ← 여기
> ③ `tts-conversational-build-design.md` — 대화체 소형 TTS 구축 설계 + 실측 교훈

"온디바이스(CM4, GPU 없음)에서 도는 작은 모델에 어떻게 표현력(감정·자연스러운 운율·NVV)을
넣나"를 **모델 / 기법 / 아키텍처 / 데이터** 네 축으로 정리했다. 목적: FreyaTTS(flow-matching)에
얹을지, 경량 모델을 새로 만들지 판단할 근거.

## 핵심 결론 (먼저)

> **NVV(웃음·한숨·숨)는 큰 AR codec-LM에만 있고(전부 GPU), CPU 실시간은 작은 VITS/flow
> 모델에만 있다(전부 밋밋함). 오늘날 둘 다 주는 기성품은 없다.** 그래서 CM4에서 표현력을
> 얻으려면 "작은 flow 모델을 조건화 + NVV 토큰으로 확장하고, 표현력 있는 서버 teacher로
> 데이터를 만들어 학습"하는 길밖에 없다.

두 독립 리서치가 같은 결론에 도달: **FreyaTTS(flow-matching)에 얹는 게 정답. StyleTTS2를 새로
짓지 마라. GPU NVV 모델을 디바이스에 올리지 마라.** FreyaTTS는 이미 가장 어려운 것(자연스러운
운율)을 갖고 있고, flow-matching field 위에 표현력을 얹는 건 가산적(additive)이며, CPU/ONNX/
스트리밍 친화적이라 조건화가 가장 쉽다.

참고: "FreyaTTS에 얹기"는 사실상 **"내 모델을 표현력 있게 키우기"**다 — FreyaTTS가 이미 자체
모델이므로. 완전히 새 아키텍처로 처음부터 짓는 건 flow-matching 베이스가 근본적으로 막힐 때만
의미 있는데, flow-matching이 바로 권장 아키텍처라 그럴 이유가 약하다.

## 1. 소형 CPU TTS 모델 지형

아키텍처 계열이 param 수보다 중요한 필터다. **VITS-GAN / flow-matching = 1~few forward pass =
CPU 실시간.** **neural-codec-LM / autoregressive = 초당 수백 코덱 토큰 = 사실상 GPU** — 그런데
NVV를 실제로 내는 건 이 무거운 AR 쪽뿐.

| 모델 | 크기 | 계열 | CPU 실시간 | 표현력 | NVV | 라이선스 | 한국어 |
|---|---|---|---|---|---|---|---|
| Piper (VITS) | ~15–20M, ~60MB | VITS-GAN | ✅ (Pi4용 설계) | flat | ❌ | MIT | 네이티브 KO |
| Kokoro-82M | 82M | StyleTTS2계(GAN) | ✅ ~6×RT | 자연,약한 스타일 | ❌ | Apache-2.0 | KO 약함(G2P 빈약) |
| **Matcha-TTS**(현 baseline) | ~18M+voc | flow-matching | ✅ RTF~0.02 | flat~중 | ❌ | MIT | sherpa KO 체크포인트 |
| VITS/VITS2 | ~30M | VITS-GAN | ✅ | 중(Style-BERT-VITS2로 감정) | ❌ | MIT | 강한 KO 생태계 |
| **MeloTTS-Korean** | VITS계 | VITS-GAN | ✅ 실시간 | 중 | ❌ | MIT | **네이티브 KO** |
| **Supertonic 3** | 99M ONNX | flow/GAN on-device | ✅ RTF~0.012(초고속) | expression 태그(v3) | 부분(태그) | 소스공개(확인) | 31개어 **KO**(Supertone=한국) |
| KittenTTS Nano | 15M, <25MB | VITS계 | ✅ "감자에서도" | flat | ❌ | Apache-2.0 | 영어만 |
| Picovoice Orca | 7MB | on-device 신경망 | ✅ 최고급 | 중 | ❌ | **상용/클로즈드** | 확인 필요 |
| StyleTTS2 | ~100M+ | style-diffusion+adversarial | △ CPU RTF 높음 | 높음(감정) | ❌(스타일만) | MIT | KO PL-BERT+파인튜닝 필요 |
| GPT-SoVITS v2/v3 | ~100M+GPT | GPT(AR)+VITS | △ RTF 0.526(M4) | 높음,클로닝 | 일부 | MIT | **네이티브 KO(v2)** |
| Fish/OpenAudio S1 | 400M–4B | codec AR | ❌ GPU | 최고(OSS) | ✅ | CC-BY-NC/혼합 | KO 2급 |
| Parler-TTS Mini | 880M | codec-LM AR | ❌ 느림 | 스타일 프롬프트 | ❌ | Apache-2.0 | 영어만 |
| OuteTTS 1.0-1B | 1B GGUF | Llama codec-LM | △ llama.cpp CPU 느림 | 중,클로닝 | 제한 | Apache/CC | KO 부분 |
| **Orpheus-3B** | 3B GGUF | Llama codec-LM(SNAC) | ❌ GPU/8GB | 매우높음 | ✅ **`<laugh><sigh><yawn><gasp>` 학습** | Apache-2.0 | 영어(KO 파인튜닝) |
| **Dia-1.6B** | 1.6B | codec-LM AR | ❌ GPU | 매우높음 | ✅ **가장 풍부 `(laughs)(sighs)`** | Apache-2.0 | 영어 |
| Sesame CSM-1B | 1B | codec-LM AR | ❌ GPU | 높음(대화) | 제한(깔끔한 태그X) | Apache-2.0 | 영어 |

**NVV 되는 건**: Orpheus·Dia·Fish/OpenAudio S1뿐 — 전부 1.6B~4B GPU codec-LM. → 디바이스엔 못
쓰고 **서버측 teacher**로만. **CPU 실시간 소형은 전부 NVV 없음.**

**한국어 CPU 실시간 후보 shortlist**: ① MeloTTS-Korean(네이티브, 안전한 baseline, 밋밋),
② Supertonic 3(초고속+expression 태그, 한국 벤더), ③ Style-BERT-VITS2 KO 파인튜닝(감정 임베딩,
CPU 가능), ④ **FreyaTTS/Matcha 개선(현 방향, 아키텍처적으로 맞음)**, ⑤ Kokoro(영어 표현력 좋으나
KO G2P 약함).

## 2. 표현력 기법 (CPU 실현성)

| 기법 | 원리 | CPU 비용 | 데이터 | 표현력 |
|---|---|---|---|---|
| **GST / 레퍼런스 인코더** | 비지도 스타일 임베딩 뱅크 + 레퍼런스 mel에 attention. 추론 시 토큰 선택만으로 스타일 제어(레퍼런스 불요) | 매우 저렴(작은 conv+GRU) | **비지도**, 라벨 불요 | 중(굵은 운율/감정) |
| **FastSpeech2 variance adaptor** | pitch/energy/duration 명시 예측·주입, 추론 시 직접 편집 | 가장 저렴, 결정론적 | pitch/energy 추출(무료) | 제어성↑, 자연스러움은 다소 flatten |
| **감정/화자 임베딩** | 룩업/학습 벡터를 조건에 concat | 사소 | 감정 라벨(KO: KESDy18, AI-Hub) | 카테고리 감정 good, fine-grained/NVV poor |
| **다중스케일 운율 예측(MsEmoTTS)** | 발화/단어/음소 계층 감정 예측·전이 | 중(CPU ok) | 라벨+레퍼런스 | fine-grained 강함 |
| **StyleTTS2 스타일-디퓨전+adversarial** | 스타일을 작은 디퓨전으로 샘플 + speech-LM 판별자 | 추론 82M ONNX-CPU 가능하나 디퓨전(~5스텝)이 지연 지배, **스트리밍X**; 학습 불안정/OOM | 대형 다화자 코퍼스 | 소형 중 최고 자연스러움, but 재현 최난 |
| **flow-matching 조건화(Matcha/FreyaTTS)** | 전역+국소 스타일 벡터로 OT-CFM 벡터필드 변조; frozen flow 위 ControlNet 어댑터로 시변 감정 | ODE 샘플링이 비용(Matcha는 few-step CPU); 조건화 추가 비용 거의 0 | 스타일 벡터는 자기지도(GST식) 또는 라벨 | 확장성 검증됨(DARS, ControlNet-over-flow) |

**CPU 최저비용·고제어**: variance adaptor + GST. **param당 최고 자연스러움**: StyleTTS2(단, 학습
고통 + 스트리밍 없음).

## 3. NVV 모델링

합의: **NVV를 인라인 특수토큰(`[웃음]`/`[한숨]`)으로** 텍스트/음소 스트림에 넣고, 모델 vocab의
추가 심볼로 취급. ASR·TTS 동일 토큰화 → 문맥 인지 **배치(placement)**를 학습해 임의 위치에서
발화.

- **NVSpeech (arXiv:2508.04195)** — 가장 직접적인 청사진. 18개 단어단위 paralinguistic 토큰,
  파이프라인 = 인간주석 48k발화 → ASR 부트스트랩으로 174k발화/573h 자동라벨 → **zero-shot TTS
  파인튜닝**으로 위치 제어 NVV 삽입. 중국어지만 레시피는 언어 불문 → **한국어로 복제할 패턴.**
- NonverbalTTS(arXiv:2507.13155) — 최대 영어 NVV 코퍼스(10종), 증강/전이 소스.
- NV-Bench(arXiv:2603.15352)/Affectron — 평가 + NV 유형·삽입위치 분포 확장(“한 번만 말고
  자연스럽게 여러 번”).
- Beyond-Words ASR(arXiv:2607.01563) — 인라인 토큰 ASR 프레이밍 + 저자원 전략.

**저자원(한국어 NVV 데이터 없음)**: 부트스트랩 라벨링이 핵심 — NVV 인식 ASR을 학습/차용해
한국어 자발적 코퍼스를 자동 태깅 후 파인튜닝. + 커리큘럼(중립→NVV) + NVV 음향의 언어 간 전이
(웃음/숨은 거의 언어 독립).

## 4. 데이터·증류 전략

- **표현력 서버 teacher → CPU 학생**: **CosyVoice3(arXiv:2505.17589)/CosyVoice2** — `[laughter]`/
  `[breath]` 태그 + 5000h instruction(100+ 감정/스타일). 이걸로 **NVV·감정 풍부한 한국어 코퍼스를
  생성**해 작은 학생 학습. Qwen3-TTS로는 못 가르친다는 기존 결론을 우회 — *표현력을 가진* teacher를
  고르면 됨. 단어단위 감정 teacher→student 검증됨(arXiv:2509.24629).
- **VC / pitch-shift 증강**: 중립 타깃화자 데이터만 있을 때 F0 조건 VC·피치시프트로 ~1h에서 표현
  음성 구축(arXiv:2204.10020, 2207.14607, Amazon 2011.05707).
- **한국어 자발적 코퍼스**: AI-Hub, **CoreaSpeech 700h(NeurIPS 2025)**, KESDy18 — 숨/비유창성 풍부한
  오디오에서 NVV 채굴. CoreaSpeech의 JAMO G2P 접근은 FreyaTTS 발음 개선에도 참고.

## 5. 아키텍처 트레이드오프 (자체 구축 시)

- **VITS2/Piper**: 초소형, RTF~0.008, <100MB, 견고한 ONNX. but end-to-end adversarial → **운율
  제어 어렵고 표현력 천장 낮음.** 좋은 baseline, 나쁜 표현 상한.
- **Matcha/flow-matching**: VITS2보다 억양/명료도↑, ONNX-CPU, few-step ODE. **조건화가 깔끔**
  (전역+국소 스타일, ControlNet 어댑터). 디퓨전보다 스트리밍 친화. **노력 대비 제어성 최고.**
- **StyleTTS2-lite**: 82M 최고 자연스러움, ONNX-CPU 가능, but 학습 불안정·스트리밍 없음·디퓨전
  지연. 새로 짓기엔 고위험.
- **NIX-TTS(arXiv:2203.15643)**: 모듈별 증류로 초소형 end-to-end 압축 — 어떤 베이스든 압축 기법으로
  유용.

## 권장 레시피 (FreyaTTS에 얹기)

**FreyaTTS(flow-matching)에 얹어라. StyleTTS2를 새로 짓지 마라.** FreyaTTS는 가장 어려운
자연스러운 운율을 이미 확보했고, flow field 위 표현력은 가산적이며, CPU/ONNX/스트리밍 친화 +
조건화 최용이.

최소 레시피:
1. **토큰/음소셋 확장** — 인라인 `[웃음]`/`[한숨]`/… (NVSpeech 패턴).
2. **경량 조건화 스택** — GST식 전역 스타일 벡터 + FastSpeech2식 variance adaptor(pitch/energy).
   둘 다 추론 시 비용 거의 0, 감정·운율 핸들 제공. (선택) frozen flow decoder 위 ControlNet
   어댑터로 시변 감정.
3. **데이터** — CosyVoice3를 teacher로 NVV+감정 한국어 코퍼스 생성; NVV 인식 ASR 부트스트랩으로
   AI-Hub/CoreaSpeech 자발적 음성 자동라벨; VC/pitch-shift 증강. **중립→표현 커리큘럼.**
4. **발음(별개 트랙)** — FreyaTTS 약한 발음은 표현력과 분리해서 한국어 G2P/자소 처리 개선
   (CoreaSpeech JAMO 접근).

→ ONNX-CPU 배포 가능, 제어 가능, 표현력 있는 한국어 TTS를 **최저 학습 위험**으로. StyleTTS2가
자연스러움은 약간 위지만 학습 불안정·스트리밍 없음·디퓨전 지연으로 CM4엔 부적합.

## "새로 만들기 vs 얹기" 판단

- 사용자가 자체 모델의 소유·학습 가치를 중시 → **좋은 소식: FreyaTTS 자체가 이미 자체 모델**이라
  얹기 = 내 모델 키우기. 조건화 스택/NVV 토큰/데이터 파이프라인이 그 "새로 만드는 노력"의 실체.
- 완전 새 아키텍처(예: VITS2 처음부터)는 flow-matching 베이스가 근본적으로 막힐 때만 — 지금은
  근거 약함. 굳이 한다면 아키텍처는 **flow-matching(Matcha류)** 권장, 위 레시피 동일 적용.

## 참고 문헌

- 모델: [Piper](https://pidiylab.com/text-to-speech-raspberry-pi-piper/) · [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)/[ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) · [Matcha-TTS](https://arxiv.org/pdf/2309.03199) · [MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean) · [Supertonic](https://github.com/supertone-inc/supertonic)/[v3](https://www.marktechpost.com/2026/05/15/supertone-releases-supertonic-v3-on-device-text-to-speech-model-with-31-language-support-fewer-reading-failures-and-expression-tags/) · [KittenTTS](https://github.com/KittenML/KittenTTS) · [StyleTTS2](https://github.com/yl4579/StyleTTS2) · [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) · [Fish-Speech/OpenAudio](https://github.com/fishaudio/fish-speech) · [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) · [Dia](https://github.com/nari-labs/dia) · [Picovoice on-device TTS 벤치](https://picovoice.ai/blog/on-device-tts/) · [Kokoro vs Supertonic CPU 벤치](https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts)
- 기법: [GST(1803.09017)](https://arxiv.org/abs/1803.09017) · [MsEmoTTS(2201.06460)](https://arxiv.org/pdf/2201.06460) · [StyleTTS2(2306.07691)](https://arxiv.org/abs/2306.07691) · [DiffStyleTTS(2412.03388)](https://arxiv.org/html/2412.03388v1) · [Hierarchical Emotion Rendering(2412.12498)](https://arxiv.org/pdf/2412.12498) · [NIX-TTS(2203.15643)](https://arxiv.org/pdf/2203.15643)
- NVV: [NVSpeech(2508.04195)](https://arxiv.org/abs/2508.04195) · [NonverbalTTS(2507.13155)](https://arxiv.org/pdf/2507.13155) · [NV-Bench(2603.15352)](https://arxiv.org/html/2603.15352) · [Beyond-Words ASR(2607.01563)](https://arxiv.org/html/2607.01563)
- 데이터/증류: [CosyVoice3(2505.17589)](https://arxiv.org/html/2505.17589v1) · [word-level emotion teacher→student(2509.24629)](https://arxiv.org/pdf/2509.24629) · [F0-VC 증강(2204.10020)](https://arxiv.org/pdf/2204.10020) · [pitch-shift 증강(2207.14607)](https://arxiv.org/abs/2207.14607) · [Amazon 저자원 표현 TTS(2011.05707)](https://arxiv.org/pdf/2011.05707) · [CoreaSpeech(NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/121811)
