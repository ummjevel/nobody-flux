# Voice Agent 오픈소스 지형 조사 & 분석

> voice agent 오픈소스를 전수 조사해 **아키텍처 / 장점 / 단점·한계 / 한계 돌파법 / 우리가 얻어갈
> 것**을 정리한 문서. 우리 관점 = **완전 로컬 온디바이스 한국어 캐스케이드(VAD→ASR→LLM→TTS),
> CM4급 CPU-only 타깃.** 이미 barge-in/backchannel/Smart Turn 엔드포인트/메모리 구현됨.

## 0. 핵심 결론 (먼저)

조사 전체를 관통하는 결론 다섯 가지:

1. **Smart Turn v3는 우리에게 정답이었다 (이미 도입함).** 8MB int8 ONNX, CPU 12ms, 한국어 포함
   23개어, 오디오 네이티브 semantic end-of-turn. 세 리서치 모두 이걸 "CM4 캐스케이드용 최선의
   엔드포인터"로 꼽음. 우리 `turn_detector.py`가 이미 이것 → 선택이 검증됨.
2. **엔드포인팅 지연을 "고정값"이 아니라 "적응형"으로.** 업계 공통 승리 패턴(LiveKit EOU, Kyutai
   semantic VAD, Smart Turn): 작은 모델이 내용·운율을 보고 **침묵 타임아웃을 동적으로 늘렸다
   줄임.** 우리의 다음 개선 1순위.
3. **턴을 3-상태(finished / unfinished / wait)로 모델링** (TEN Framework). 이진 end-of-turn보다
   깔끔하고, "문장 중간 멈춤(unfinished)"·"잠깐만(wait)"이 우리 barge-in/backchannel에 직결.
4. **GLaDOS가 턴테이킹/barge-in 레퍼런스 구현.** 언어 불문이라 한국어에 바로 이식 — 특히 문장
   단위 스트리밍 TTS + "끼어들면 재생 중단하고 응답을 히스토리에서 잘라내기".
5. **한국어 TTS가 최대 리스크(재확인).** CPU + 상업 라이선스 한국어 신경망 TTS는 기성품이 없음
   (Piper 한국어 없음, XTTS는 GPU+비상업, Supertonic GPU) → 우리 TTS 문서(`tts-*.md`) 결론과 일치.

## 1. 지형 개관 (5개 카테고리)

| 카테고리 | 대표 | 우리와의 거리 |
|---|---|---|
| 캐스케이드 오케스트레이터 | Pipecat, LiveKit Agents, HF speech-to-speech, Vocode | **가장 가까움** — 우리가 바로 이 구조 |
| 자가호스팅·온디바이스 비서 | HA Assist+Wyoming, OVOS, GLaDOS, Rhasspy, Willow | **CM4 배포 패턴의 교과서** |
| realtime STT/TTS 툴킷·로컬 서버 | RealtimeSTT/TTS, speaches, whisper_streaming, WhisperLive | 스트리밍/레이턴시 부품 |
| 풀-듀플렉스 스피치 네이티브 | Moshi/Unmute, Ultravox, TEN, Hertz-dev, Freeze-Omni, CSM | **GPU 전용 = 손 못 댐**, 아이디어만 |
| 프로토콜·wake-word tier | Wyoming, openWakeWord/microWakeWord, Porcupine | 구조·2단 컴퓨트 아이디어 |

## 2. 캐스케이드 오케스트레이터 (우리 구조와 동일)

### Pipecat (Daily) — 가장 가까운 아키텍처 아날로그 · Apache-2.0
- **아키텍처**: 에이전트를 **frame processor 파이프라인**(transport→VAD→STT→LLM→TTS→transport)으로.
  프레임이 스테이지 간 스트리밍돼 부분 transcript가 LLM에, LLM 토큰이 TTS에 흘러 **스테이지가
  겹침(overlap)**. transport 무관(WebRTC/WS/로컬).
- **장점**: 음성 특화 배관(backpressure, barge-in, endpointing, 문장 단위 TTS)이 기본 내장. 완전
  자가호스팅 가능하고 **오케스트레이터는 CPU에서** 돌며 무거운 모델만 따로 스케일. Smart Turn v3 탑재.
- **단점/한계**: Python async 오버헤드. STT/LLM/TTS 백엔드가 강력(보통 GPU/클라우드)하다고 가정.
  한국어 모델은 직접 공급해야 함.
- **돌파법**: ① 스트리밍 overlap으로 TTFB↓ ② **VAD 세그먼트 후 STT**(부분 transcript 오버헤드 회피)
  ③ TTS 스트리밍 출력(Kokoro 등)으로 첫 바이트↓ ④ barge-in = VAD 감지 시 in-flight TTS 취소 +
  backchannel 필터. Modal 사례: CPU 오케스트레이터+GPU 서비스로 **~1초 voice-to-voice**.
- **우리가 얻어갈 것**: **이게 곧 우리 아키텍처.** frame overlap, VAD-세그먼트-후-STT(CPU에서
  스트리밍 STT 복잡도 회피), TTFB용 스트리밍 TTS, CPU-오케스트레이터 패턴 — 전부 이식 가능.

### Smart Turn v3 (Pipecat/Daily) — 가장 직접 재사용 · Apache-2.0
Whisper-tiny 인코더 + 선형 head = **8M 파라미터, 8MB int8(QAT), ONNX. CPU 12ms**(싼 인스턴스 ~60ms).
**한국어 포함 23개어.** raw waveform에서 turn-complete 예측(transcript 불요), Silero VAD와 ≤8초 청크로
병용. **완전 로컬 CPU.** → **우리 CM4 엔드포인팅의 드롭인. (이미 `turn_detector.py`로 도입됨.)**

### LiveKit Agents — Apache-2.0 (단 WebRTC 인프라 가정)
- **턴테이킹(핵심 가치)**: **transformer EOU 모델**(135M, SmolLM v2 파인튜닝, CPU ~50ms)이 **최근
  4턴 transcript**를 읽어 **VAD 침묵 타임아웃을 동적으로 단축/연장** → VAD-only 대비 오끼어듦 85%↓.
  **현재 영어 전용.**
- **단점**: WebRTC 중심(단일 온디바이스 로봇엔 과함). EOU가 텍스트 기반(영어) → 한국어엔 Smart
  Turn의 오디오 네이티브가 더 나음.
- **우리가 얻어갈 것**: **"모델이 VAD 타임아웃을 동적 조정"이 가장 값진 턴테이킹 아이디어.** 단
  구현은 텍스트 EOU 대신 오디오 네이티브·한국어 되는 Smart Turn으로.

### HuggingFace speech-to-speech — 최고의 오프라인 레퍼런스 구현 · Apache-2.0
- **아키텍처**: 모듈형 **VAD(Silero v5)→Whisper STT→아무 HF LLM→TTS(Parler/MeloTTS/ChatTTS)**,
  OpenAI-Realtime 호환 WebSocket. **완전 로컬.** Reachy Mini 로봇 수천 대 구동.
- **우리가 얻어갈 것**: 전부 오픈웨이트 캐스케이드가 **임베디드 로봇에서 검증** — 우리와 가장 가까운
  배포 아날로그. (한국어는 Whisper/TTS 선택에 의존.)

### Vocode — MIT (단 코어 유지보수 정체)
BYO STT/LLM/TTS 스트리밍. 주목: **`interrupt_sensitivity` low/high** — low는 말하는 중 맞장구
("uh-huh")를 무시(우리 backchannel 필터와 동형). **우리가 얻어갈 것**: 우리 backchannel 설계가
업계에서도 쓰는 패턴임을 검증. 단 유지보수 정체라 의존성으론 부적합.

### Bolna / Dograh / 기타 (텔레포니·노코드) — 온디바이스 관련성 낮음
Bolna(텔레포니 우선), Dograh(Vapi/Retell 대체, 워크플로우). 클라우드/텔레포니 형태라 우리와 멀다.
그 외 발견: **FastRTC**(Python 함수를 VAD/턴테이킹 내장 WebRTC 스트림으로), **flowcat**(Rust 단일
바이너리, 에어갭), **LLMRTC**(서버측 barge-in). FastRTC는 로컬 CPU 배포용으로 지켜볼 만함.

## 3. 자가호스팅·온디바이스 비서 (CM4 배포 교과서)

### GLaDOS (dnhkng) — 우리 턴테이킹/barge-in 레퍼런스 · MIT
- **아키텍처**: 저지연 대화 companion. **~600ms 왕복 목표.** 순환 오디오 버퍼 + **Silero VAD**(32ms
  청크, prob>0.8, 800ms pre-roll, 640ms 종료 침묵). STT=Parakeet TDT(ONNX 스트리밍), TTS=Kokoro,
  LLM=OpenAI 호환(로컬 Ollama). **barge-in: 사용자 발화가 즉시 재생 중단 + 응답을 히스토리에서 잘라냄.**
- **장점**: 오픈 중 최고 수준 turn-taking/interruptibility. 스트리밍 ASR + 문장 단위 TTS로 레이턴시↓.
  ONNX 양자화. CPU-only(느림) 또는 RK3588 NPU SBC.
- **단점**: 영어 중심(**한국어 없음**), 600ms엔 GPU/NPU 권장, wake word 없음(상시).
- **우리가 얻어갈 것**: **우리 barge-in의 성숙판.** VAD 임계값·pre-roll 버퍼·문장 스트리밍 TTS·"끼어들면
  응답 clip" 로직이 언어 불문 → 그대로 이식. (우리는 barge-in 있으나 문장단위 스트리밍 TTS·clip은 아직.)

### Home Assistant "Assist" + Wyoming + wyoming-satellite — Apache-2.0 · 매우 활발
- **아키텍처**: **Wyoming** = 이벤트 기반 TCP 프로토콜(start/chunk/stop). wake-word/STT/TTS 서비스를
  믹스매치. 위성이 wake word를 돌려 **활성화 이후 오디오만 전송**. 서버는 Whisper+Piper+openWakeWord.
- **장점**: 깔끔한 프로토콜 분리, 온디바이스 wake-word 게이팅, 완전 로컬, 거대·건강한 커뮤니티.
  **Wyoming 프로토콜이 여기서 가장 채택할 만한 아이디어** — 오디오 장치↔추론 서비스 간 안정적 계약.
- **단점**: Assist는 명령/인텐트 지향(대화형 companion 아님, LLM은 bolt-on). 분리 구조는 별도 상시
  서버 가정. **Piper 한국어 음성 없음.**
- **우리가 얻어갈 것**: **Wyoming의 start/chunk/stop 스트리밍 이벤트를 내부 오디오 계약으로 채택** —
  올인원이어도. CM4가 약하면 satellite-server 분리로 탈출할 여지.

### OpenVoiceOS (OVOS) + Neon AI — Apache-2.0 · 활발
- Mycroft 후계, **완전 모듈·플러그인**. Pi/데스크톱/서버, 오프라인 가능. STT 플러그인 풍부: **Vosk
  (한국어 지원·스트리밍)**, faster-whisper(CPU/GPU), whisper.cpp, 신규 ONNX-ASR. `raspOVOS` Pi 이미지.
- **우리가 얻어갈 것**: 우리가 원하는 것과 가장 가까운 **완전 로컬 Pi 지향 프레임워크.** 플러그인 스왑
  패턴(우리 프리셋 구조와 유사)과 **Vosk 한국어 스트리밍 STT**를 참고.

### Rhasspy 2.5 (레거시) · Willow(HeyWillow) · ESPHome Voice PE
- **Rhasspy**: Wyoming의 조상, MQTT 위성+베이스. **grammar 인텐트**(LLM 없이 흔한 명령을 지연 0으로).
  → 우리도 흔한 명령은 grammar 패스트패스로 LLM 건너뛰기 가능. 지금은 사실상 레거시.
- **Willow**: ESP32-S3-BOX가 DSP로 **50~80mW 상시 wake word**. 단 WIS(Whisper 서버)는 **≥4GB RAM +
  사실상 GPU** → 무거운 tier는 CM4급 아님. 2023 열기 이후 모멘텀 식음.
- **ESPHome Voice PE + microWakeWord**: **$5 MCU에서 wake word** 구동 증명(단 ESP32-S3 필요).
  → 상시 청취 tier를 CM4에서 떼어내는 모델.

### june / Leon / Dicio
- **june**: Ollama+Whisper+Coqui 최소 캐스케이드 = 우리가 만드는 그 루프의 가장 읽기 쉬운 레퍼런스.
- **Dicio(Android)**: **Vosk-small(~50MB) 온디바이스** = 제약 ARM에서 STT 가능 증명(CM4 STT tier 선례).
- **Leon**: 오프라인 모델(Coqui STT/Flite)이 낡음, 참고도 낮음.

### wake-word tier
**openWakeWord**(Apache, CPU 저지연, HA/Wyoming 표준) 권장. Porcupine은 상업 라이선스 주의.

## 4. realtime STT/TTS 툴킷 · 로컬 서버 (스트리밍 부품)

- **whisper_streaming (LocalAgreement)** — MIT · **가장 이식성 높은 스트리밍 ASR 트릭.** 연속 Whisper
  실행 간 **최장 공통 접두사 합의**로 토큰 커밋, **자기적응 지연**(모델 불확실성에 비례), confidence>0.95
  즉시 커밋. → **스트리밍 네이티브 모델 없이 CPU에서 안정적 부분 transcript.** 우리 streaming-zipformer를
  진짜 스트리밍으로 쓸 때의 알고리즘.
- **RealtimeSTT/TTS/VoiceChat (KoljaB)** — MIT · 2단 VAD(WebRTC 게이트→Silero 확인)로 **침묵 즉시
  전사 중지**(낭비 계산↓), 토큰스트림→TTS 체이닝(TTFA↓), 동적 침묵 임계값. GPU 권장이나 패턴은 CPU 이식.
- **speaches** — MIT · OpenAI-Realtime 호환 로컬 서버(faster-whisper+piper/Kokoro). "STT/TTS의 Ollama."
  → **OpenAI-Realtime 호환 WS를 안정적 인터페이스로** 삼는 아이디어 + faster-whisper CPU 베이스라인.
- **WhisperLive**(OpenVINO = 좋은 CPU 경로), **Wyoming faster-whisper/piper**, **LocalAI**(GPU 불요
  오케스트레이터), **VoiceStreamAI**(레퍼런스) — 전부 로컬/CPU 지향.

## 5. 풀-듀플렉스·스피치 네이티브 (GPU 전용, 아이디어만)

- **Moshi/Unmute (Kyutai)**: Mimi 코덱(80ms) + 사용자·모델을 **병렬 오디오 2스트림**으로 모델링 →
  overlap/backchannel/barge-in이 내재(명시적 턴 없음). ~160~200ms. **영어·GPU.** **Unmute의 "flush
  trick"**(STT가 실시간 ~4배 → 조기 커밋으로 응답 지연↓)과 **semantic-VAD-in-STT**(운율·내용 적응
  지연)가 프론티어. → 우리는 flush 트릭·semantic VAD 개념을 캐스케이드에 이식 가능.
- **Ultravox**: 오디오를 오픈 LLM에 직접 투사(STT+LLM 통합), ~150ms TTFT. **GPU 필수, CM4 불가.**
- **TEN Framework(구 Astra)**: 그래프 멀티모달. **TEN Turn Detection = Qwen2.5-7B로 finished/unfinished/
  wait 3-상태**(영/중, GPU). → **3-상태 턴 모델 추상화**를 우리 엔드포인트 로직에 채택(모델은 너무 무거움).
- **Sesame CSM**(max-new-tokens로 **TTFA ~2초 바운딩**), **Hertz-dev**(2화자 overlap 생성, 120ms/GPU),
  **Freeze-Omni**("텍스트 LLM은 freeze하고 스트리밍 speech I/O만 얹기" = 우리 철학과 동형, GPU),
  **GPT-SoVITS**(네이티브 한국어 G2P·서브초, 단 GPU·TTS-only).

## 6. 횡단 정리 — 우리가 실제로 얻어갈 것

### (A) 턴테이킹/엔드포인팅 — 가장 중요
- **Smart Turn v3 채택 = 정답(완료).** 오디오 네이티브·한국어·CPU 12ms.
- **적응형 침묵 타임아웃**(LiveKit/Kyutai 공통): 작은 모델로 침묵 대기를 내용·운율에 따라 늘렸다 줄임.
  우리 `endpoint_grace_ms`를 고정 대신 Smart Turn 확률로 적응화 → 다음 개선 1순위.
- **3-상태(finished/unfinished/wait)**(TEN): 우리 barge-in(끼어들기)·backchannel(맞장구)·endpoint(멈춤)를
  하나의 3-상태 판단으로 통합하면 로직이 깔끔.
- **backchannel 스킵**(Vocode `interrupt_sensitivity`) = 우리 설계 검증됨.

### (B) 레이턴시 엔지니어링
- **TTFA(첫 오디오까지 시간)가 체감 지표** — 총 레이턴시 아님. 바운딩 수단: max-new-tokens 캡(CSM),
  LLM 토큰스트림→TTS 체이닝(KoljaB), 문장 단위 스트리밍 TTS(GLaDOS/Pipecat), 침묵 시 STT 정지.
- **스테이지 overlap**(Pipecat): VAD-세그먼트→STT(스트리밍 STT 복잡도 회피) + 스트리밍 TTS.
- **flush 트릭**(Unmute): STT를 실시간보다 빠르게 → 조기 커밋/엔드포인트.
- **LocalAgreement**(whisper_streaming): CPU 스트리밍 ASR을 하려면 이 알고리즘.

### (C) 온디바이스 배포 패턴
- **2단 컴퓨트**: 상시 싼 wake-word 게이트(openWakeWord) + 필요 시 비싼 캐스케이드 기동. CM4 발열/RAM에
  중요. (우리는 지금 상시 VAD — wake word 도입은 선택.)
- **Wyoming 프로토콜**을 내부 계약으로 → 스테이지 분리 + satellite-server 탈출구.
- **grammar 패스트패스**(Rhasspy/OVOS): 흔한 명령은 LLM 건너뛰어 지연 0.
- **플러그인 스왑**(OVOS) = 우리 프리셋 구조와 동일 사상 → 잘 가고 있음.

### (D) 리스크 재확인
- **한국어 TTS**: CPU+상업라이선스 기성품 없음 → 자체 학습(우리 `tts-*.md` Path A)이 정공법.
- **풀-듀플렉스는 CM4에 못 올림** — Moshi/Ultravox/TEN-7B의 *행동*(맞장구·barge-in·경직 없는 턴)을
  캐스케이드의 턴 감지+빠른 중단으로 흉내낼 것.

## 7. 우선순위 액션 (제안)

1. **`endpoint_grace_ms` 적응화** — 고정 대신 Smart Turn `prob_complete`로 침묵 대기를 동적 조정
   (LiveKit/Kyutai 패턴, 우리가 이미 가진 부품으로 가능).
2. **문장 단위 스트리밍 TTS + "끼어들면 응답 clip"** — GLaDOS 방식으로 barge-in을 성숙화(TTFA↓ + 자연스러운 중단).
3. **3-상태 턴 추상화** — barge-in/backchannel/endpoint를 finished/unfinished/wait로 통합 리팩터(TEN).
4. **(선택) 진짜 스트리밍 ASR** — streaming-zipformer에 LocalAgreement 얹어 부분 transcript/조기 엔드포인트.
5. **(선택) 2단 컴퓨트/Wyoming** — CM4 실기에서 발열/RAM 문제 시 wake-word 게이트 + 프로토콜 분리.

## 8. 비교표 (요약)

| 프로젝트 | 유형 | 로컬 CPU | 한국어 | 턴/중단 | 라이선스 | 우리 관련도 |
|---|---|---|---|---|---|---|
| Pipecat | 캐스케이드 | 오케스트레이터 O | Smart Turn 경유 | 강함 | Apache-2.0 | ★★★ 동일 구조 |
| **Smart Turn v3** | 엔드포인터 | **O** | **O(23개어)** | 오디오 EOU | Apache-2.0 | ★★★ 이미 도입 |
| GLaDOS | 올인원 | O(느림)/NPU | ✗ | **최고** | MIT | ★★★ 턴 레퍼런스 |
| HF speech-to-speech | 캐스케이드 | O | 모델 의존 | Silero VAD | Apache-2.0 | ★★★ 배포 아날로그 |
| HA Assist+Wyoming | 위성+서버 | 위성 O | Whisper✅/Piper✗ | 기본 | Apache-2.0 | ★★ 프로토콜 |
| OVOS+Neon | 모듈형 | **O(raspOVOS)** | **Vosk✅** | 인텐트+ | Apache-2.0 | ★★ 형제 |
| whisper_streaming | 스트리밍 ASR 알고 | O | Whisper | — | MIT | ★★ 트릭 |
| LiveKit Agents | 캐스케이드 | 일부(WebRTC) | EOU 영어 | 적응 타임아웃 | Apache-2.0 | ★★ 아이디어 |
| Vocode | 캐스케이드 | O(정체) | 모델 의존 | backchannel skip | MIT | ★ 설계 검증 |
| Moshi/Unmute | 풀-듀플렉스 | **✗ GPU** | 영/불 | 내재 | 관대 | ★ 아이디어 |
| TEN | 그래프 | **✗ GPU** | **✗** | 3-상태 | Apache-2.0 | ★ 추상화 |
| Ultravox | 스피치 네이티브 | **✗ GPU** | LLM 의존 | 내재 | 오픈웨이트 | 참고 |

## 9. 참고 링크

- 오케스트레이터: [Pipecat](https://github.com/pipecat-ai/pipecat) · [Smart Turn v3](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/) / [repo](https://github.com/pipecat-ai/smart-turn) · [LiveKit EOU](https://livekit.com/blog/using-a-transformer-to-improve-end-of-turn-detection) / [agents](https://github.com/livekit/agents) · [HF speech-to-speech](https://github.com/huggingface/speech-to-speech) · [Vocode](https://github.com/vocodedev/vocode-core) · [1s voice-to-voice(Modal+Pipecat)](https://modal.com/blog/low-latency-voice-bot) · [awesome-voice-agents](https://github.com/yzfly/awesome-voice-agents)
- 온디바이스 비서: [GLaDOS](https://github.com/dnhkng/GLaDOS) · [Wyoming](https://www.home-assistant.io/integrations/wyoming/) / [satellite](https://github.com/rhasspy/wyoming-satellite) · [OVOS](https://github.com/OpenVoiceOS/OpenVoiceOS) / [raspOVOS](https://github.com/OpenVoiceOS/raspOVOS) · [Rhasspy](https://github.com/rhasspy/rhasspy) · [Willow](https://github.com/HeyWillow/willow) · [june](https://github.com/mezbaul-h/june) · [Dicio](https://github.com/Stypox/dicio-android) · [openWakeWord](https://github.com/dscripka/openWakeWord)
- 스트리밍/부품: [whisper_streaming(2307.14743)](https://arxiv.org/html/2307.14743v2) · [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) / [RealtimeVoiceChat](https://github.com/KoljaB/RealtimeVoiceChat) · [speaches](https://github.com/speaches-ai/speaches) · [WhisperLive](https://github.com/collabora/WhisperLive)
- 풀-듀플렉스: [Moshi](https://github.com/kyutai-labs/moshi) / [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) · [Ultravox](https://github.com/fixie-ai/ultravox) · [TEN](https://github.com/TEN-framework/ten-framework) / [turn-detection](https://github.com/ten-framework/ten-turn-detection) · [Freeze-Omni](https://github.com/VITA-MLLM/Freeze-Omni) · [Sesame CSM](https://github.com/SesameAILabs/csm) · [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
