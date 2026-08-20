# 다음에 이어서 할 것

> 2026-08-18~20 리서치·구현 세션이 끝난 지점. **결정된 것은 다시 논의하지 말고, 남은 것은
> 무엇에 막혀 있는지로 분류해뒀다.** 근거 문서는 `docs/output/research-delta-20260818.md`
> (이하 §), 상태 표는 `docs/FEATURES.md`, 라이선스는 `THIRD-PARTY-NOTICES.md`.

## 0. 한 문단 요약

리서치 5축 전부 종결, 실험 4트랙 중 3개 반영·1개 폐기. **조사로 답할 수 있는 항목은 남지
않았다.** 남은 것은 (a) 법무 판단 3건 — 전부 **기본 파이프라인 스테이지**다, (b) 실기 측정,
(c) 상류 릴리스 대기, (d) 지금 착수 가능한 개발 몇 개. 자동 테스트 337개 통과.

## 1. 결정된 것 — 다시 논의하지 않는다

| 결정 | 근거 |
|---|---|
| **TTS 기본값은 자체 `matcha-ko`** | §12. CER 구간이 supertonic-3·2와 겹치지 않고, 자체 학습이라 재학습 가능 |
| Supertonic 3·2는 **비교용 프리셋으로 유지** | §11.2, §13.3. v3의 실질 장점은 라이선스가 아니라 **전방 호환성** |
| **CM4를 사지 않고 CM5를 타깃** | §10(A72는 Armv8.0-A로 dotprod 없음) + §15(A76은 Armv8.2-A, 2.4GHz) |
| 스트리밍 ASR은 **chunked-SenseVoice** | §9, §7. zipformer는 실제 발화에서 빈 문자열(상류 #2886, 우리가 못 고침) |
| **투기적 프리필(트랙 C-2)은 만들지 않는다** | §8. 절약 ~43ms 대비 비용 ~200ms |
| Kokoro 한국어 트랙 폐기 | 한국어 음성이 **아예 없다** |
| **Smart Turn v3 유지** | §2 + §14. 학술 대안 전수 조사로 검증됨 |
| VAP 계열 종결 | §14.5. 한국어 없고, 논문 자신이 언어 전이를 반박 |
| 턴 판정은 **4-state** (`TurnVerdict`) | §14.4에서 Easy Turn과 독립 수렴 |
| VAD 엔진은 **설정값** (`ten-vad` 기본) | `configs/vad.yaml`. 라이선스 판단이 코드 변경이 되지 않게 |

## 2. 남은 것 — 막힌 이유별

### A. 🚨 법무 판단 (3건 전부 **기본 스테이지**다)

우리가 꼼꼼히 본 건 후보들이었고, 안 본 건 기본값이었다. 세 건 다 그렇다.

| # | 대상 | 답해야 하는 질문 |
|---|---|---|
| A-1 | **TEN-VAD** (기본 VAD) | Apache-2.0 **+ 추가 조건**. Agora 경쟁 금지 조항이 우리 제품 형태에 걸리는가? 그리고 **이 레포가 공개**인데 조항 1의 *"enabling any third party to develop or deploy Applications"*에 닿는가? (가중치는 재배포하지 않고 스크립트가 받아온다) → `THIRD-PARTY-NOTICES.md` §2.3 |
| A-2 | **SenseVoice** (기본 ASR) | FunASR Model Open Source License v1.1(Alibaba). 번들 LICENSE가 URL 한 줄뿐 → §11.1 |
| A-3 | **espeak-ng** (기본 TTS의 G2P) | GPL-3.0이 sherpa-onnx 바이너리에 **정적 링크**. 기기 이미지 배포 시 고지·소스 제공을 어떤 형태로? → §13, `THIRD-PARTY-NOTICES.md` §1.1 |

**A-1이 가장 급하다** — 제품화 이슈가 아니라 **이미 사실인 상태**다(공개 레포).
막히면 대안은 이미 배선돼 있다: `configs/vad.yaml`에서 `engine: silero-vad`(MIT) 한 줄.
단 threshold 재캘리브레이션이 필요하다(§2-B).

### B. 실기·사람 필요

| 항목 | 필요한 것 |
|---|---|
| CM4/CM5 실측 | 보드. §10.2의 13~18초 응답·130~230초 워밍업은 **Pi 4 역산 추정치**다 |
| `runtime.yaml` 스테이지 스레드 배분 | 실측. 파일 자신이 "CM4 실측 전까지 고정하지 말 것". **그리고 §15.3에서 전력 문제이기도 함이 밝혀졌다**(5V 2.5A, 데이터시트가 완화책으로 클럭 다운을 제시) |
| Smart Turn 임계값 스윕 (0.5/0.6/0.7/0.8) | 실제 발화. `_calibrate_turn_params.py`. **§14.2에서 우선순위가 올라갔다** — Smart Turn **V2**의 `ACC_incp`가 62%로 보고됐고, 그게 우리 적응형 grace가 의존하는 신호다 |
| Silero 재캘리브레이션 | 마이크. `_calibrate_vad_threshold.py`(이제 yaml의 `engine`을 읽는다). **임계값이 낮으면 턴이 영원히 안 끝난다** — 이 레포가 실제로 겪은 실패 |
| 한 글자 "네"의 의미 | 라벨 녹음. 맞장구("응, 계속해") vs 응답("네, 해줘")이 텍스트+길이로 구분 불가 |
| TTS 자연스러움 | 사람 청취. **CER은 명료도지 자연스러움이 아니다** |
| `labels.json` | 사람 라벨 — 단 §2-D-1의 우회로가 있다 |

### C. 상류 대기

**sherpa-onnx 2.0.0**이 espeak-ng을 제거한다([#3731](https://github.com/k2-fsa/sherpa-onnx/issues/3731)).
그때 `matcha-ko`가 좌초된다 — `tokens.txt`가 espeak IPA 음소 목록이고 메타데이터가
`has_espeak: 1`이다. 프론트엔드가 **설정이 아니라 ONNX 메타데이터로** 선택되므로
`lexicon`을 설정해도 무시된다(§13.8).

- 그래서 `pyproject.toml`이 `sherpa-onnx>=1.13.4,<2`로 상한을 걸고 있다
- 업스트림의 "모델마다 lexicon.txt 추가"는 **자기 레포 모델 한정** → 우리 건 우리가 만들어야
- **형식이 확정되기 전에 만들지 않는다.** 근거는 §13.8에 정리돼 있다(음절 단위가 맞고,
  espeak 한국어가 얇아서 잃을 것도 적다 — `ko_dict` 47KB, `lang/ko` 51바이트)

### D. 지금 착수 가능한 개발

**우선순위 순.**

1. **한국어 EOT 라벨을 사람 없이 만든다** ← 가장 값진 미착수 항목
   - Next-Turn: 학습 타깃을 *time-to-next-speech-onset*으로 두면 **"require no additional
     annotation"** — 타임스탬프에서 유도. 320ms 내 정확도 +25.9%p (§14.3-a)
   - Thai EOT: **YODAS 자막**으로 비영어 EOT 부트스트랩, 종결어미 같은 언어별 단서 활용 (§14.3-b)
   - **우리에게 이미 재료가 있다** — `data/sessions/`에 실사용 턴 오디오와 타임스탬프
   - 이게 §2-B의 임계값 스윕과 `labels.json`을 동시에 푼다. **"라벨이 없어서 못 한다"는
     더 이상 정확한 서술이 아니다**
2. **`is_empty_transcript` 한 글자 정책 보강** — `ONE_SYLLABLE_WORDS = {뭐, 뭘, 왜}`가 지금
   최소 집합이다. 실제 발화가 생기면 확장. 나/너는 의도적으로 제외(어절 첫 음절일 가능성)
3. **OpenLive에서 훔칠 패턴** — end-of-turn 모델을 ASR 디바이스와 **분리**해 ASR이 가속기에
   있어도 EOT는 무조건 CPU EP에서 돌린다 (§11, survey 각주 8)
4. **`_ab_asr.py`에 Silero/ten-vad 분절 비교 추가** — 지금은 깨끗한 wav 예비 확인만 있다
   (speech 3.01s vs 2.88s, 이벤트 시퀀스 동일). 실제 캡처 셋으로 재야 한다

## 3. 재개 전에 읽을 것 — 이번에 비싸게 배운 함정

**같은 실수를 다시 하지 않으려면 이 목록이 문서보다 중요하다.**

1. **1회 측정은 측정이 아니다.** Matcha는 flow-matching + `noise_scale`이고 sherpa가 프로세스
   단위로 시드한다 → 같은 문장이 md5·샘플수까지 다르다. 94자 세트 CER 해상도 **±0.04**.
   `_ab_tts.py --repeat`(기본 3) 필수. 이걸 모르고 낸 10화자 순위는 전부 무효였다 (§7, §12)
2. **espeak-ng은 프로세스 전역 싱글턴이다.** 한 프로세스에서 정상 경로를 먼저 초기화하면
   뒤이은 잘못된 경로가 그 상태를 재사용해 **거짓 통과**한다. 조건마다 프로세스를 분리하라
   (`_probe_espeak_dependency.py`가 그렇게 한다) (§13.4)
3. **문서보다 ONNX 메타데이터가 정확하다.** 이번에 출처·라이선스·열화 경고가 세 번 파일
   안에서 나왔다 — `vocos`(`model_author: BSC-LT`), `ten-vad`(라이선스 URL + *"uses 0 as the
   pitch feature, which may degrade the performance"*), `matcha-ko`(`has_espeak: 1`).
   재배포 모델을 만나면 **메타데이터를 먼저 읽어라** (`THIRD-PARTY-NOTICES.md` 규칙 5)
4. **검색 요약을 믿지 마라.** espeak 경로 우선순위를 "ESPEAK_DATA_PATH가 최우선"이라고
   요약했는데 소스가 반박했다(명시 경로가 최우선). `strings`가 없어서 "espeak 심볼 0개"라는
   거짓 음성도 받았다 — **대조군을 넣어라**(sherpa-onnx DLL에서 "sherpa"를 세보는 것) (§13.1)
5. **substring 단정문은 그 버그를 못 본다.** `"유월" in expand("6월")`이 통과해서
   `"유월 월"` 버그를 숨겼다. 등가 비교를 써라
6. **config 캐시를 변형하지 마라.** `_load_yaml`이 mtime 캐시된 dict를 그대로 주는데 빌더들이
   `update`/`pop`한다 → 두 번째 빌드가 첫 override를 물려받고 엔진 블록을 잃는다.
   deepcopy로 고쳤고 테스트 15개로 고정했다
7. **가짜 모델 파일로 테스트하면 프로세스가 죽는다.** 0바이트 `phontab`을 만들어 가드를
   통과시키면 espeak이 C에서 파싱하다 **pytest를 abort**한다(실패 리포트 없이 exit 1).
   가드 테스트는 duck-typed stub으로 메서드를 직접 불러라
8. **worktree 정리 시 junction을 따라가지 마라.** 39개 junction이 실제 `models/`(11GB)를
   가리키고 있었고 `rm -rf`는 그걸 따라간다. `[System.IO.Directory]::Delete(path, $false)`로
   링크만 끊어라
9. **PDF 추출이 막히면** `uv run --no-project --with pypdf`. **`--no-project` 필수** —
   프로젝트 안에서 그냥 `uv run`을 쓰면 uv가 `.venv`를 건드리려다 실패한다
10. **가중치 라이선스는 코드 라이선스가 아니다.** 이번 세션에 이 계열 함정을 **일곱 번**
    만났다. 규칙은 `THIRD-PARTY-NOTICES.md` §4에 있다. **[미확인]로 적는 것은 허용,
    추측해서 적는 것은 금지**

## 4. 게이트

```powershell
.venv-win\Scripts\python.exe -m pytest                     # 337개, <3s, 가중치 불요
.venv-win\Scripts\python.exe scripts\_smoke_imports.py
.venv-win\Scripts\python.exe scripts\_smoke_turn.py        # 가중치 필요
```

측정용 (**반드시 직렬** — 다른 실험이 유휴일 때):
```powershell
$env:NOBODY_CPU_BUDGET=4    # CM4 4코어 프록시
.venv-win\Scripts\python.exe scripts\_ab_tts.py --repeat 3
.venv-win\Scripts\python.exe scripts\_ab_asr.py
.venv-win\Scripts\python.exe scripts\_verify_kv_prefix.py
.venv-win\Scripts\python.exe scripts\_probe_espeak_dependency.py
```

> ⚠️ `NOBODY_CPU_BUDGET`을 설정한 채로 pytest를 돌리면 예전엔 실패했다. autouse
> `monkeypatch.delenv` 픽스처로 고쳐뒀지만, 새 런타임 테스트를 추가할 때 이 함정을 기억하라.
