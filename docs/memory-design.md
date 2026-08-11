# 기억(개인화) 설계

> 세션을 넘어 사용자를 기억하는 장기 기억 기능. **구현 완료** — 설계·구현·검증 결과를 한 문서에.

## 요약

- **무엇**: 세션 종료 시 대화에서 사용자 관련 사실을 추출해 SQLite에 저장하고, 다음 세션 시작 시
  불러와 시스템 프롬프트에 주입한다.
- **어디**: 추출/consolidation/recall 포맷팅은 `src/nobody_flux/memory.py`, 저장은
  `storage.py`(`memories` 테이블 + `save_memory`/`update_memory`/`recent_memories`), 연결은
  `scripts/talk.py`(세션 시작 recall + 종료 추출).
- **상태**: 동작·검증 완료. 남은 건 여러 세션 실사용 튜닝(아래 "다음 단계").

## 왜 필요한가

`NobodyLLM`은 세션(프로세스) 안에서만 히스토리를 들고 있다(`stage/llm.py`의 `max_history_turns`,
기본 6턴). 프로세스가 끝나면 대화가 사라진다. "더 personal"해지려면 세션을 넘어 지속되는
무언가가 필요하다 — 매번 "퀜, 나 기억해?"에 "처음 뵙는데요"라고 답하지 않으려면.

## 설계 결정

### 무엇을 추출할까 (카테고리)

| 카테고리 | 예시 | 비고 |
|---|---|---|
| `identity` | 이름/닉네임 | 가장 기본, 호칭에 직접 반영 |
| `interest` | 취미, 관심사, 최근 한 것 | 대화 소재로 재활용 |
| `recurring_topic` | 자주 언급하는 사람/장소/일정 | "그 프로젝트 잘 끝났어?" 팔로업에 필요 |
| `preference` | 음식/음악/말투 취향 | 톤 조정에도 사용 가능 |
| `context` | 직업, 사는 곳, 시간대 | 응답의 현실성에 영향 |

`memories.category`가 이 문자열을 그대로 받는다. enum으로 강제하지 않은 이유: 초기엔 경계가
애매하므로 실제 추출값을 보고 분류를 다시 잡는 게 정확하다.

### 언제 추출할까 — 세션 종료 시 일괄

- **매 턴 증분 추출**(기각): 반응성은 빠르나 0.6B에 매 턴 추가 추론 부담 + 짧은 턴에선 노이즈
  누적.
- **세션 종료 시 일괄**(채택): 세션 전체 turns를 한 번에 넣어 "기억할 만한 사실"을 뽑는다.
  실시간 지연 없음 + 반복·의미 있는 내용을 걸러낼 확률이 높음. `talk.py`의 `finally` 블록
  (세션 종료 처리부)에 붙인다.

### 어떻게 추출할까 — 방어적 파싱 + Mem0식 consolidation

1. **추출(`extract_memories`)**: 세션 turns를 `NobodyLLM.generate_raw`(persona/히스토리 우회)에
   넣어 `[{category, key, value, confidence}]` JSON 배열로 뽑는다.
2. **방어적 파싱(`_extract_json_array`)**: 0.6B는 "JSON만 출력"을 안정적으로 못 지킨다(코드펜스,
   잡설, 깨진 JSON). 첫 `[`~마지막 `]` 구간만 파싱하고, 실패하면 빈 배열로 취급 — 한 번의 나쁜
   생성이 세션 전체 추출을 날리지 않게.
3. **consolidation(`consolidate_memories`, Mem0식, arXiv:2504.19413)**: 새 사실을 무조건
   저장하지 않고 기존 기억과 비교해 **ADD/UPDATE/NOOP**로 처리한다(값이 바뀌면 UPDATE).
   **DELETE는 일부러 뺌** — 0.6B의 구조화 출력이 불안정한데 잘못된 DELETE는 되돌릴 수 없이
   데이터를 날리는 반면, 잘못된 ADD/UPDATE는 stale row로 남아 read-time dedup이 걸러낸다. 파싱
   실패 시 전부 ADD로 폴백 → consolidation은 개선만 하지 절대 퇴행 안 함.

### 중복 정리 (2단)

- **한 세션 안**(`_dedupe_memories`): 한 추출 호출이 같은 `(category, key)`에 서로 다른 값을 두
  개 뽑으면 confidence 높은 쪽만 남김. `MAX_MEMORIES_PER_SESSION` 자르기 **전에** 적용(자른 뒤
  중복 제거하면 상한 안에 중복만 남고 다른 사실이 밀려남).
- **세션 간**(`recent_memories`): 같은 사실이 여러 세션에 반복 추출될 때, SQL window function으로
  `(category, key)`별 최고 confidence/최신 행만 recall. 테이블 자체는 안 지움(읽을 때만 접는 뷰,
  전체 이력 보존).

### 다음 세션에 어떻게 반영할까 — recall 주입

세션 시작 시(`STSPipeline` 생성 후, 첫 턴 전) `recent_memories()`(confidence·최신순 상위 N개)를
`format_recall_block()`으로 불릿 리스트로 만들어 `NobodyLLM.system_prompt_suffix`에 주입한다.
`persona.py`의 `SYSTEM_PROMPT` 뒤에 붙는다:

```
[기억]
- 이름: 지수
- 최근 이직 준비 중이라고 언급함
- 고양이를 키움 ("루비")
```

상한을 두는 이유: `SYSTEM_PROMPT`가 길수록 처리 비용↑ + 0.6B는 긴 컨텍스트 반영 능력이 약하다.
그래서 최신·고신뢰 상위 N개로 제한한다.

## 검증 결과 (실측)

- `qwen3-0.6b-gguf`로 3턴 샘플 대화(이름/반려동물 언급) 추출 → `identity`/`interest` 정확히 파싱.
- consolidation도 0.6B로 "이름:지수(동일)→NOOP, 사는 곳:서울→부산(변경)→UPDATE, 취미:등산(신규)
  →ADD"를 정확히 판정.
- **핵심 교훈**: 처음엔 모델이 사소한 사실을 자꾸 빈 배열로 건너뛰었다 → 추출·consolidation
  프롬프트 모두 **원샷 예시**를 넣어 해결. 위 "리스크"가 우려한 그 문제였다.

## 다음 단계

- 여러 세션 실사용으로 추출/consolidation 품질 확인(특히 UPDATE 오판정, `confidence`의 의미).
- 테이블 무한 증가: UPDATE가 "값 변경"은 잡지만 "완전히 다른 새 사실"이 계속 쌓이는 건 여전 —
  오래된/낮은 confidence 행 정리 로직은 아직 없음(지금 스케일에선 보류).
