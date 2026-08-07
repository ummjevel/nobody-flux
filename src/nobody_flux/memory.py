"""Session-end memory extraction: turns a session's turns into structured
facts for storage.py's `memories` table.

Design + the risks/tradeoffs behind these choices are written up in
docs/memory-design.md (batch-at-session-end vs per-turn, category
vocabulary, why extraction output can't be trusted to be clean JSON). This
module is the "다음 단계" that doc left unimplemented -- talk.py wires it in
at session start (recall injection) and session end (extraction).
"""

from __future__ import annotations

import json
import re

# docs/memory-design.md's "기억 항목 수에 상한을 두거나" suggestion -- caps
# how many facts one session's extraction pass can add, so one unusually
# chatty or repetitive session can't flood the table (and, via
# ConversationStore.recent_memories, next session's prompt) on its own.
MAX_MEMORIES_PER_SESSION = 10

# JSON-array-only instruction, deliberately strict (docs/memory-design.md
# flags 0.6B-class instruction-following as the real risk here) -- paired
# with _extract_json_array's defensive parsing below rather than trusted to
# always hold.
EXTRACTION_SYSTEM_PROMPT = """\
너는 대화 기록에서 나중에 기억해두면 좋을 사실만 뽑아내는 도구야. 아래 대화를 분석해서
사용자에 대한 사실을 JSON 배열로만 출력해. 인사말, 설명, 마크다운 코드블록 없이 배열 자체만
출력해.

각 항목 형식: {"category": "...", "key": "...", "value": "...", "confidence": 0.0에서 1.0
사이 숫자}
category는 다음 중 하나: identity(이름/닉네임), interest(취미/관심사),
recurring_topic(자주 언급하는 사람/장소/일정), preference(취향), context(직업/거주지 등
상황 정보).
확실한 사실일수록 confidence를 높게, 추측이면 낮게 매겨.
이름, 취미, 반려동물처럼 구체적인 정보가 나오면 반드시 뽑아야 해 -- 사소해 보여도 놓치지 마.
날씨 얘기처럼 사용자에 대한 정보가 전혀 없는 대화에서만 빈 배열 []을 출력해.

예시 입력:
사용자: 안녕 나는 민준이야
퀜: 안녕 민준아, 반가워.
사용자: 나 요즘 러닝 시작했어
퀜: 오 멋있다, 얼마나 뛰어?

예시 출력:
[{"category": "identity", "key": "이름", "value": "민준", "confidence": 0.9}, {"category": "interest", "key": "취미", "value": "러닝을 시작함", "confidence": 0.8}]
"""


def _build_transcript(turns: list[tuple[str, str]]) -> str:
    lines = []
    for user_text, reply_text in turns:
        lines.append(f"사용자: {user_text}")
        lines.append(f"퀜: {reply_text}")
    return "\n".join(lines)


def _extract_json_array(raw_text: str) -> list[dict]:
    """Defensively pull a JSON array of memory dicts out of raw_text.

    docs/memory-design.md's flagged risk, made concrete: a 0.6B model told
    "output only a JSON array" is not guaranteed to actually do that -- it
    may wrap it in a ```json code fence, prepend a sentence, or emit
    something that isn't valid JSON at all. Rather than a bare
    json.loads(raw_text) that throws on the first stray character, this
    finds the first '[' .. last ']' span and parses just that; any failure
    along the way (no match, invalid JSON, wrong top-level type, malformed
    items) returns [] -- treated the same as "the model decided there was
    nothing worth remembering" rather than raising and losing the whole
    session's extraction over one bad generation.
    """
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    memories = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        category, key, value = item.get("category"), item.get("key"), item.get("value")
        if not (isinstance(category, str) and isinstance(key, str) and isinstance(value, str)):
            continue
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = None
        memories.append(
            {"category": category, "key": key, "value": value, "confidence": confidence}
        )
    return memories


def extract_memories(llm, turns: list[tuple[str, str]]) -> list[dict]:
    """Run one extraction pass over a session's (user_text, reply_text)
    turns, returning parsed memory dicts (category/key/value/confidence).
    Caller is responsible for writing them to storage (talk.py's
    session-end handling calls ConversationStore.save_memory per dict).

    `llm` needs a generate_raw(system_prompt, user_text, max_new_tokens)
    method (NobodyLLM/NobodyLLMGguf both have one) -- deliberately NOT
    llm.reply(), which would run through persona.SYSTEM_PROMPT and read/
    write the live conversation history; extraction needs neither and
    would corrupt both if it used reply() instead.

    Returns [] immediately for an empty session, without spending a
    generation call extracting nothing from nothing.
    """
    if not turns:
        return []
    transcript = _build_transcript(turns)
    raw_text = llm.generate_raw(EXTRACTION_SYSTEM_PROMPT, transcript, max_new_tokens=512)
    return _extract_json_array(raw_text)[:MAX_MEMORIES_PER_SESSION]


def format_recall_block(memories: list[tuple]) -> str:
    """Renders ConversationStore.recent_memories()'s rows into the bullet
    block docs/memory-design.md's "다음 세션에 어떻게 반영할까" section
    sketches, for NobodyLLM/NobodyLLMGguf's system_prompt_suffix. Returns ""
    (not e.g. "[기억]\\n") for an empty list, so talk.py can skip setting
    system_prompt_suffix at all when there's nothing to recall yet (first
    session ever) instead of appending an empty-looking header.
    """
    if not memories:
        return ""
    lines = "\n".join(f"- {key}: {value}" for _category, key, value, _confidence in memories)
    return f"[기억]\n{lines}"
