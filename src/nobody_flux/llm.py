"""Local LLM stage: Korean text in -> short conversational reply out.

Qwen3-0.6B (Apache-2.0, no NC restriction) via transformers. Not Korean-specialized
(EXAONE/Kanana score higher on Korean benchmarks but are both NC-licensed -- see
docs/output/ondevice_asr_llm_tts_research_20260716.md for the full comparison), so
treat reply quality here as a baseline to improve on, not a final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .persona import SYSTEM_PROMPT

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


@dataclass
class NobodyLLM:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 96
    # Each stored turn is re-sent to the model (no cross-call KV cache -- see
    # reply() below), so an unbounded history means both re-encoding cost and
    # prompt length grow with conversation length. Capping keeps per-turn
    # latency roughly flat; 6 turns is plenty of context for this persona's
    # one/two-sentence exchanges.
    max_history_turns: int = 6
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
            .to(self.device)
            .eval()
        )

    def reset(self):
        self.history = []

    @torch.no_grad()
    def reply(self, user_text: str) -> str:
        """One turn: append user_text, generate a reply, append it, return it.

        Note: this re-sends the full history as text on every call rather than
        reusing a KV cache across turns, so cost grows with history length --
        acceptable for this prototype's short exchanges (see max_history_turns),
        but worth revisiting during the on-device optimization pass this repo defers.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": user_text},
        ]

        # Qwen3 supports an explicit reasoning/thinking mode; off here since we
        # want short, fast, direct replies for a voice pipeline, not a scratchpad.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            # Mild sampling (not greedy) so replies don't feel robotic/repetitive
            # turn to turn; kept conservative since this is a companion voice,
            # not a creative-writing task.
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = out_ids[0, inputs["input_ids"].shape[1] :]
        reply_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply_text})
        # Trim oldest turns first (2 messages/turn: user + assistant) -- see
        # max_history_turns above.
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
        return reply_text
