"""Local LLM stage: Korean text in -> short conversational reply out.

Qwen3-0.6B (Apache-2.0, no NC restriction) via transformers. Not Korean-specialized
(EXAONE/Kanana score higher on Korean benchmarks but are both NC-licensed -- see
docs/output/ondevice_asr_llm_tts_research_20260716.md for the full comparison), so
treat reply quality here as a baseline to improve on, not a final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import PROJECT_ROOT
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


DEFAULT_GGUF_MODEL_PATH = PROJECT_ROOT / "models" / "qwen3-0.6b-gguf" / "Qwen3-0.6B-Q4_K_M.gguf"


@dataclass
class NobodyLLMGguf:
    """Second LLM candidate: same Qwen3-0.6B weights, but quantized to GGUF
    (Q4_K_M, community-requantized by bartowski from the official
    Qwen/Qwen3-0.6B checkpoint) and run through llama-cpp-python instead of
    raw transformers.

    Why this exists: benchmarked raw transformers on CPU earlier (this
    project's dev-box measurements) at ~3.9-5.1s/turn -- confirmed by hand
    that swapping to GGUF+llama.cpp cuts that to ~1.5-2.0s/turn on the same
    CPU, matching the pattern already proven elsewhere in this project
    (VibeVoice-ASR-BitNet/TEN-VAD: GGML-family CPU runtimes beat raw
    PyTorch/transformers by a wide margin). Model is unchanged -- same
    weights, same Apache-2.0 license, same Korean support -- only the
    execution backend differs, so this is a fair like-for-like comparison
    against NobodyLLM, not a different model.

    The one real wrinkle: llama-cpp-python's high-level
    `create_chat_completion()` does NOT expose Qwen3's `enable_thinking`
    template toggle (confirmed by inspecting its signature -- no
    `chat_template_kwargs`/`enable_thinking` parameter exists), unlike
    llama.cpp's own `llama-server` HTTP mode (fixed in upstream PR #22336,
    which isn't what this class uses). Without disabling it, Qwen3 emits a
    `<think>...</think>` block and can burn the entire max_tokens budget on
    reasoning without ever producing a visible reply -- confirmed by hand.
    Sidestepped here by NOT using create_chat_completion() at all: this
    class reuses NobodyLLM's own tokenizer (transformers is already a
    project dependency) purely to render the prompt string via
    apply_chat_template(..., enable_thinking=False) -- exactly like
    NobodyLLM does -- and feeds that plain string into llama-cpp-python's
    low-level create_completion() (pure text completion, no chat-template
    logic of its own to fight with).
    """

    model_path: Path = DEFAULT_GGUF_MODEL_PATH
    # Tokenizer-only load (no model weights) purely to render the chat
    # template the same way NobodyLLM does -- see class docstring. Must
    # match the GGUF's source checkpoint's template, not an arbitrary choice.
    tokenizer_id: str = DEFAULT_MODEL_ID
    n_ctx: int = 4096
    n_threads: int = 8
    max_new_tokens: int = 96
    max_history_turns: int = 6
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

    def reset(self):
        self.history = []

    def reply(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": user_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        out = self._llm.create_completion(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            # ChatML turn boundary -- without this, raw create_completion()
            # keeps generating past the reply into a hallucinated next turn
            # (create_chat_completion() would normally stop here itself, but
            # we're deliberately not using it -- see class docstring).
            stop=["<|im_end|>"],
        )
        reply_text = out["choices"][0]["text"].strip()

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply_text})
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
        return reply_text
