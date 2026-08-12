"""Local LLM stage: Korean text in -> short conversational reply out.

Qwen3-0.6B (Apache-2.0, no NC restriction) via transformers. Not Korean-specialized
(EXAONE/Kanana score higher on Korean benchmarks but are both NC-licensed -- see
docs/output/ondevice_asr_llm_tts_research_20260716.md for the full comparison), so
treat reply quality here as a baseline to improve on, not a final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Iterator

from ..paths import PROJECT_ROOT
from ..persona import FEWSHOT_MESSAGES, SYSTEM_PROMPT

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"

# `transformers` is imported lazily too, for a plainer reason than torch: it is
# genuinely slow to import. Measured cold on the Windows environment at ~42s,
# and around a second even warm -- it eagerly walks a large module tree at
# import. Since registry.py imports this file merely to look up a class, every
# entry point (including scripts that never build an LLM at all) was paying
# that. Both classes need it only inside __post_init__, so deferring it costs
# nothing and removes the single largest fixed startup cost in the project.
#
# torch is deferred for a different and stronger reason: correctness of the
# dependency graph. The default preset (qwen3-0.6b-gguf) runs through llama.cpp
# and never touches torch, so an absent or broken torch -- a CPU-slim
# environment, a CUDA build mismatched to the driver -- should disable the raw-
# transformers presets, not fail the whole program at import. Deferring makes
# the cost and the risk land on exactly the presets that asked for them, which
# is what lets requirements/windows-cpu.txt treat torch as optional.
#
# The reason is the default preset: qwen3-0.6b-gguf runs through llama.cpp and
# never touches torch. But registry.py imports this module to reach *both* LLM
# classes, and pipeline.py imports it for a type annotation -- so a module-level
# `import torch` made every entry point pay for it, and made a torch that was
# absent or broken (a CPU-slim environment, a CUDA build mismatched to the
# driver) fail the whole program at import rather than only the presets that
# actually need it.
#
# Deferring it means the cost and the risk land on exactly the presets that
# asked for them: build an LFM2 or raw-Qwen preset and torch loads; stay on the
# GGUF default and it never does. That is worth roughly a second of startup on
# the default path, and it is what lets requirements/windows-cpu.txt treat torch
# as genuinely optional.


def _resolve_device(preference: str) -> str:
    """Turn a device preference into a concrete torch device string.

    ``"auto"`` prefers cuda (Nvidia) > mps (Apple Silicon) > cpu. Any other
    value is passed through untouched, so a preset can pin ``"cpu"`` to get
    CM4-realistic timings on a machine that has a GPU.

    Resolved per instance, at construction, rather than once at import: import
    time is the wrong moment to probe for a GPU (it charges every caller for a
    CUDA context they may not want), and a module-level constant cannot see a
    ``CUDA_VISIBLE_DEVICES`` set after import.
    """
    if preference != "auto":
        return preference

    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class NobodyLLM:
    model_id: str = DEFAULT_MODEL_ID
    # "auto" resolves at construction -- see _resolve_device. configs/models.yaml
    # may pin a concrete device instead.
    device: str = "auto"
    max_new_tokens: int = 96
    # Each stored turn is re-sent to the model (no cross-call KV cache -- see
    # reply() below), so an unbounded history means both re-encoding cost and
    # prompt length grow with conversation length. Capping keeps per-turn
    # latency roughly flat; 6 turns is plenty of context for this persona's
    # one/two-sentence exchanges.
    max_history_turns: int = 6
    history: list[dict] = field(default_factory=list)
    # Appended (blank-line separated) to persona.SYSTEM_PROMPT in reply() --
    # talk.py sets this after loading configs/memory-design.md-style recalled
    # memories (registry.py's presets never set it; there's no yaml field for
    # it, this is a runtime-only knob). Empty by default: no memory, no
    # change in behavior from before memory.py existed.
    system_prompt_suffix: str = ""

    def __post_init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _resolve_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                # bfloat16 on CUDA halves weight memory and bandwidth at
                # negligible quality cost for a 0.6B chat model. Not used on
                # cpu/mps: CPU bfloat16 matmul falls back to slow paths on most
                # x86, and mps support for it is uneven -- float32 is faster on
                # both despite being larger.
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
            .to(self.device)
            .eval()
        )

    def reset(self):
        self.history = []

    def _build_prompt(self, user_text: str) -> str:
        system_content = SYSTEM_PROMPT
        if self.system_prompt_suffix:
            system_content = f"{SYSTEM_PROMPT}\n\n{self.system_prompt_suffix}"
        messages = [
            {"role": "system", "content": system_content},
            # Ahead of the real history, so the model has seen the tone
            # demonstrated before it sees anything the user actually said. See
            # persona.FEWSHOT_MESSAGES for why demonstration beats description
            # at this model size.
            *FEWSHOT_MESSAGES,
            *self.history,
            {"role": "user", "content": user_text},
        ]
        # Qwen3 supports an explicit reasoning/thinking mode; off here since we
        # want short, fast, direct replies for a voice pipeline, not a scratchpad.
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def _remember(self, user_text: str, reply_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply_text})
        # Trim oldest turns first (2 messages/turn: user + assistant) -- see
        # max_history_turns above.
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def reply(self, user_text: str) -> str:
        """One turn: append user_text, generate a reply, append it, return it.

        Delegates to reply_stream so the streaming and non-streaming paths can't
        drift apart (same prompt, same history bookkeeping) -- callers that
        don't want incremental output (run_pipeline.py, benchmark.py) just join
        the stream.
        """
        return "".join(self.reply_stream(user_text))

    def reply_stream(self, user_text: str) -> Iterator[str]:
        """Same turn as reply(), but yields the reply in text pieces as the
        model produces them (see pipeline.run_streaming / textchunk.py) instead
        of only returning once the whole reply exists. History is updated once,
        after the stream is exhausted.

        Note: re-sends the full history as text every call (no cross-turn KV
        cache) -- acceptable for this prototype's short exchanges (see
        max_history_turns), worth revisiting in the deferred on-device pass.
        """
        import torch
        from transformers import TextIteratorStreamer

        prompt = self._build_prompt(user_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            # Mild sampling (not greedy) so replies don't feel robotic/repetitive
            # turn to turn; kept conservative since this is a companion voice,
            # not a creative-writing task.
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # generate() runs on a worker thread so this generator can yield the
        # streamer's pieces as they arrive. torch.no_grad is thread-local, so
        # it has to be entered inside the worker, not via a decorator here.
        def _run():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        thread = Thread(target=_run, daemon=True)
        thread.start()

        pieces: list[str] = []
        for text in streamer:
            pieces.append(text)
            yield text
        thread.join()

        self._remember(user_text, "".join(pieces).strip())

    def generate_raw(
        self, system_prompt: str, user_text: str, max_new_tokens: int | None = None
    ) -> str:
        """One-off generation with an arbitrary system_prompt -- no history
        read or write, no persona.SYSTEM_PROMPT. Used by memory.py's
        extraction pass, which needs a completely different instruction
        ("extract facts as JSON") than this instance's conversational
        persona/history and shouldn't read from or pollute either. Reuses
        this instance's already-loaded tokenizer/model rather than spinning
        up a second one just for a single extraction call.

        do_sample=False (greedy) unlike reply()'s sampling -- structured
        extraction should be deterministic/repeatable, not varied turn to
        turn like a chat reply.
        """
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # A context manager rather than the @torch.no_grad() decorator this
        # used to carry: a decorator is evaluated when the class body runs,
        # which would reintroduce the module-level torch import this file
        # deliberately avoids.
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


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
    # Turn boundary tokens to stop generation on. Defaults to ChatML's, which is
    # what Qwen uses; every model family spells this differently (EXAONE
    # "[|endofturn|]", Gemma "<end_of_turn>"), and getting it wrong does not
    # error -- the model simply runs on past its reply, inventing the user's
    # next line, until max_tokens. A preset that swaps the model must swap this
    # too, which is why it is a field rather than a constant.
    stop: list[str] = field(default_factory=lambda: ["<|im_end|>"])
    # Extra arguments for apply_chat_template. enable_thinking=False is Qwen3's
    # switch for suppressing its <think> block (see the class docstring); other
    # families neither need nor accept it, so _build_prompt drops these if the
    # template rejects them.
    template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})
    n_ctx: int = 4096
    n_threads: int = 8
    # 0 = CPU only (default, works everywhere incl. CM4). On Apple Silicon set
    # to -1 (offload all layers) to use Metal via llama.cpp's Metal backend --
    # the pip wheel for macOS ships it. Left at 0 by default so the CM4/CPU
    # target and Linux boxes are unaffected; it's a per-preset opt-in.
    n_gpu_layers: int = 0
    max_new_tokens: int = 96
    max_history_turns: int = 6
    history: list[dict] = field(default_factory=list)
    # Same purpose as NobodyLLM's field of the same name -- see its docstring.
    system_prompt_suffix: str = ""

    def __post_init__(self):
        from llama_cpp import Llama
        from transformers import AutoTokenizer

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

    def reset(self):
        self.history = []

    def _build_prompt(self, user_text: str) -> str:
        system_content = SYSTEM_PROMPT
        if self.system_prompt_suffix:
            system_content = f"{SYSTEM_PROMPT}\n\n{self.system_prompt_suffix}"
        messages = [
            {"role": "system", "content": system_content},
            # Ahead of the real history, so the model has seen the tone
            # demonstrated before it sees anything the user actually said. See
            # persona.FEWSHOT_MESSAGES for why demonstration beats description
            # at this model size.
            *FEWSHOT_MESSAGES,
            *self.history,
            {"role": "user", "content": user_text},
        ]
        return self._render(messages)

    def _render(self, messages: list[dict]) -> str:
        """Apply the model's own chat template to a message list.

        Shared by the conversational path and generate_raw so a preset's
        template settings cannot apply to one and not the other -- they diverged
        once already, which is the kind of bug that shows up as one code path
        mysteriously ignoring the model's turn markers.
        """
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs
            )
            return self._strip_duplicate_bos(rendered)
        except (TypeError, ValueError):
            # A template that rejects one of the extra kwargs, or one with no
            # system role (Gemma raises ValueError for that). enable_thinking is
            # Qwen's; passing it to a template that does not declare it can be a
            # hard error rather than a no-op. Retry plainly rather than make
            # every non-Qwen preset remember to clear the field.
            if not self.template_kwargs:
                raise
            return self._strip_duplicate_bos(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

    def _strip_duplicate_bos(self, prompt: str) -> str:
        """Remove a leading BOS token that llama.cpp is about to add again.

        Llama-3 style templates (Mi:dm, Kanana) emit <|begin_of_text|> as part
        of the rendered string, and create_completion() tokenizes with add_bos
        on top of that. llama.cpp notices and warns that the duplicate "will
        likely reduce response quality" -- the model sees a sequence that never
        occurred in training. ChatML templates do not emit one, so this is a
        no-op for the Qwen presets.
        """
        bos = getattr(self.tokenizer, "bos_token", None)
        if bos and prompt.startswith(bos):
            return prompt[len(bos):]
        return prompt

    def _remember(self, user_text: str, reply_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply_text})
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def warm_up(self) -> None:
        """Prefill the static part of the prompt so the first real turn is not
        the one that pays for it.

        Every turn re-sends the same prefix -- the persona, the few-shot
        examples, and for some models a large built-in system prompt of their
        own (Mi:dm prepends about a thousand tokens). llama.cpp caches that
        prefix's KV across calls, so from the second turn on it costs nothing;
        the first turn pays for all of it at once. Measured on Mi:dm: 6.7s cold
        against 0.7s warm.

        The fix is not to shrink the prompt but to move the cost somewhere the
        user is not waiting. scripts/talk.py calls this while the greeting is
        being synthesized and played, which is dead time of roughly the right
        size.

        Generates a single token rather than zero: llama.cpp populates the
        cache as part of evaluating a request, so asking for no output would
        not prime anything. Failures are swallowed -- this is an optimization,
        and a session that starts slowly is better than one that does not
        start.
        """
        try:
            self._llm.create_completion(
                self._build_prompt("안녕"), max_tokens=1, temperature=0.0, stop=self.stop
            )
        except Exception:
            pass

    def reply(self, user_text: str) -> str:
        return "".join(self.reply_stream(user_text))

    def reply_stream(self, user_text: str) -> Iterator[str]:
        """Streaming counterpart of reply() -- yields text pieces as llama.cpp
        decodes them. Same history bookkeeping, updated once after the stream
        ends. See NobodyLLM.reply_stream for the shared rationale."""
        prompt = self._build_prompt(user_text)

        stream = self._llm.create_completion(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            # Turn boundary -- without it, raw create_completion() keeps
            # generating past the reply into a hallucinated next turn
            # (create_chat_completion() would normally stop here itself, but
            # we're deliberately not using it -- see class docstring).
            stop=self.stop,
            stream=True,
        )

        pieces: list[str] = []
        for chunk in stream:
            piece = chunk["choices"][0]["text"]
            if piece:
                pieces.append(piece)
                yield piece

        self._remember(user_text, "".join(pieces).strip())

    def generate_raw(
        self, system_prompt: str, user_text: str, max_new_tokens: int | None = None
    ) -> str:
        """See NobodyLLM.generate_raw's docstring -- same contract, same
        reason to exist (memory.py's extraction pass), llama-cpp-python
        backend instead of raw transformers.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt = self._render(messages)
        out = self._llm.create_completion(
            prompt,
            max_tokens=max_new_tokens or self.max_new_tokens,
            temperature=0.0,
            stop=self.stop,
        )
        return out["choices"][0]["text"].strip()
