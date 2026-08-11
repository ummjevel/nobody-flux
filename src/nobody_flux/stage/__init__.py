"""The three swappable pipeline stages, plus their shared plumbing.

A "stage" is one model-backed step of a turn. Each has a deliberately narrow
interface so ``configs/models.yaml`` can substitute implementations without
any caller knowing which one is active (see ``nobody_flux.registry``):

===========  ================================================================
Stage        Contract
===========  ================================================================
ASR          ``transcribe_file(wav_path) -> str``
LLM          ``reply(text) -> str``, ``reply_stream(text) -> Iterator[str]``,
             ``generate_raw(...)``, ``reset()``, ``history``
TTS          ``synthesize(text, out_path) -> str``,
             ``synthesize_audio(text) -> (samples, sample_rate)``
===========  ================================================================

``asr_stream`` is the exception to the table: it is an ASR *engine* rather
than an ASR stage. Where the stage contract is file-in/text-out and inherently
batch, ``asr_stream`` consumes microphone frames as they arrive and exposes a
running hypothesis, so it can only be driven by something that owns the
capture loop (``nobody_flux.turn.controller``). Both exist because they answer
different questions: the batch stages are what ``scripts/benchmark.py``
compares, the streaming engine is what removes recognition latency from a live
conversation.

Nothing is re-exported here on purpose -- see the note on import cost in
``nobody_flux/__init__.py``. Importing this package must not drag in torch,
llama-cpp *and* sherpa-onnx when the caller wanted one of them.
"""
