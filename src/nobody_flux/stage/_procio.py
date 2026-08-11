"""Shared helpers for driving a persistent line-protocol subprocess (used by
VibeAsrBitnet in asr.py and FreyaTtsKo in tts.py -- both spawn a long-lived
child process once, then send it one line per request and read one line (or
more, terminated by a sentinel) back per response, instead of spawning a
fresh subprocess -- and reloading the model -- on every call).

Both `LineReader` (for stdout, where we need "give me the next line, or tell
me within N seconds that none is coming") and `StderrDrainer` (for stderr,
where we just want to keep recent output around for error messages) use the
same shape: one dedicated background thread per stream, doing nothing but
blocking reads, handing lines off to the main thread through a thread-safe
handoff (queue.Queue / collections.deque). This is deliberate -- see
LineReader's docstring for the alternative (select() on the stream directly)
that looks simpler but is actually broken for this use case.
"""

from __future__ import annotations

import os
import queue
import threading
from collections import deque
from typing import IO


def clean_subprocess_env() -> dict[str, str]:
    """A copy of this process's environment, minus LD_LIBRARY_PATH.

    Why: scripts/env.sh sets LD_LIBRARY_PATH so *this* process's `import
    sherpa_onnx` can dlopen its bundled libonnxruntime.so (see asr.py's
    module-level comment). subprocess.Popen inherits the parent environment
    by default, so every child we spawn -- asr_stream_server (a totally
    unrelated C++ binary, linked against its own vendored ggml/llama.cpp, not
    onnxruntime) and the FreyaTTS server (its own isolated venv/torch build)
    alike -- would otherwise inherit an LD_LIBRARY_PATH override that has
    nothing to do with them. This is general hygiene, not a fix for any one
    specific bug (see LineReader/StderrDrainer below for the two that
    actually bit us during development).
    """
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    return env


class LineReader:
    """Reads lines from a subprocess stream (typically its stdout) on a
    dedicated background thread, and hands them to the caller one at a time
    via `get_line(timeout)`, which raises TimeoutError instead of blocking
    forever if the child goes quiet.

    Why a background thread + queue, instead of the more "obvious"
    `select.select([stream], [], [], timeout)` before each `readline()`:
    that approach was tried first and is subtly broken. `select()` only sees
    the OS-level pipe buffer -- but `stream` (an `io.TextIOWrapper`) sits on
    top of an `io.BufferedReader` that does its OWN internal buffering. A
    single underlying read() can pull in MORE than one line's worth of bytes
    (e.g. a response's content line immediately followed by its "---END---"
    marker, written by the child in quick succession) and `readline()` only
    hands back the first one, leaving the rest sitting in the
    BufferedReader's memory -- not the OS pipe. The *next* `select()` call
    then correctly reports "no new OS-level bytes are available" even though
    `readline()` could return that second line instantly from what's already
    buffered on our side. That looks exactly like the child hanging, and (in
    this codebase's case) reliably broke every multi-line response.

    The "obvious" follow-up fix -- check `stream.buffer.peek(1)` first, only
    `select()` if it's empty -- turned out to be broken in the *opposite*
    direction: `BufferedReader.peek()` performs a real (blocking!) read on
    the underlying stream when its internal buffer is empty, which defeats
    the entire point of doing a timeout-bounded check first. Confirmed by
    hand: with that version, a request to a genuinely stuck child blocked
    forever instead of raising TimeoutError, exactly the failure mode this
    class exists to prevent.

    A background thread sidesteps all of this: it just calls the stream's
    blocking `readline()` in a loop and pushes each line onto a
    `queue.Queue`, which supports a proper `get(timeout=...)` with no
    buffering-layer ambiguity -- there is exactly one place a line can be
    (the queue) and exactly one correct way to wait for one.
    """

    def __init__(self, stream: IO[str]):
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self._thread.start()

    def _read(self, stream: IO[str]) -> None:
        for line in stream:
            self._queue.put(line)
        # Stream hit EOF (child closed stdout, e.g. it exited) -- push a
        # sentinel so a waiting get_line() doesn't block forever waiting for
        # a line that will never come.
        self._queue.put(None)

    def get_line(self, timeout: float) -> str:
        """The next line (including its trailing newline, like
        `readline()`), or "" if the child's stream has hit EOF. Raises
        TimeoutError if neither arrives within `timeout` seconds."""
        try:
            line = self._queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no output from subprocess after {timeout}s") from None
        return line if line is not None else ""


class StderrDrainer:
    """Continuously reads a subprocess's stderr in a background thread and
    keeps only the last `keep_lines` lines, so error messages are still
    available (via `.tail()`) without ever letting the pipe fill up.

    Why this exists -- found the hard way: a child process's `stdin`/`stdout`
    protocol messages arrive fine even while its `stderr` pipe is left
    completely unread, RIGHT UP UNTIL that stderr pipe's OS buffer (64KB on
    Linux) fills. At that point the child's next `fprintf(stderr, ...)`
    blocks inside the kernel `write()` syscall -- and since it's now stuck
    mid-request, it never reaches the point where it'd write the stdout
    response we're waiting for either. From the outside this looks exactly
    like a hang: the child is still alive and even still burning CPU (if the
    stderr write happens to land after some further computation), just stuck
    on a full pipe nobody is emptying. Switching to `stderr=subprocess.DEVNULL`
    "fixes" this too, but throws away diagnostics we want on a real failure
    -- hence draining into a bounded buffer instead of just discarding.
    """

    def __init__(self, stream: IO[str], keep_lines: int = 200):
        self._lines: deque[str] = deque(maxlen=keep_lines)
        self._thread = threading.Thread(target=self._drain, args=(stream,), daemon=True)
        self._thread.start()

    def _drain(self, stream: IO[str]) -> None:
        for line in stream:
            self._lines.append(line)

    def tail(self) -> str:
        return "".join(self._lines)
