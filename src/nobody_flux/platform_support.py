"""Every OS-specific workaround this project needs, in one place.

Before this module the same platform knowledge was scattered across three
files, in three different shapes:

  * ``asr.py`` ran a block of ``glob``/``os.symlink`` code *at import time,
    above its own imports*, to make sherpa-onnx find onnxruntime's shared
    library. It hard-coded ``<project>/.venv/lib/*/site-packages`` as the
    install location, which is only true for one of the three environments
    this project now runs in.
  * ``__init__.py`` pinned two OpenMP environment variables for macOS.
  * ``tts.py`` built isolated-venv interpreter paths as ``<venv>/bin/python``,
    which does not exist on Windows.

Collecting them here buys three things. It makes the platform matrix legible
(one file answers "what is different about Windows?"). It removes the
duplicated, subtly-wrong site-packages derivation. And it gives the native
setup an explicit, callable entry point -- ``prepare_native_runtime()`` --
instead of leaving correctness dependent on a module happening to be imported
in the right order.

Import-time side effects are deliberately confined to ``__init__.py``, which
calls ``prepare_native_runtime()`` exactly once before any submodule loads a
native library. Everything here is idempotent, so a second call is harmless.

The platform matrix
-------------------

=============  ==========================  ====================================
Concern        Linux / WSL2                Windows / macOS
=============  ==========================  ====================================
onnxruntime    Bare-name ``.so`` symlink   Windows: add DLL directories.
discovery      + ``LD_LIBRARY_PATH``       macOS: symlink the versioned dylib
               (``scripts/env.sh``)        into ``sherpa_onnx/lib``.
OpenMP         No conflict observed        macOS: pin ``KMP_DUPLICATE_LIB_OK``
                                           + ``OMP_NUM_THREADS``.
venv layout    ``<venv>/bin/python``       Windows: ``<venv>/Scripts/python.exe``
=============  ==========================  ====================================
"""

from __future__ import annotations

import os
import platform
import site
import sys
import sysconfig
from pathlib import Path

# Resolved once at import. These are the canonical spelling of the platform
# test for the whole project -- prefer them over scattered
# ``platform.system() == ...`` / ``sys.platform == ...`` comparisons, which
# were previously written both ways in different modules for the same check.
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

# True when running under the Windows Subsystem for Linux. WSL reports itself
# as Linux (so IS_LINUX is also True), but its audio path goes through WSLg's
# PulseAudio bridge rather than real hardware, which is exactly why the live
# mic loop has never been validated on it -- see docs/FEATURES.md. Worth
# distinguishing so diagnostics can say "your mic is probably the problem"
# instead of leaving the user to guess.
IS_WSL = IS_LINUX and "microsoft" in platform.uname().release.lower()


def platform_label() -> str:
    """Short human-readable environment name for log lines and error messages
    (``"windows"``, ``"macos"``, ``"wsl2"``, ``"linux"``). Purely cosmetic."""
    if IS_WINDOWS:
        return "windows"
    if IS_MACOS:
        return "macos"
    if IS_WSL:
        return "wsl2"
    if IS_LINUX:
        return "linux"
    return sys.platform


def site_packages_dirs() -> list[Path]:
    """Directories where this interpreter's third-party packages actually live.

    Replaces the previous ``glob(PROJECT_ROOT / ".venv" / "lib" / "*" /
    "site-packages")``. That pattern encoded two assumptions that are both
    false in at least one supported environment: that the active environment
    is the project-root ``.venv`` (it is ``.venv-win`` on Windows, and may be
    any path at all when ``UV_PROJECT_ENVIRONMENT`` or a plain ``pip install``
    is used), and that the layout is POSIX ``lib/pythonX.Y/site-packages``
    (Windows uses ``Lib\\site-packages``, with no version component).

    Asking ``sysconfig``/``site`` instead means we find the packages belonging
    to whichever interpreter is running, which is the only set that can
    possibly matter -- the native libraries we are about to patch are the ones
    this process will ``dlopen``.
    """
    candidates: list[str] = []
    # purelib is the authoritative answer for the *running* interpreter and is
    # correct inside a venv, where site.getsitepackages() has historically been
    # unreliable across Python versions and platforms.
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        candidates.append(purelib)
    # getsitepackages() is absent on some stripped/virtualenv builds, hence the
    # guard rather than a bare call.
    if hasattr(site, "getsitepackages"):
        try:
            candidates.extend(site.getsitepackages())
        except Exception:  # pragma: no cover - defensive, platform dependent
            pass

    # Deduplicate while preserving order, and drop anything that isn't really
    # there (sysconfig happily reports paths that were never created).
    seen: set[Path] = set()
    resolved: list[Path] = []
    for raw in candidates:
        path = Path(raw)
        if not path.is_dir():
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved


def venv_interpreter(venv_dir: Path) -> Path:
    """Path to the ``python`` executable inside ``venv_dir``.

    The isolated-venv TTS backends (MOSS-TTS-Nano, FreyaTTS -- see
    ``stage/tts.py``) shell out to an interpreter in a *different* virtual
    environment, because those projects pin torch versions that conflict with
    this one's. Their layout differs by OS: POSIX puts the executable at
    ``bin/python``, Windows at ``Scripts/python.exe``. Returning the path
    unconditionally (rather than only when it exists) keeps the "which file did
    you expect?" information available to callers writing error messages.
    """
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def pin_native_runtime_env() -> None:
    """Set environment variables that must be in place before any native
    library initializes its threading runtime.

    macOS only, and only because of a genuine crash: torch, onnxruntime,
    llama.cpp and sherpa-onnx each link their own copy of ``libomp.dylib``.
    Whichever OpenMP runtime initializes second aborts the process with
    ``OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
    already initialized`` (confirmed on Apple Silicon).

    ``KMP_DUPLICATE_LIB_OK`` gets past the abort, but its documented side
    effect is that the two runtimes then contend -- observed here as
    llama.cpp's ``generate()`` deadlocking. Pinning ``OMP_NUM_THREADS=1``
    removes the contention. This costs nothing on the LLM stage: llama.cpp's
    matmul uses ggml's own thread pool (``n_threads``), not OpenMP, so the pin
    only reins in the OpenMP-using libraries that caused the conflict.

    ``setdefault``, not assignment, so an explicit override from the shell
    still wins. Windows and Linux are left untouched -- neither exhibits the
    duplicate-runtime abort, and pinning threads there would be a pure
    performance loss.
    """
    if not IS_MACOS:
        return
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _link_onnxruntime_macos(package_dirs: list[Path]) -> None:
    """macOS: make ``libonnxruntime.<ver>.dylib`` visible to sherpa-onnx.

    sherpa-onnx's compiled extension references the dylib as
    ``@rpath/libonnxruntime.<ver>.dylib`` and searches its *own* directory
    (``sherpa_onnx/lib/``) -- not ``onnxruntime/capi/``, where the pip wheel
    actually ships it. ``DYLD_LIBRARY_PATH`` is not a usable workaround because
    System Integrity Protection strips it from the environment of protected
    processes. So we symlink the real dylib into the directory ``@rpath``
    already looks in.
    """
    for pkg_dir in package_dirs:
        sources = list((pkg_dir / "onnxruntime").rglob("libonnxruntime.*.dylib"))
        lib_dir = pkg_dir / "sherpa_onnx" / "lib"
        if not sources or not lib_dir.is_dir():
            continue
        for src in sources:
            dst = lib_dir / src.name
            if dst.exists():
                continue  # a working link or a real file is already there
            # Not resolvable, so it is either missing or a *broken* symlink --
            # e.g. one created by hand with a relative target that no longer
            # points anywhere. Clear the broken link so the absolute one below
            # can replace it; exists() is False for broken links, which is why
            # this check has to come after it rather than instead of it.
            if dst.is_symlink():
                dst.unlink(missing_ok=True)
            try:
                dst.symlink_to(src)  # absolute target, resolved from site-packages
            except OSError:
                # Read-only site-packages, a concurrent install, or a
                # filesystem without symlink support. Import may still succeed
                # (a newer sherpa-onnx wheel may bundle its own runtime), so
                # this is not fatal -- failing loudly here would break setups
                # that never needed the link.
                pass


def _link_onnxruntime_linux(package_dirs: list[Path]) -> None:
    """Linux/WSL2: create the bare-name ``libonnxruntime.so`` sherpa-onnx
    ``dlopen``s.

    The pip onnxruntime wheel ships only the versioned
    ``libonnxruntime.so.1.27.0``, while sherpa-onnx's extension asks the
    dynamic linker for an unversioned ``libonnxruntime.so``. A relative symlink
    next to the real file satisfies the name.

    This is only half the fix. glibc's dynamic linker reads ``LD_LIBRARY_PATH``
    once, at process startup, so the directory must already be on that path
    before Python launches -- setting ``os.environ`` from here would be far too
    late. That is what ``scripts/env.sh`` is for, and why it must be *sourced*
    rather than executed.
    """
    for pkg_dir in package_dirs:
        capi = pkg_dir / "onnxruntime" / "capi"
        if not capi.is_dir():
            continue
        for versioned in capi.glob("libonnxruntime.so.*"):
            unversioned = capi / "libonnxruntime.so"
            if unversioned.exists():
                continue
            try:
                # Relative target: keeps working if the environment is later
                # relocated or copied, which absolute targets would not.
                unversioned.symlink_to(versioned.name)
            except OSError:
                pass  # see the macOS branch for why this is non-fatal


def _link_onnxruntime_windows(package_dirs: list[Path]) -> None:
    """Windows: register the native DLL directories with the loader.

    Windows needs no symlinks -- the wheels ship ``onnxruntime.dll`` and
    sherpa-onnx's DLLs under names their extensions already reference. What it
    *does* need is a search path. Since Python 3.8, extension modules no longer
    resolve their DLL dependencies from ``PATH`` or the current directory; only
    the directories explicitly registered via ``os.add_dll_directory`` (plus
    the system directories) are searched.

    sherpa-onnx's own ``__init__`` normally does this for its bundled DLLs, so
    in the common case this function changes nothing. It is here for the
    mixed-wheel case -- a sherpa-onnx build that expects to find onnxruntime's
    DLL beside it rather than bundling its own -- where the failure mode is
    otherwise an opaque ``ImportError: DLL load failed while importing
    _sherpa_onnx``. Registering both directories up front is cheap and turns
    that class of failure into a non-event.

    The returned handles are intentionally leaked: the directories must stay
    registered for the entire process lifetime, and closing them would undo
    exactly what we came here to do.
    """
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:  # pragma: no cover - non-Windows guard
        return
    for pkg_dir in package_dirs:
        for relative in (("onnxruntime", "capi"), ("sherpa_onnx", "lib")):
            dll_dir = pkg_dir.joinpath(*relative)
            if not dll_dir.is_dir():
                continue
            try:
                add_dll_directory(str(dll_dir))
            except OSError:
                pass  # already registered, or path vanished mid-startup


def link_native_libraries() -> None:
    """Make onnxruntime loadable by sherpa-onnx on whichever OS this is.

    Dispatches to the per-OS helper above. Safe to call more than once: each
    branch checks for the state it is about to create before creating it.
    """
    package_dirs = site_packages_dirs()
    if not package_dirs:  # pragma: no cover - would mean a broken interpreter
        return
    if IS_WINDOWS:
        _link_onnxruntime_windows(package_dirs)
    elif IS_MACOS:
        _link_onnxruntime_macos(package_dirs)
    else:
        _link_onnxruntime_linux(package_dirs)


def prepare_native_runtime() -> None:
    """One call that makes this process safe to import native libraries in.

    Order matters: the OpenMP variables must be set before any library that
    links OpenMP is loaded, and the library-discovery fixups must be in place
    before ``import sherpa_onnx``. ``nobody_flux/__init__.py`` calls this at
    package import, which is guaranteed to run before any submodule of the
    package -- so no submodule needs to care about ordering, and none of them
    should perform native setup of their own.
    """
    pin_native_runtime_env()
    link_native_libraries()
