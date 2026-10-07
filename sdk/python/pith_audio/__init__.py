# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pith-hash
"""pith-audio SDK: WAV decode, spectral-peak signatures and
delta-t histogram matching through ctypes.

The single Rust core (the ``pith-audio`` cdylib built by
``cargo build --release``) is loaded at runtime; this package carries
no third-party dependency — ``ctypes`` is the standard library.

Discovery order (the suite's cdylib convention):

1. ``PITH_CDYLIB`` — an explicit cdylib *file* path;
2. ``PITH_CDYLIB_DIR`` — a *directory* scanned for the cdylib names
   (the CD pipeline points this at ``target/release``);
3. the package directory itself (the built wheel ships the cdylib as
   package data);
4. ``<repo root>/target/release`` — the repository working-tree layout,
   so a source checkout runs against a local cargo build with no
   configuration.

The FFI surface (all byte streams are big-endian headers plus
little-endian ``i32`` sample/record bodies — see the parse helpers):

* ``pith_audio_decode_wav`` decodes a whole RIFF/WAVE stream into the
  **decode stream** the ``pcm_i32_le_*`` digests are defined over;
* ``pith_audio_signature_wav`` / ``pith_audio_signature_pcm`` extract
  the **signature stream** (frames, peak count, fingerprint, peaks);
* ``pith_audio_voices_pcm`` synthesizes the SplitMix64 square-voice
  corpus *inside the cdylib* — the same pure recipe the synthetic
  ``reference.json`` vectors are built from, so SDK tests never
  reimplement synthesis and the vectors carry zero platform
  sensitivity;
* ``pith_audio_index_new`` / ``pith_audio_index_add`` /
  ``pith_audio_match`` build an incremental signature index (ids are
  insertion order) and match PCM against it (**match stream**);
* ``pith_audio_free`` / ``pith_audio_index_free`` release everything
  the FFI handed out.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "WavFacts",
    "SignatureFacts",
    "MatchEntry",
    "MatchTable",
    "AudioIndex",
    "FfiError",
    "LibraryNotFoundError",
    "find_cdylib",
    "decode_wav",
    "signature_wav",
    "signature_pcm",
    "voices_pcm",
    "parse_decode_stream",
    "parse_signature_stream",
    "parse_match_stream",
    "mono_average",
    "CDYLIB_NAMES",
    "STATUS_OK",
    "STATUS_INVALID",
    "STATUS_REJECTED",
]

#: Status: success.
STATUS_OK = 0
#: Status: a caller argument is invalid (a null pointer, a length that
#: is not a whole number of i32 samples, or a zero/oversized channel
#: count).
STATUS_INVALID = -1
#: Status: the core pipeline refused the input (malformed WAV, or PCM
#: the signature stage rejects).
STATUS_REJECTED = -2

#: Every cdylib file name cargo may drop into the build directory, per
#: platform (windows / linux / macOS).
CDYLIB_NAMES = ("pith_audio.dll", "libpith_audio.so", "libpith_audio.dylib")


class LibraryNotFoundError(OSError):
    """No cdylib was found through the discovery chain."""


class FfiError(Exception):
    """A non-zero status code came back from the cdylib."""

    def __init__(self, op: str, status: int) -> None:
        kind = {
            STATUS_INVALID: "invalid argument",
            STATUS_REJECTED: "input rejected",
        }.get(status, "unknown failure")
        super().__init__(f"{op} failed: {kind} (status {status})")
        #: The raw status code the FFI returned.
        self.status = status


def find_cdylib() -> Path:
    """Locates the cdylib through the suite's discovery chain."""
    explicit = os.environ.get("PITH_CDYLIB")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    env_dir = os.environ.get("PITH_CDYLIB_DIR")
    candidates: list[Path] = []
    if env_dir:
        env_dir_path = Path(env_dir)
        candidates.append(env_dir_path)
        if not env_dir_path.is_absolute():
            # CD and local runs invoke tools from the repository root or
            # from sdk/<lang>; resolve the env value against both.
            candidates.append(Path.cwd() / env_dir_path)
            candidates.append(Path(__file__).resolve().parents[3] / env_dir_path)
    candidates.append(Path(__file__).resolve().parent)  # packaged wheel
    candidates.append(Path(__file__).resolve().parents[3] / "target" / "release")
    for directory in candidates:
        for name in CDYLIB_NAMES:
            p = directory / name
            if p.is_file():
                return p
    raise LibraryNotFoundError(
        "no pith-audio cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR, "
        "the package directory and <repo>/target/release); "
        "run `cargo build --release` first"
    )


_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(find_cdylib()))

        def bytes_out(fn: ctypes._FuncPointer) -> None:
            fn.argtypes = [
                ctypes.c_void_p,  # data
                ctypes.c_size_t,  # len
                ctypes.POINTER(ctypes.c_void_p),  # out buffer
                ctypes.POINTER(ctypes.c_size_t),  # out length
            ]
            fn.restype = ctypes.c_int32

        bytes_out(lib.pith_audio_decode_wav)
        bytes_out(lib.pith_audio_signature_wav)
        lib.pith_audio_signature_pcm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,  # channels
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.pith_audio_signature_pcm.restype = ctypes.c_int32
        lib.pith_audio_voices_pcm.argtypes = [
            ctypes.c_uint64,  # seed
            ctypes.c_size_t,  # n_samples
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.pith_audio_voices_pcm.restype = ctypes.c_int32
        lib.pith_audio_index_new.argtypes = []
        lib.pith_audio_index_new.restype = ctypes.c_void_p
        lib.pith_audio_index_add.argtypes = [
            ctypes.c_void_p,  # index handle
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
        ]
        lib.pith_audio_index_add.restype = ctypes.c_int32
        lib.pith_audio_match.argtypes = [
            ctypes.c_void_p,  # index handle
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.pith_audio_match.restype = ctypes.c_int32
        lib.pith_audio_index_free.argtypes = [ctypes.c_void_p]
        lib.pith_audio_index_free.restype = None
        lib.pith_audio_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.pith_audio_free.restype = None
        _lib = lib
    return _lib


def _bytes_out(op: str, fn: ctypes._FuncPointer, *args: object) -> bytes:
    """Runs one byte-stream FFI op: hands the buffer back to the
    caller as ``bytes``, always freeing the cdylib's copy."""
    out = ctypes.c_void_p()
    out_len = ctypes.c_size_t()
    status = fn(*args, ctypes.byref(out), ctypes.byref(out_len))
    if status != STATUS_OK:
        raise FfiError(op, status)
    try:
        return ctypes.string_at(out, out_len.value)
    finally:
        _load().pith_audio_free(out, out_len.value)


def decode_wav(data: bytes) -> bytes:
    """Decodes a complete RIFF/WAVE stream into the decode stream the
    ``pcm_i32_le_sha256`` / ``pcm_i32_le_fnv1a64`` digests are defined
    over (16-byte big-endian header, then i32 LE samples).

    Raises :class:`FfiError` with ``status == STATUS_REJECTED`` for any
    malformed input; the decoder never panics through this boundary.
    """
    return _bytes_out("pith_audio_decode_wav", _load().pith_audio_decode_wav, data, len(data))


def signature_wav(data: bytes) -> bytes:
    """Decodes a RIFF/WAVE stream and extracts its signature (the
    ``signature_of_wav`` facade: anything but 44 100 Hz is refused)
    into the signature stream.

    Raises :class:`FfiError` with ``status == STATUS_REJECTED`` for
    malformed input or a refused sample rate.
    """
    return _bytes_out("pith_audio_signature_wav", _load().pith_audio_signature_wav, data, len(data))


def signature_pcm(data: bytes, channels: int) -> bytes:
    """Extracts a signature from interleaved little-endian i32 PCM at
    44 100 Hz into the signature stream.

    Raises :class:`FfiError` with ``status == STATUS_INVALID`` when
    ``len(data)`` is not a whole number of i32 samples or
    ``channels`` is zero / beyond 16 bits.
    """
    if channels < 0 or channels > 0xFFFF:
        raise FfiError("pith_audio_signature_pcm", STATUS_INVALID)
    return _bytes_out(
        "pith_audio_signature_pcm",
        _load().pith_audio_signature_pcm,
        data,
        len(data),
        channels,
    )


def voices_pcm(seed: int, n_samples: int) -> bytes:
    """Synthesizes the SplitMix64 square-voice corpus *inside the
    cdylib* — ``n_samples`` interleaved i32 LE samples, exactly the
    bytes the synthetic ``reference.json`` vectors are built from. The
    recipe is pure integer/exact-float math, so no SDK ever
    reimplements it and no platform sensitivity enters the vectors.
    """
    if seed < 0 or seed >= 1 << 64:
        raise FfiError("pith_audio_voices_pcm", STATUS_INVALID)
    return _bytes_out("pith_audio_voices_pcm", _load().pith_audio_voices_pcm, seed, n_samples)


class AudioIndex:
    """An incremental signature index: PCM signatures accumulate in
    insertion order, and the id a match reports is that order.

    The handle is released by :meth:`close` (also on ``__del__``);
    a failed :meth:`add` leaves the index alive and unchanged.
    """

    def __init__(self) -> None:
        self._handle: int | None = _load().pith_audio_index_new()

    def add(self, data: bytes, channels: int) -> None:
        """Appends the signature of interleaved i32 LE PCM to the
        index.

        Raises :class:`FfiError` with ``STATUS_INVALID`` for bad
        geometry, ``STATUS_REJECTED`` when the signature stage refuses
        the PCM — the index stays usable either way.
        """
        if self._handle is None:
            raise FfiError("pith_audio_index_add", STATUS_INVALID)
        if channels < 0 or channels > 0xFFFF:
            raise FfiError("pith_audio_index_add", STATUS_INVALID)
        status = _load().pith_audio_index_add(self._handle, data, len(data), channels)
        if status != STATUS_OK:
            raise FfiError("pith_audio_index_add", status)

    def match(self, data: bytes, channels: int) -> bytes:
        """Matches the signature of interleaved i32 LE PCM against the
        index and returns the match stream. An empty index is legal
        and yields zero matches.
        """
        if self._handle is None:
            raise FfiError("pith_audio_match", STATUS_INVALID)
        if channels < 0 or channels > 0xFFFF:
            raise FfiError("pith_audio_match", STATUS_INVALID)
        return _bytes_out(
            "pith_audio_match",
            _load().pith_audio_match,
            self._handle,
            data,
            len(data),
            channels,
        )

    def close(self) -> None:
        """Releases the handle; idempotent."""
        if self._handle is not None:
            _load().pith_audio_index_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # interpreter shutdown may have torn the lib down
            pass


@dataclass(frozen=True)
class WavFacts:
    """One decoded WAV, re-expressed from the decode stream.

    ``samples`` are the decoded i32 values in file order — exactly the
    bytes the ``pcm_i32_le_*`` digests cover (``raw[16:]``).
    """

    #: Sample rate the file declared.
    sample_rate: int
    #: Channel count the file declared.
    channels: int
    #: Bits per sample the file declared (8/16/24/32/64).
    bits_per_sample: int
    #: Whole frames the payload carries.
    frames: int
    #: Decoded interleaved sample count (``frames * channels``).
    decoded_samples: int
    #: The decoded samples as i32 values.
    samples: tuple[int, ...]
    #: The raw decode stream the digests are computed over.
    raw: bytes


@dataclass(frozen=True)
class SignatureFacts:
    """One extracted signature, re-expressed from the signature
    stream."""

    #: Analysis frames covered (``HOP`` = 2048 samples each).
    frames: int
    #: Number of spectral peaks.
    peak_count: int
    #: The presence-mask fingerprint as an integer.
    fingerprint: int
    #: The sorted ``(t, f)`` peak list.
    peaks: tuple[tuple[int, int], ...]
    #: The raw signature stream.
    raw: bytes


@dataclass(frozen=True)
class MatchEntry:
    """One corpus member's hit."""

    #: Position of the matched signature in the index (insertion order).
    id: int
    #: Modal ``query_t - stored_t`` in analysis frames.
    delta_t: int
    #: Votes within the tolerance window around ``delta_t``.
    votes: int
    #: Total raw votes the id received across all offsets.
    total_votes: int


@dataclass(frozen=True)
class MatchTable:
    """One match run, re-expressed from the match stream."""

    #: Analysis frames covered by the query signature.
    query_frames: int
    #: Peak count of the query signature.
    query_peak_count: int
    #: One entry per matched id, best first.
    matches: tuple[MatchEntry, ...]
    #: The raw match stream.
    raw: bytes


def parse_decode_stream(raw: bytes) -> WavFacts:
    """Re-expresses the decode stream as a :class:`WavFacts`."""
    if len(raw) < 16:
        raise ValueError("decode stream is shorter than the 16-byte header")
    decoded_samples = int.from_bytes(raw[12:16], "big")
    body = raw[16:]
    if len(body) != decoded_samples * 4:
        raise ValueError("decode stream body does not match decoded_samples")
    samples = tuple(int.from_bytes(body[i : i + 4], "little", signed=True) for i in range(0, len(body), 4))
    return WavFacts(
        sample_rate=int.from_bytes(raw[0:4], "big"),
        channels=int.from_bytes(raw[4:6], "big"),
        bits_per_sample=int.from_bytes(raw[6:8], "big"),
        frames=int.from_bytes(raw[8:12], "big"),
        decoded_samples=decoded_samples,
        samples=samples,
        raw=raw,
    )


def parse_signature_stream(raw: bytes) -> SignatureFacts:
    """Re-expresses the signature stream as a :class:`SignatureFacts`."""
    if len(raw) < 16:
        raise ValueError("signature stream is shorter than its 16-byte header")
    peak_count = int.from_bytes(raw[4:8], "big")
    body = raw[16:]
    if len(body) != peak_count * 6:
        raise ValueError("signature stream body does not match peak_count")
    peaks = tuple(
        (int.from_bytes(body[i : i + 4], "big"), int.from_bytes(body[i + 4 : i + 6], "big"))
        for i in range(0, len(body), 6)
    )
    return SignatureFacts(
        frames=int.from_bytes(raw[0:4], "big"),
        peak_count=peak_count,
        fingerprint=int.from_bytes(raw[8:16], "big"),
        peaks=peaks,
        raw=raw,
    )


def parse_match_stream(raw: bytes) -> MatchTable:
    """Re-expresses the match stream as a :class:`MatchTable`."""
    if len(raw) < 12:
        raise ValueError("match stream is shorter than the 12-byte header")
    match_count = int.from_bytes(raw[8:12], "big")
    body = raw[12:]
    if len(body) != match_count * 20:
        raise ValueError("match stream body does not match match_count")
    matches = tuple(
        MatchEntry(
            id=int.from_bytes(body[i : i + 4], "big"),
            delta_t=int.from_bytes(body[i + 4 : i + 12], "big", signed=True),
            votes=int.from_bytes(body[i + 12 : i + 16], "big"),
            total_votes=int.from_bytes(body[i + 16 : i + 20], "big"),
        )
        for i in range(0, len(body), 20)
    )
    return MatchTable(
        query_frames=int.from_bytes(raw[0:4], "big"),
        query_peak_count=int.from_bytes(raw[4:8], "big"),
        matches=matches,
        raw=raw,
    )


def mono_average(interleaved, channels):
    """The explicit mono average of an interleaved stream — mirrors
    ``pith_audio::reference::mono_average``: the per-frame sum
    accumulates in i64 and divides by the channel count **truncating
    toward zero** (Rust integer division), never by reimplementing the
    FFT pipeline.
    """
    ch = int(channels)
    out = []
    for i in range(0, len(interleaved) - ch + 1, ch):
        acc = 0
        for c in range(ch):
            acc += interleaved[i + c]
        q = abs(acc) // ch
        out.append(q if acc >= 0 else -q)
    return tuple(out)
