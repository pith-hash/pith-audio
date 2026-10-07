# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pith-hash
"""Hex-exact conformance: the committed reference vectors through ctypes.

Every vector in the repository-root ``reference.json`` is replayed
through the cdylib and compared byte-exact — decode facts plus the
``pcm_i32_le_*`` digests over the decode stream body, signature peaks /
fingerprint / ``peaks_sha256`` recomputed from the LE re-serialization,
and the full offset-match table. The synthetic corpora are rebuilt via
``voices_pcm`` (synthesis runs inside the cdylib — no SDK-side float
reimplementation). The same vectors the Rust ``gen-reference verify``
gate and the Node/Go SDKs check.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from pith_audio import (
    AudioIndex,
    FfiError,
    decode_wav,
    find_cdylib,
    mono_average,
    parse_decode_stream,
    parse_match_stream,
    parse_signature_stream,
    signature_pcm,
    signature_wav,
    voices_pcm,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

REFERENCE = json.loads((REPO_ROOT / "reference.json").read_text(encoding="utf-8"))
VECTORS = {v["name"]: v for v in REFERENCE["vectors"]}
DECODE_VECTORS = [
    "fixture-tone-wav",
    "pcm8-offset-centred",
    "pcm24-sign-extension",
    "float32-clamp-and-nan",
    "float64-halves-and-thirds",
    "riff-odd-chunk-pad",
    "partial-frame-dropped",
]
SIGNATURE_VECTORS = [
    "fixture-tone-signature",
    "voices-signature-mono",
    "voices-stereo-mixdown",
    "voices-stereo-mono-average",
]


def vector_bytes(vector: dict) -> bytes:
    """The vector's input bytes: the committed fixture or the inline
    hex."""
    if vector["input_kind"] == "fixture-file":
        return (REPO_ROOT / "tests" / "fixtures" / "tone.wav").read_bytes()
    return bytes.fromhex(vector["input_hex"])


def fnv1a64(data: bytes) -> int:
    """Standard FNV-1a 64 — the digest ``pcm_i32_le_fnv1a64`` pins."""
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def peaks_le_bytes(peaks) -> bytes:
    """Peaks as (t u32 LE, f u16 LE) — the exact layout the
    ``peaks_sha256`` digest hashes (the FFI stream carries them
    big-endian; the digest is over the LE form)."""
    return b"".join(struct.pack("<IH", t, f) for (t, f) in peaks)


def test_cdylib_is_discoverable() -> None:
    path = find_cdylib()
    assert path.is_file(), path


@pytest.mark.parametrize("name", DECODE_VECTORS)
def test_decode_vector_is_reproduced_hex_exact(name: str) -> None:
    vector = VECTORS[name]
    facts = parse_decode_stream(decode_wav(vector_bytes(vector)))

    assert facts.sample_rate == vector["sample_rate"], name
    assert facts.channels == vector["channels"], name
    assert facts.bits_per_sample == vector["bits_per_sample"], name
    assert facts.frames == vector["frames"], name
    assert facts.decoded_samples == vector["decoded_samples"], name
    if "samples_i32" in vector:
        assert list(facts.samples) == vector["samples_i32"], name
    body = facts.raw[16:]
    assert hashlib.sha256(body).hexdigest() == vector["pcm_i32_le_sha256"], name
    assert f"{fnv1a64(body):016x}" == vector["pcm_i32_le_fnv1a64"], name


def interleave(left: bytes, right: bytes) -> bytes:
    """Interleaves two i32-LE mono buffers (bytes) into a stereo
    interleaved buffer."""
    n = len(left) // 4
    l = struct.unpack(f"<{n}i", left)
    r = struct.unpack(f"<{n}i", right)
    mixed = [v for pair in zip(l, r) for v in pair]
    return struct.pack(f"<{2 * n}i", *mixed)


@pytest.mark.parametrize("name", SIGNATURE_VECTORS)
def test_signature_vector_is_reproduced_hex_exact(name: str) -> None:
    vector = VECTORS[name]
    if name == "fixture-tone-signature":
        stream = signature_wav(vector_bytes(vector))
    elif name == "voices-signature-mono":
        stream = signature_pcm(voices_pcm(0x00C0FFEE, 66_150), 1)
    else:
        # The stereo pair: pipeline mixdown vs explicit mono average —
        # both must land on the same committed signature.
        stereo = interleave(voices_pcm(0x00000011, 88_200), voices_pcm(0x00000022, 88_200))
        if name == "voices-stereo-mixdown":
            stream = signature_pcm(stereo, 2)
        else:
            mono = mono_average(struct.unpack(f"<{len(stereo) // 4}i", stereo), 2)
            stream = signature_pcm(struct.pack(f"<{len(mono)}i", *mono), 1)

    facts = parse_signature_stream(stream)
    assert facts.frames == vector["frames"], name
    assert facts.peak_count == vector["peak_count"], name
    assert [list(p) for p in facts.peaks] == vector["peaks"], name
    assert f"{facts.fingerprint:016x}" == vector["fingerprint"], name
    assert (
        hashlib.sha256(peaks_le_bytes(facts.peaks)).hexdigest() == vector["peaks_sha256"]
    ), name


def test_offset_match_vector_is_reproduced_hex_exact() -> None:
    vector = VECTORS["voices-offset-match"]
    # Query = the 2 s prefix followed by the 3 s indexed content.
    query = voices_pcm(0x0000BEEF, 88_200) + voices_pcm(0x0000D1CE, 132_300)
    index = AudioIndex()
    try:
        for seed, n in [
            (0x0000D1CE, 132_300),
            (0x0000F00D, 132_300),
            (0x0000E5A5, 44_100),
        ]:
            index.add(voices_pcm(seed, n), 1)
        table = parse_match_stream(index.match(query, 1))
    finally:
        index.close()

    assert table.query_frames == vector["query_frames"]
    assert table.query_peak_count == vector["query_peak_count"]
    assert [
        {"id": m.id, "delta_t": m.delta_t, "votes": m.votes, "total_votes": m.total_votes}
        for m in table.matches
    ] == vector["matches"]
    assert table.matches[0].id == 0, "the content member must win"


def test_full_stream_matches_a_rust_pinned_value() -> None:
    # fixture-tone-signature's facts, pinned in the committed
    # reference.json and re-derived by the Rust unit tests; this test
    # fails loudly even if reference.json were regenerated wrongly.
    wav = (REPO_ROOT / "tests" / "fixtures" / "tone.wav").read_bytes()
    facts = parse_signature_stream(signature_wav(wav))
    assert facts.frames == 20
    assert facts.peak_count == 42
    assert f"{facts.fingerprint:016x}" == "0400000002001425"
    assert (
        hashlib.sha256(peaks_le_bytes(facts.peaks)).hexdigest()
        == "8c6ac0719bb7caed61e66f978da85ab5b66639aa279457d8f88731be2dea1ec8"
    )


@pytest.mark.parametrize(
    "name",
    ["short-riff-header", "not-a-wav", "adpcm-format-tag", "wrong-sample-rate-facade"],
)
def test_error_vector_is_refused_not_crashing(name: str) -> None:
    vector = next(v for v in REFERENCE["error_vectors"] if v["name"] == name)
    data = bytes.fromhex(vector["input_hex"])
    op = decode_wav if vector["api"] == "wav.decode" else signature_wav
    with pytest.raises(FfiError) as err:
        op(data)
    assert err.value.status == -2, name


def test_bad_pcm_arguments_are_invalid() -> None:
    with pytest.raises(FfiError) as err:
        signature_pcm(b"\x01\x00\x00", 1)  # not a whole i32 sample
    assert err.value.status == -1
    with pytest.raises(FfiError) as err:
        signature_pcm(b"\x00\x00\x00\x00", 0)  # zero channels
    assert err.value.status == -1


def test_failed_add_leaves_the_index_alive() -> None:
    index = AudioIndex()
    try:
        with pytest.raises(FfiError) as err:
            index.add(b"\x01\x00\x00", 1)  # partial sample: caller bug
        assert err.value.status == -1
        # The handle survived: a real corpus lands, and the match runs.
        index.add(voices_pcm(0x00C0FFEE, 66_150), 1)
        table = parse_match_stream(index.match(voices_pcm(0x00C0FFEE, 66_150), 1))
        assert table.matches[0].id == 0
    finally:
        index.close()
