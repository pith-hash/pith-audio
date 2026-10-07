// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
"use strict";

// Signature conformance for PCM inputs: the three synthetic-voices
// vectors. The corpora are synthesized INSIDE the cdylib via
// voices_pcm (SplitMix64 square voices — a pure integer/exact-float
// recipe), so no SDK-side float reimplementation exists and the vectors
// carry zero platform sensitivity. Stereo interleave and the explicit
// mono average are the only SDK-side mixdown steps, mirroring
// reference::mono_average (i64 sum, truncating division).

const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const {
  FfiError,
  interleave,
  monoAverage,
  parseSignatureStream,
  signaturePcm,
  voicesPcm,
} = require("../index.js");

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

const REFERENCE = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, "reference.json"), "utf8"));
const VECTORS = new Map(REFERENCE.vectors.map((v) => [v.name, v]));

/** Peaks as (t u32 LE, f u16 LE) — the exact layout the peaks_sha256
 * digest hashes. */
function peaksLeBytes(peaks) {
  const out = Buffer.alloc(peaks.length * 6);
  for (const [i, [t, f]] of peaks.entries()) {
    out.writeUInt32LE(t, i * 6);
    out.writeUInt16LE(f, i * 6 + 4);
  }
  return out;
}

/** Asserts a signature stream against its committed vector. */
function assertHexExact(name, stream) {
  const vector = VECTORS.get(name);
  const facts = parseSignatureStream(stream);
  assert.equal(facts.frames, vector.frames, name);
  assert.equal(facts.peakCount, vector.peak_count, name);
  assert.deepEqual(facts.peaks, vector.peaks, name);
  assert.equal(facts.fingerprint.toString(16).padStart(16, "0"), vector.fingerprint, name);
  assert.equal(
    crypto.createHash("sha256").update(peaksLeBytes(facts.peaks)).digest("hex"),
    vector.peaks_sha256,
    name,
  );
}

test("voices-signature-mono is reproduced hex-exact", () => {
  assertHexExact("voices-signature-mono", signaturePcm(voicesPcm(0x00c0ffee, 66150), 1));
});

test("voices-stereo-mixdown is reproduced hex-exact", () => {
  const left = readI32(voicesPcm(0x00000011, 88200));
  const right = readI32(voicesPcm(0x00000022, 88200));
  assertHexExact("voices-stereo-mixdown", signaturePcm(writeI32(interleave(left, right)), 2));
});

test("voices-stereo-mono-average is reproduced hex-exact", () => {
  const left = readI32(voicesPcm(0x00000011, 88200));
  const right = readI32(voicesPcm(0x00000022, 88200));
  const mono = monoAverage(interleave(left, right), 2);
  assertHexExact("voices-stereo-mono-average", signaturePcm(writeI32(mono), 1));
});

test("bad pcm arguments are invalid", () => {
  assert.throws(
    () => signaturePcm(Buffer.from([1, 0, 0]), 1),
    (err) => err instanceof FfiError && err.status === -1,
  );
  assert.throws(
    () => signaturePcm(Buffer.alloc(4), 0),
    (err) => err instanceof FfiError && err.status === -1,
  );
});

/** Reads interleaved i32 LE samples from a buffer. */
function readI32(buf) {
  const out = new Array(buf.length / 4);
  for (let i = 0; i < out.length; i++) {
    out[i] = buf.readInt32LE(i * 4);
  }
  return out;
}

/** Writes i32 samples as interleaved i32 LE bytes. */
function writeI32(samples) {
  const out = Buffer.alloc(samples.length * 4);
  samples.forEach((s, i) => out.writeInt32LE(s, i * 4));
  return out;
}
