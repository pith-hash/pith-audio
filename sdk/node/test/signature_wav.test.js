// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
"use strict";

// Signature conformance for WAV inputs: the committed fixture vector
// through signature_wav, including the rust-pinned literal values, and
// the facade refusal paths (non-44100 Hz, garbage).

const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const { FfiError, parseSignatureStream, signatureWav } = require("../index.js");

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

const REFERENCE = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, "reference.json"), "utf8"));
const VECTORS = new Map(REFERENCE.vectors.map((v) => [v.name, v]));
const ERROR_VECTORS = new Map(REFERENCE.error_vectors.map((v) => [v.name, v]));

/** Peaks as (t u32 LE, f u16 LE) — the exact layout the peaks_sha256
 * digest hashes (the FFI stream carries them big-endian; the digest is
 * over the LE form). */
function peaksLeBytes(peaks) {
  const out = Buffer.alloc(peaks.length * 6);
  for (const [i, [t, f]] of peaks.entries()) {
    out.writeUInt32LE(t, i * 6);
    out.writeUInt16LE(f, i * 6 + 4);
  }
  return out;
}

test("fixture-tone-signature is reproduced hex-exact", () => {
  const vector = VECTORS.get("fixture-tone-signature");
  const wav = fs.readFileSync(path.join(REPO_ROOT, "tests", "fixtures", "tone.wav"));

  const facts = parseSignatureStream(signatureWav(wav));
  assert.equal(facts.frames, vector.frames);
  assert.equal(facts.peakCount, vector.peak_count);
  assert.deepEqual(facts.peaks, vector.peaks);
  assert.equal(facts.fingerprint.toString(16).padStart(16, "0"), vector.fingerprint);
  assert.equal(crypto.createHash("sha256").update(peaksLeBytes(facts.peaks)).digest("hex"), vector.peaks_sha256);
});

test("full stream matches a rust-pinned value", () => {
  // fixture-tone-signature's facts, pinned in the committed
  // reference.json and re-derived by the Rust unit tests; this test
  // fails loudly even if reference.json were regenerated wrongly.
  const wav = fs.readFileSync(path.join(REPO_ROOT, "tests", "fixtures", "tone.wav"));
  const facts = parseSignatureStream(signatureWav(wav));
  assert.equal(facts.frames, 20);
  assert.equal(facts.peakCount, 42);
  assert.equal(facts.fingerprint.toString(16).padStart(16, "0"), "0400000002001425");
  assert.equal(
    crypto.createHash("sha256").update(peaksLeBytes(facts.peaks)).digest("hex"),
    "8c6ac0719bb7caed61e66f978da85ab5b66639aa279457d8f88731be2dea1ec8",
  );
});

for (const name of ["wrong-sample-rate-facade", "garbage-input-facade"]) {
  test(`error vector ${name} is refused, not crashing`, () => {
    const vector = ERROR_VECTORS.get(name);
    assert.throws(
      () => signatureWav(Buffer.from(vector.input_hex, "hex")),
      (err) => {
        assert.ok(err instanceof FfiError);
        assert.equal(err.status, -2);
        return true;
      },
    );
  });
}
