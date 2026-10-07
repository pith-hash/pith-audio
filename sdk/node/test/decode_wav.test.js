// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
"use strict";

// Hex-exact decode conformance: every decode vector in the
// repository-root reference.json is replayed through the cdylib and
// compared byte-exact — the recorded WAV facts plus the
// pcm_i32_le_* digests over the decode-stream body. The same vectors
// the Rust gen-reference verify gate and the Python/Go SDKs check.

const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const { FfiError, decodeWav, findCdylib, parseDecodeStream } = require("../index.js");

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

const REFERENCE = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, "reference.json"), "utf8"));
const VECTORS = new Map(REFERENCE.vectors.map((v) => [v.name, v]));
const DECODE_VECTORS = [
  "fixture-tone-wav",
  "pcm8-offset-centred",
  "pcm24-sign-extension",
  "float32-clamp-and-nan",
  "float64-halves-and-thirds",
  "riff-odd-chunk-pad",
  "partial-frame-dropped",
];

/** The vector's input bytes: the committed fixture or the inline hex. */
function vectorBytes(vector) {
  if (vector.input_kind === "fixture-file") {
    return fs.readFileSync(path.join(REPO_ROOT, "tests", "fixtures", "tone.wav"));
  }
  return Buffer.from(vector.input_hex, "hex");
}

/** Standard FNV-1a 64 — the digest pcm_i32_le_fnv1a64 pins. */
function fnv1a64(data) {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  for (const b of data) {
    h ^= BigInt(b);
    h = (h * prime) & 0xffffffffffffffffn;
  }
  return h;
}

test("cdylib is discoverable", () => {
  assert.ok(fs.statSync(findCdylib()).isFile());
});

for (const name of DECODE_VECTORS) {
  const vector = VECTORS.get(name);
  test(`reference vector ${name} is reproduced hex-exact`, () => {
    const facts = parseDecodeStream(decodeWav(vectorBytes(vector)));

    assert.equal(facts.sampleRate, vector.sample_rate, name);
    assert.equal(facts.channels, vector.channels, name);
    assert.equal(facts.bitsPerSample, vector.bits_per_sample, name);
    assert.equal(facts.frames, vector.frames, name);
    assert.equal(facts.decodedSamples, vector.decoded_samples, name);
    if (vector.samples_i32) {
      assert.deepEqual(facts.samples, vector.samples_i32, name);
    }
    const body = facts.raw.subarray(16);
    assert.equal(crypto.createHash("sha256").update(body).digest("hex"), vector.pcm_i32_le_sha256, name);
    assert.equal(fnv1a64(body).toString(16).padStart(16, "0"), vector.pcm_i32_le_fnv1a64, name);
  });
}

test("malformed input is refused, not crashing", () => {
  assert.throws(() => decodeWav(Buffer.from("not a wav at all........")), (err) => {
    assert.ok(err instanceof FfiError);
    assert.equal(err.status, -2);
    return true;
  });
});

test("empty input is refused", () => {
  assert.throws(() => decodeWav(Buffer.alloc(0)), FfiError);
});
