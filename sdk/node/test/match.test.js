// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
"use strict";

// Matching conformance: the committed voices-offset-match vector
// through the index triad (new / add x3 / match), plus the empty-index
// legality, refusal paths and handle lifetime.

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const { AudioIndex, FfiError, parseMatchStream, voicesPcm } = require("../index.js");

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

const REFERENCE = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, "reference.json"), "utf8"));
const VECTOR = REFERENCE.vectors.find((v) => v.name === "voices-offset-match");

test("voices-offset-match is reproduced hex-exact", () => {
  // Query = the 2 s prefix followed by the 3 s indexed content.
  const query = Buffer.concat([voicesPcm(0x0000beef, 88200), voicesPcm(0x0000d1ce, 132300)]);
  const index = new AudioIndex();
  try {
    for (const [seed, n] of [
      [0x0000d1ce, 132300],
      [0x0000f00d, 132300],
      [0x0000e5a5, 44100],
    ]) {
      index.add(voicesPcm(seed, n), 1);
    }
    const table = parseMatchStream(index.match(query, 1));
    assert.equal(table.queryFrames, VECTOR.query_frames);
    assert.equal(table.queryPeakCount, VECTOR.query_peak_count);
    assert.deepEqual(
      table.matches.map((m) => ({ id: m.id, delta_t: m.deltaT, votes: m.votes, total_votes: m.totalVotes })),
      VECTOR.matches,
    );
    assert.equal(table.matches[0].id, 0, "the content member must win");
  } finally {
    index.close();
  }
});

test("empty index is legal and matches nothing", () => {
  const index = new AudioIndex();
  try {
    const table = parseMatchStream(index.match(voicesPcm(7, 8192), 1));
    assert.equal(table.matches.length, 0);
    assert.deepEqual(table.matches, []);
  } finally {
    index.close();
  }
});

test("failed add leaves the index alive", () => {
  const index = new AudioIndex();
  try {
    assert.throws(
      () => index.add(Buffer.from([1, 0, 0]), 1),
      (err) => err instanceof FfiError && err.status === -1,
    );
    index.add(voicesPcm(0x00c0ffee, 66150), 1);
    const table = parseMatchStream(index.match(voicesPcm(0x00c0ffee, 66150), 1));
    assert.equal(table.matches[0].id, 0);
  } finally {
    index.close();
  }
});

test("refusals and handle lifetime", () => {
  const pcm = voicesPcm(7, 8192);
  const index = new AudioIndex();
  assert.throws(
    () => index.add(pcm, 0),
    (err) => err instanceof FfiError && err.status === -1,
  );
  index.close();
  assert.throws(() => index.add(pcm, 1), FfiError);
  assert.throws(() => index.match(pcm, 1), FfiError);
  index.close(); // idempotent
});
