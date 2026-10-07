// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
"use strict";

/**
 * pith-audio SDK: WAV decode, spectral-peak signatures and delta-t
 * histogram matching through koffi.
 *
 * The single Rust core (the `pith-audio` cdylib built by
 * `cargo build --release`) is loaded at runtime; `koffi` is the only
 * dependency.
 *
 * Discovery order (the suite's cdylib convention):
 *
 * 1. `PITH_CDYLIB` — an explicit cdylib *file* path;
 * 2. `PITH_CDYLIB_DIR` — a *directory* scanned for the cdylib names
 *    (the CD pipeline points this at `target/release`);
 * 3. `prebuilds/<platform>-<arch>/` then `prebuilds/` flat (the
 *    packaged layout);
 * 4. `<repo root>/target/release` — the repository working-tree
 *    layout, so a source checkout runs against a local cargo build
 *    with no configuration.
 *
 * The FFI surface (all byte streams are big-endian headers plus
 * little-endian i32 sample/record bodies — see the parse helpers):
 * `pith_audio_decode_wav` produces the **decode stream** the
 * `pcm_i32_le_*` digests are defined over; `pith_audio_signature_wav`
 * / `pith_audio_signature_pcm` produce the **signature stream**;
 * `pith_audio_voices_pcm` synthesizes the SplitMix64 square-voice
 * corpus *inside the cdylib* (the same pure recipe the synthetic
 * reference.json vectors are built from — no SDK-side float
 * reimplementation); `pith_audio_index_new` / `pith_audio_index_add`
 * / `pith_audio_match` build an incremental signature index (ids are
 * insertion order) and produce the **match stream**; `pith_audio_free`
 * / `pith_audio_index_free` release everything the FFI handed out.
 */

const koffi = require("koffi");
const fs = require("node:fs");
const path = require("node:path");

const STATUS_OK = 0;
const STATUS_INVALID = -1;
const STATUS_REJECTED = -2;

/** Every cdylib file name cargo may drop into the build directory, per platform. */
const CDYLIB_NAMES = ["pith_audio.dll", "libpith_audio.so", "libpith_audio.dylib"];

const PKG_ROOT = path.join(__dirname);
const REPO_ROOT = path.resolve(__dirname, "..", "..");

/** FfiError: a non-zero status code came back from the cdylib. */
class FfiError extends Error {
  /**
   * @param {string} op the FFI operation name
   * @param {number} status the raw status code
   */
  constructor(op, status) {
    const kind = { [STATUS_INVALID]: "invalid argument", [STATUS_REJECTED]: "input rejected" }[status] ?? "unknown failure";
    super(`${op} failed: ${kind} (status ${status})`);
    this.name = "FfiError";
    /** The raw status code the FFI returned. */
    this.status = status;
  }
}

/**
 * Locates the cdylib through the suite's discovery chain.
 * @returns {string} an absolute path to the cdylib file
 * @throws {Error} when nothing is found
 */
function findCdylib() {
  const explicit = process.env.PITH_CDYLIB;
  if (explicit && fs.statSync(explicit, { throwIfNoEntry: false })?.isFile()) {
    return path.resolve(explicit);
  }
  /** @type {string[]} */
  const dirs = [];
  const envDir = process.env.PITH_CDYLIB_DIR;
  if (envDir) {
    dirs.push(envDir);
    if (!path.isAbsolute(envDir)) {
      dirs.push(path.join(REPO_ROOT, envDir));
    }
  }
  const osArch = `${process.platform}-${process.arch}`;
  dirs.push(path.join(PKG_ROOT, "prebuilds", osArch));
  dirs.push(path.join(PKG_ROOT, "prebuilds"));
  dirs.push(path.join(REPO_ROOT, "target", "release"));
  for (const dir of dirs) {
    for (const name of CDYLIB_NAMES) {
      const p = path.join(dir, name);
      if (fs.statSync(p, { throwIfNoEntry: false })?.isFile()) return p;
    }
  }
  throw new Error(
    "no pith-audio cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR, prebuilds/ and <repo>/target/release); " +
      "run `cargo build --release` first",
  );
}

let cached = undefined;

/**
 * Loads the cdylib and binds the exported symbols (lazily, once).
 * @returns {{decodeWav: Function, signatureWav: Function, signaturePcm: Function,
 *   voicesPcm: Function, indexNew: Function, indexAdd: Function, match: Function,
 *   indexFree: Function, free: Function}}
 */
function loadLibrary() {
  if (cached) return cached;
  const lib = koffi.load(findCdylib());
  const bytesOut = (name, params) =>
    lib.func(name, "int32_t", [
      ...params,
      koffi.out(koffi.pointer("void *")),
      koffi.out(koffi.pointer("size_t")),
    ]);
  const decodeWav = bytesOut("pith_audio_decode_wav", ["const uint8_t *", "size_t"]);
  const signatureWav = bytesOut("pith_audio_signature_wav", ["const uint8_t *", "size_t"]);
  const signaturePcm = bytesOut("pith_audio_signature_pcm", [
    "const uint8_t *",
    "size_t",
    "uint32_t",
  ]);
  const voicesPcm = bytesOut("pith_audio_voices_pcm", ["uint64_t", "size_t"]);
  const indexNew = lib.func("pith_audio_index_new", "void *", []);
  const indexAdd = lib.func("pith_audio_index_add", "int32_t", [
    "void *",
    "const uint8_t *",
    "size_t",
    "uint32_t",
  ]);
  const match = bytesOut("pith_audio_match", [
    "void *",
    "const uint8_t *",
    "size_t",
    "uint32_t",
  ]);
  const indexFree = lib.func("void pith_audio_index_free(void *idx)");
  const free = lib.func("void pith_audio_free(void *ptr, size_t len)");
  cached = { decodeWav, signatureWav, signaturePcm, voicesPcm, indexNew, indexAdd, match, indexFree, free };
  return cached;
}

/**
 * Runs one byte-stream FFI op and copies the handed-out buffer into a
 * JS Buffer before the cdylib's copy is freed.
 *
 * @param {string} op the FFI operation name (for errors)
 * @param {Function} fn the bound koffi function
 * @param {unknown[]} args the op's leading arguments
 * @returns {Buffer} the canonical stream
 * @throws {FfiError} on a non-zero status
 */
function bytesOut(op, fn, ...args) {
  const { free } = loadLibrary();
  const out = [null];
  const outLen = [0];
  const status = fn(...args, out, outLen);
  if (status !== STATUS_OK) {
    throw new FfiError(op, status);
  }
  try {
    // koffi.decode hands back a Uint8Array view over the external
    // buffer; copy it into a Buffer before the cdylib buffer is freed.
    return Buffer.from(koffi.decode(out[0], "uint8_t", Number(outLen[0])));
  } finally {
    free(out[0], Number(outLen[0]));
  }
}

/** Validates the PCM argument shape the FFI enforces (u16 channels, nonzero). */
function checkChannels(op, channels) {
  if (!Number.isInteger(channels) || channels <= 0 || channels > 0xffff) {
    throw new FfiError(op, STATUS_INVALID);
  }
}

/**
 * Decodes a complete RIFF/WAVE stream into the decode stream the
 * `pcm_i32_le_sha256` / `pcm_i32_le_fnv1a64` digests are defined over
 * (16-byte big-endian header, then i32 LE samples).
 *
 * @param {Buffer} data the complete WAV file bytes
 * @returns {Buffer} the decode stream
 * @throws {FfiError} with `status === -2` for any malformed input
 */
function decodeWav(data) {
  if (!Buffer.isBuffer(data)) {
    throw new TypeError("data must be a Buffer");
  }
  return bytesOut("pith_audio_decode_wav", loadLibrary().decodeWav, data, data.length);
}

/**
 * Decodes a RIFF/WAVE stream and extracts its signature (the
 * `signature_of_wav` facade: anything but 44100 Hz is refused) into
 * the signature stream.
 *
 * @param {Buffer} data the complete WAV file bytes
 * @returns {Buffer} the signature stream
 * @throws {FfiError} with `status === -2` for malformed input or a
 *   refused sample rate
 */
function signatureWav(data) {
  if (!Buffer.isBuffer(data)) {
    throw new TypeError("data must be a Buffer");
  }
  return bytesOut("pith_audio_signature_wav", loadLibrary().signatureWav, data, data.length);
}

/**
 * Extracts a signature from interleaved little-endian i32 PCM at
 * 44100 Hz into the signature stream.
 *
 * @param {Buffer} data the interleaved i32 LE PCM
 * @param {number} channels the channel count (nonzero)
 * @returns {Buffer} the signature stream
 * @throws {FfiError} with `status === -1` for bad geometry
 */
function signaturePcm(data, channels) {
  if (!Buffer.isBuffer(data)) {
    throw new TypeError("data must be a Buffer");
  }
  checkChannels("pith_audio_signature_pcm", channels);
  return bytesOut("pith_audio_signature_pcm", loadLibrary().signaturePcm, data, data.length, channels);
}

/**
 * Synthesizes the SplitMix64 square-voice corpus *inside the cdylib*:
 * `nSamples` interleaved i32 LE samples, exactly the bytes the
 * synthetic reference.json vectors are built from. The recipe is pure
 * integer/exact-float math, so no SDK ever reimplements it and no
 * platform sensitivity enters the vectors.
 *
 * @param {number|bigint} seed the SplitMix64 seed
 * @param {number} nSamples the sample count
 * @returns {Buffer} the i32 LE PCM
 */
function voicesPcm(seed, nSamples) {
  return bytesOut("pith_audio_voices_pcm", loadLibrary().voicesPcm, seed, nSamples);
}

/**
 * An incremental signature index: PCM signatures accumulate in
 * insertion order, and the id a match reports is that order.
 */
class AudioIndex {
  constructor() {
    /** @type {unknown} the opaque cdylib handle (never null here) */
    this.handle = loadLibrary().indexNew();
    /** @type {boolean} */
    this.closed = false;
  }

  /**
   * Appends the signature of interleaved i32 LE PCM to the index.
   * A failed add leaves the index alive and unchanged.
   *
   * @param {Buffer} data the interleaved i32 LE PCM
   * @param {number} channels the channel count (nonzero)
   * @throws {FfiError} on refusal (-2) or bad geometry (-1)
   */
  add(data, channels) {
    if (this.closed || !this.handle) {
      throw new FfiError("pith_audio_index_add", STATUS_INVALID);
    }
    checkChannels("pith_audio_index_add", channels);
    const status = loadLibrary().indexAdd(this.handle, data, data.length, channels);
    if (status !== STATUS_OK) {
      throw new FfiError("pith_audio_index_add", status);
    }
  }

  /**
   * Matches the signature of interleaved i32 LE PCM against the index
   * and returns the match stream. An empty index is legal and yields
   * zero matches.
   *
   * @param {Buffer} data the interleaved i32 LE PCM
   * @param {number} channels the channel count (nonzero)
   * @returns {Buffer} the match stream
   * @throws {FfiError} on refusal (-2) or bad geometry (-1)
   */
  match(data, channels) {
    if (this.closed || !this.handle) {
      throw new FfiError("pith_audio_match", STATUS_INVALID);
    }
    checkChannels("pith_audio_match", channels);
    return bytesOut("pith_audio_match", loadLibrary().match, this.handle, data, data.length, channels);
  }

  /** Releases the handle; idempotent. */
  close() {
    if (!this.closed && this.handle) {
      loadLibrary().indexFree(this.handle);
      this.closed = true;
      this.handle = null;
    }
  }
}

/**
 * Re-expresses the decode stream as a plain object. `samples` are the
 * decoded i32 values — exactly the bytes the `pcm_i32_le_*` digests
 * cover (`raw.subarray(16)`).
 *
 * @param {Buffer} raw the decode stream
 * @returns {{sampleRate: number, channels: number, bitsPerSample: number,
 *   frames: number, decodedSamples: number, samples: number[], raw: Buffer}}
 */
function parseDecodeStream(raw) {
  if (!Buffer.isBuffer(raw) || raw.length < 16) {
    throw new TypeError("decode stream is shorter than the 16-byte header");
  }
  const decodedSamples = raw.readUInt32BE(12);
  if (raw.length !== 16 + decodedSamples * 4) {
    throw new TypeError("decode stream body does not match decodedSamples");
  }
  const samples = [];
  for (let i = 0; i < decodedSamples; i++) {
    samples.push(raw.readInt32LE(16 + i * 4));
  }
  return {
    sampleRate: raw.readUInt32BE(0),
    channels: raw.readUInt16BE(4),
    bitsPerSample: raw.readUInt16BE(6),
    frames: raw.readUInt32BE(8),
    decodedSamples,
    samples,
    raw,
  };
}

/**
 * Re-expresses the signature stream as a plain object.
 *
 * @param {Buffer} raw the signature stream
 * @returns {{frames: number, peakCount: number, fingerprint: bigint,
 *   peaks: Array<[number, number]>, raw: Buffer}}
 */
function parseSignatureStream(raw) {
  if (!Buffer.isBuffer(raw) || raw.length < 16) {
    throw new TypeError("signature stream is shorter than its 16-byte header");
  }
  const peakCount = raw.readUInt32BE(4);
  if (raw.length !== 16 + peakCount * 6) {
    throw new TypeError("signature stream body does not match peakCount");
  }
  const peaks = [];
  for (let i = 0; i < peakCount; i++) {
    peaks.push([raw.readUInt32BE(16 + i * 6), raw.readUInt16BE(16 + i * 6 + 4)]);
  }
  return {
    frames: raw.readUInt32BE(0),
    peakCount,
    fingerprint: raw.readBigUInt64BE(8),
    peaks,
    raw,
  };
}

/**
 * Re-expresses the match stream as a plain object.
 *
 * @param {Buffer} raw the match stream
 * @returns {{queryFrames: number, queryPeakCount: number, matches: Array<{
 *   id: number, deltaT: number, votes: number, totalVotes: number}>, raw: Buffer}}
 */
function parseMatchStream(raw) {
  if (!Buffer.isBuffer(raw) || raw.length < 12) {
    throw new TypeError("match stream is shorter than the 12-byte header");
  }
  const matchCount = raw.readUInt32BE(8);
  if (raw.length !== 12 + matchCount * 20) {
    throw new TypeError("match stream body does not match matchCount");
  }
  const matches = [];
  for (let i = 0; i < matchCount; i++) {
    const at = 12 + i * 20;
    matches.push({
      id: raw.readUInt32BE(at),
      deltaT: Number(raw.readBigInt64BE(at + 4)),
      votes: raw.readUInt32BE(at + 12),
      totalVotes: raw.readUInt32BE(at + 16),
    });
  }
  return {
    queryFrames: raw.readUInt32BE(0),
    queryPeakCount: raw.readUInt32BE(4),
    matches,
    raw,
  };
}

/**
 * The explicit mono average of an interleaved stream — mirrors
 * `pith_audio::reference::mono_average`: the per-frame sum accumulates
 * exactly (i32 sums fit a double) and divides by the channel count
 * truncating toward zero (Rust integer division semantics).
 *
 * @param {number[]} interleaved the interleaved i32 samples
 * @param {number} channels the channel count
 * @returns {number[]} the mono i32 samples
 */
function monoAverage(interleaved, channels) {
  const out = new Array(Math.floor(interleaved.length / channels));
  for (let i = 0; i < out.length; i++) {
    let acc = 0;
    for (let c = 0; c < channels; c++) {
      acc += interleaved[i * channels + c];
    }
    out[i] = Math.trunc(acc / channels);
  }
  return out;
}

/**
 * Interleaves two i32 sample arrays into one stereo array.
 *
 * @param {number[]} left the left-channel samples
 * @param {number[]} right the right-channel samples
 * @returns {number[]} the interleaved samples
 */
function interleave(left, right) {
  const out = new Array(left.length * 2);
  for (let i = 0; i < left.length; i++) {
    out[2 * i] = left[i];
    out[2 * i + 1] = right[i];
  }
  return out;
}

module.exports = {
  STATUS_OK,
  STATUS_INVALID,
  STATUS_REJECTED,
  CDYLIB_NAMES,
  FfiError,
  findCdylib,
  decodeWav,
  signatureWav,
  signaturePcm,
  voicesPcm,
  AudioIndex,
  parseDecodeStream,
  parseSignatureStream,
  parseMatchStream,
  monoAverage,
  interleave,
};
