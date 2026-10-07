//! The C ABI surface of `pith-audio`: the entry points the Python
//! (ctypes), Node (koffi) and Go (cgo) SDKs bind through.
//!
//! The suite's FFI convention, defined by this module and mirrored by
//! every `pith-*` cdylib:
//!
//! * one flat set of `#[unsafe(no_mangle)] pub unsafe extern "C"`
//!   functions — raw pointers plus lengths, no structs across the
//!   boundary;
//! * every function returns a status code (see the constants below),
//!   never a `Result`, never a panic: a `panic = "abort"` cdylib must
//!   not be reachable from a foreign caller;
//! * an operation either hands ownership to the caller (and ships a
//!   matching `_free`) or writes into caller-provided out-parameters;
//! * the `unsafe` allowance is confined to this module; every core
//!   module stays unsafe-free behind the crate-root `#![deny]`.
//!
//! Only the operations the `reference.json` vectors pin are exposed.
//! `signature_of_flac` stays Rust-only: no committed vector exercises
//! FLAC through the facade, and an unpinned export is an unverified
//! contract.
//!
//! Wire formats (the byte contracts the SDK tests replay; the exact
//! layouts also live in [`crate::reference`]'s module docs):
//!
//! **decode stream** (`pith_audio_decode_wav`): `[0..4] sample_rate`
//! u32 BE, `[4..6] channels` u16 BE, `[6..8] bits_per_sample` u16 BE,
//! `[8..12] frames` u32 BE, `[12..16] decoded_samples` u32 BE,
//! `[16..]` samples i32 LE — `pcm_i32_le_sha256 == sha256(stream[16..])`.
//!
//! **signature stream** (`pith_audio_signature_wav`,
//! `pith_audio_signature_pcm`): `[0..4] frames` u32 BE, `[4..8]
//! peak_count` u32 BE, `[8..16] fingerprint` u64 BE, then per peak
//! `t` u32 BE, `f` u16 BE. `peaks_sha256` hashes the LITTLE-endian
//! re-serialization (t u32 LE, f u16 LE per peak).
//!
//! **match stream** (`pith_audio_match`): `[0..4] query_frames` u32
//! BE, `[4..8] query_peak_count` u32 BE, `[8..12] match_count` u32 BE,
//! then per match `id` u32 BE, `delta_t` i64 BE, `votes` u32 BE,
//! `total_votes` u32 BE.
//!
//! **voices output** (`pith_audio_voices_pcm`): `n_samples` × i32 LE.
//! The SplitMix64 square-voice synthesis runs inside the cdylib on
//! purpose: it is a pure integer/exact-float recipe, so the language
//! SDKs never reimplement it and the synthetic vectors carry zero
//! platform sensitivity.

#![allow(unsafe_code)]

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::reference::{decode_stream, match_stream, signature_stream, voices_samples};
use crate::wav;
use crate::{Signature, build_index, match_signature, signature, signature_of_wav};

/// Status: success.
pub const PITH_OK: i32 = 0;
/// Status: a caller argument is invalid — a null pointer, a length
/// that is not a whole number of `i32` samples, or a zero/oversized
/// channel count.
pub const PITH_E_INVALID: i32 = -1;
/// Status: the core pipeline refused the input (malformed WAV, or PCM
/// the signature stage rejects).
pub const PITH_E_REJECTED: i32 = -2;

/// The incremental signature index behind the `pith_audio_index_*`
/// handles: signatures accumulate in insertion order, and the id a
/// match reports is that order — the position in the eventual
/// [`build_index`] input.
///
/// Opaque by construction: the field is private, the type is only ever
/// handled through the FFI (create with [`pith_audio_index_new`],
/// release with [`pith_audio_index_free`]).
pub struct IndexBuilder {
    sigs: Vec<Signature>,
}

/// Interprets `bytes` as little-endian `i32` samples. The caller has
/// already checked the `len % 4 == 0` geometry.
fn pcm_i32_le(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Serializes interleaved `i32` samples little-endian (the voices
/// output wire format).
fn i32_le_bytes(samples: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Validates the shared PCM argument shape: a whole number of 32-bit
/// samples and a channel count that fits `u16` and is nonzero.
fn pcm_args(len: usize, channels: u32) -> Result<u16, i32> {
    if len % 4 != 0 {
        return Err(PITH_E_INVALID);
    }
    let channels = u16::try_from(channels).map_err(|_| PITH_E_INVALID)?;
    if channels == 0 {
        return Err(PITH_E_INVALID);
    }
    Ok(channels)
}

/// Hands ownership of `stream` to the caller through the out-params:
/// the exact-length buffer's address and length, for a matching
/// [`pith_audio_free`].
fn hand_out(stream: Vec<u8>, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    let len = stream.len();
    // Hand the exact-length buffer to the caller; `pith_audio_free`
    // reconstructs the boxed slice from the same length.
    let ptr = Box::into_raw(stream.into_boxed_slice());
    unsafe {
        *out = ptr.cast::<u8>();
        *out_len = len;
    }
    PITH_OK
}

/// The safe core of [`pith_audio_decode_wav`]: decode, then serialize
/// into the decode stream. Decoding failures map to
/// [`PITH_E_REJECTED`].
fn decode_and_stream(bytes: &[u8]) -> Result<Vec<u8>, i32> {
    let decoded = wav::decode(bytes).map_err(|_| PITH_E_REJECTED)?;
    Ok(decode_stream(&decoded))
}

/// The safe core of [`pith_audio_signature_pcm`]: LE samples through
/// the signature stage, then the signature stream. Core refusals map
/// to [`PITH_E_REJECTED`].
fn signature_pcm_stream(samples: &[i32], channels: u16) -> Result<Vec<u8>, i32> {
    let sig = signature(samples, channels).map_err(|_| PITH_E_REJECTED)?;
    Ok(signature_stream(&sig))
}

/// Decodes a RIFF/WAVE stream into the decode stream the
/// `pcm_i32_le_*` digests are defined over.
///
/// `data` points at `len` bytes of the complete WAV file. On success
/// the function allocates a buffer, writes its address through `out`,
/// its length through `out_len`, and returns [`PITH_OK`]; the caller
/// owns the buffer and must release it with [`pith_audio_free`],
/// passing back the same pointer *and* length.
///
/// # Safety
///
/// `data` must point to `len` readable bytes; `out` to one writable
/// pointer; `out_len` to one writable `usize`. All must stay valid for
/// the duration of the call; the function retains nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_audio_decode_wav(
    data: *const u8,
    len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if data.is_null() || out.is_null() || out_len.is_null() {
        return PITH_E_INVALID;
    }
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    match decode_and_stream(bytes) {
        Ok(stream) => hand_out(stream, out, out_len),
        Err(status) => status,
    }
}

/// Decodes a RIFF/WAVE stream and extracts its signature (the
/// `signature_of_wav` facade: non-44 100 Hz is refused) into the
/// signature stream.
///
/// The buffer handed out through `out`/`out_len` follows the same
/// ownership contract as [`pith_audio_decode_wav`].
///
/// # Safety
///
/// `data` must point to `len` readable bytes; `out` to one writable
/// pointer; `out_len` to one writable `usize`. All must stay valid for
/// the duration of the call; the function retains nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_audio_signature_wav(
    data: *const u8,
    len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if data.is_null() || out.is_null() || out_len.is_null() {
        return PITH_E_INVALID;
    }
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    match signature_of_wav(bytes) {
        Ok(sig) => hand_out(signature_stream(&sig), out, out_len),
        Err(_) => PITH_E_REJECTED,
    }
}

/// Extracts a signature from interleaved little-endian `i32` PCM at
/// 44 100 Hz (the [`signature`] API) into the signature stream.
///
/// # Safety
///
/// `data` must point to `len` readable bytes; `out` to one writable
/// pointer; `out_len` to one writable `usize`. All must stay valid for
/// the duration of the call; the function retains nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_audio_signature_pcm(
    data: *const u8,
    len: usize,
    channels: u32,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if data.is_null() || out.is_null() || out_len.is_null() {
        return PITH_E_INVALID;
    }
    let channels = match pcm_args(len, channels) {
        Ok(channels) => channels,
        Err(status) => return status,
    };
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    match signature_pcm_stream(&pcm_i32_le(bytes), channels) {
        Ok(stream) => hand_out(stream, out, out_len),
        Err(status) => status,
    }
}

/// Synthesizes the SplitMix64 square-voice corpus
/// ([`voices_samples`]) into `n_samples` × i32 LE — the same bytes
/// the synthetic `reference.json` vectors are built from.
///
/// # Safety
///
/// `out` must point to one writable pointer and `out_len` to one
/// writable `usize`; both stay valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_audio_voices_pcm(
    seed: u64,
    n_samples: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if out.is_null() || out_len.is_null() {
        return PITH_E_INVALID;
    }
    hand_out(i32_le_bytes(&voices_samples(seed, n_samples)), out, out_len)
}

/// Creates an empty signature index. The returned handle is owned by
/// the caller and must be released with [`pith_audio_index_free`];
/// signatures are appended with [`pith_audio_index_add`], and the id a
/// match reports is the insertion order.
///
/// # Safety
///
/// Infallible for every caller; the `unsafe` qualifier follows the
/// suite's uniform export convention.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_audio_index_new() -> *mut IndexBuilder {
    Box::into_raw(Box::new(IndexBuilder { sigs: Vec::new() }))
}

/// Appends one signature — extracted from interleaved little-endian
/// `i32` PCM — to the index. A failed add leaves the handle alive and
/// unchanged (never freed here), so callers can correct and retry.
///
/// # Safety
///
/// `idx` must be a handle from [`pith_audio_index_new`] that has not
/// been freed; `data` must point to `len` readable bytes. Both stay
/// valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_audio_index_add(
    idx: *mut IndexBuilder,
    data: *const u8,
    len: usize,
    channels: u32,
) -> i32 {
    if idx.is_null() || data.is_null() {
        return PITH_E_INVALID;
    }
    let channels = match pcm_args(len, channels) {
        Ok(channels) => channels,
        Err(status) => return status,
    };
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    match signature(&pcm_i32_le(bytes), channels) {
        Ok(sig) => {
            let builder = unsafe { &mut *idx };
            builder.sigs.push(sig);
            PITH_OK
        }
        Err(_) => PITH_E_REJECTED,
    }
}

/// Matches the signature of interleaved little-endian `i32` PCM
/// against the index and writes the match stream: query facts, match
/// count, then one record per matched id (id, delta_t, votes,
/// total_votes — all big-endian). An empty index is legal and yields
/// `match_count == 0`. The buffer follows the same ownership contract
/// as [`pith_audio_decode_wav`].
///
/// # Safety
///
/// `idx` must be a handle from [`pith_audio_index_new`] that has not
/// been freed; `data` must point to `len` readable bytes; `out` to one
/// writable pointer; `out_len` to one writable `usize`. All stay valid
/// for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_audio_match(
    idx: *const IndexBuilder,
    data: *const u8,
    len: usize,
    channels: u32,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if idx.is_null() || data.is_null() || out.is_null() || out_len.is_null() {
        return PITH_E_INVALID;
    }
    let channels = match pcm_args(len, channels) {
        Ok(channels) => channels,
        Err(status) => return status,
    };
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    let query = match signature(&pcm_i32_le(bytes), channels) {
        Ok(sig) => sig,
        Err(_) => return PITH_E_REJECTED,
    };
    let builder = unsafe { &*idx };
    let index = build_index(&builder.sigs);
    let matches = match_signature(&query, &index);
    hand_out(match_stream(&query, &matches), out, out_len)
}

/// Releases an index handle from [`pith_audio_index_new`].
///
/// # Safety
///
/// `idx` must be a handle from [`pith_audio_index_new`] that has not
/// been released before. Null is accepted and ignored, so callers can
/// free unconditionally on the error path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_audio_index_free(idx: *mut IndexBuilder) {
    if idx.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(idx) });
}

/// Releases a buffer handed out by [`pith_audio_decode_wav`],
/// [`pith_audio_signature_wav`], [`pith_audio_signature_pcm`],
/// [`pith_audio_voices_pcm`] or [`pith_audio_match`].
///
/// # Safety
///
/// `ptr` must be a pointer handed out by one of those operations with
/// the `out_len` value that came back with it, and must not have been
/// released (or otherwise freed) before. Null is accepted and ignored,
/// so callers can free unconditionally on the error path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_audio_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let slice = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
    drop(unsafe { Box::from_raw(slice) });
}

#[cfg(test)]
mod tests {
    use super::{
        PITH_E_INVALID, PITH_E_REJECTED, PITH_OK, decode_and_stream, i32_le_bytes, pcm_args,
        pcm_i32_le, pith_audio_decode_wav, pith_audio_free, pith_audio_index_add,
        pith_audio_index_free, pith_audio_index_new, pith_audio_match, pith_audio_signature_pcm,
        pith_audio_signature_wav, pith_audio_voices_pcm, signature_pcm_stream,
    };
    use crate::reference::voices_samples;

    /// The committed decode/signature conformance fixture.
    fn fixture() -> alloc::vec::Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/tone.wav",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("fixture")
    }

    /// The i32-LE bytes of the mono signature corpus (seed and length
    /// exactly the `voices-signature-mono` vector's input).
    fn mono_corpus() -> alloc::vec::Vec<u8> {
        i32_le_bytes(&voices_samples(0xC0_FFEE, 66_150))
    }

    /// Reads a big-endian `u32` field of a stream.
    fn be32(bytes: &[u8], at: usize) -> u32 {
        u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }

    /// Reads a big-endian `u64` field of a stream.
    fn be64(bytes: &[u8], at: usize) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&bytes[at..at + 8]);
        u64::from_be_bytes(b)
    }

    /// A fixture decoded end-to-end through the raw FFI: status OK,
    /// the 16-byte decode-stream header, the exact body, and a clean
    /// free.
    #[test]
    fn ffi_decode_wav_reproduces_the_decode_stream() {
        let wav = fixture();
        let expected = decode_and_stream(&wav).expect("decode");
        let mut out: *mut u8 = core::ptr::null_mut();
        let mut out_len: usize = 0;
        let status =
            unsafe { pith_audio_decode_wav(wav.as_ptr(), wav.len(), &mut out, &mut out_len) };
        assert_eq!(status, PITH_OK);
        assert_eq!(out_len, expected.len());
        let handed = unsafe { core::slice::from_raw_parts(out, out_len) };
        assert_eq!(handed, expected.as_slice());
        // 44 100 Hz, mono, 16-bit, 44 100 frames, 44 100 samples.
        assert_eq!(
            &handed[..16],
            &[
                0x00, 0x00, 0xAC, 0x44, 0x00, 0x01, 0x00, 0x10, 0x00, 0x00, 0xAC, 0x44, 0x00, 0x00,
                0xAC, 0x44
            ]
        );
        unsafe { pith_audio_free(out, out_len) };
    }

    /// The fixture signature through the raw FFI carries the pinned
    /// facts (frames 20, 42 peaks, the reference.json fingerprint).
    #[test]
    fn ffi_signature_wav_reproduces_the_signature_stream() {
        let wav = fixture();
        let mut out: *mut u8 = core::ptr::null_mut();
        let mut out_len: usize = 0;
        let status =
            unsafe { pith_audio_signature_wav(wav.as_ptr(), wav.len(), &mut out, &mut out_len) };
        assert_eq!(status, PITH_OK);
        let handed = unsafe { core::slice::from_raw_parts(out, out_len) };
        assert_eq!(be32(handed, 0), 20);
        assert_eq!(be32(handed, 4), 42);
        assert_eq!(be64(handed, 8), 0x0400_0000_0200_1425);
        assert_eq!(out_len, 16 + 42 * 6);
        unsafe { pith_audio_free(out, out_len) };
    }

    /// `voices_pcm` emits exactly the synthesis bytes, and the corpus
    /// signature through `signature_pcm` carries the pinned vector
    /// facts (frames 31, 53 peaks, the reference.json fingerprint).
    #[test]
    fn ffi_voices_and_signature_pcm_reproduce_the_mono_vector() {
        let pcm = mono_corpus();
        let mut out: *mut u8 = core::ptr::null_mut();
        let mut out_len: usize = 0;
        let status = unsafe { pith_audio_voices_pcm(0xC0_FFEE, 66_150, &mut out, &mut out_len) };
        assert_eq!(status, PITH_OK);
        assert_eq!(out_len, pcm.len());
        let handed = unsafe { core::slice::from_raw_parts(out, out_len) };
        assert_eq!(handed, pcm.as_slice());
        unsafe { pith_audio_free(out, out_len) };

        let stream = signature_pcm_stream(&pcm_i32_le(&pcm), 1).expect("signature");
        assert_eq!(be32(&stream, 0), 31);
        assert_eq!(be32(&stream, 4), 53);
        assert_eq!(be64(&stream, 8), 0x0040_0000_0290_9467);
    }

    /// The offset-match scenario through the index triad: the query is
    /// the 2 s prefix followed by the 3 s indexed content, the index
    /// holds [content, other, filler] — member 0 wins with the modal
    /// offset at +43 frames.
    #[test]
    fn ffi_index_and_match_reproduce_the_offset_scenario() {
        let content = i32_le_bytes(&voices_samples(0xD1_CE, 132_300));
        let prefix = i32_le_bytes(&voices_samples(0xBE_EF, 88_200));
        let mut query = prefix;
        query.extend_from_slice(&content);
        let other = i32_le_bytes(&voices_samples(0xF0_0D, 132_300));
        let filler = i32_le_bytes(&voices_samples(0xE5_A5, 44_100));

        let idx = unsafe { pith_audio_index_new() };
        assert_eq!(
            unsafe { pith_audio_index_add(idx, content.as_ptr(), content.len(), 1) },
            PITH_OK
        );
        assert_eq!(
            unsafe { pith_audio_index_add(idx, other.as_ptr(), other.len(), 1) },
            PITH_OK
        );
        assert_eq!(
            unsafe { pith_audio_index_add(idx, filler.as_ptr(), filler.len(), 1) },
            PITH_OK
        );

        let mut out: *mut u8 = core::ptr::null_mut();
        let mut out_len: usize = 0;
        let status = unsafe {
            pith_audio_match(idx, query.as_ptr(), query.len(), 1, &mut out, &mut out_len)
        };
        assert_eq!(status, PITH_OK);
        let handed = unsafe { core::slice::from_raw_parts(out, out_len) };
        assert_eq!(be32(handed, 0), 106); // query_frames
        assert_eq!(be32(handed, 4), 156); // query_peak_count
        assert_eq!(be32(handed, 8), 3); // match_count
        // matches[0]: id 0, delta_t 43, votes 94, total_votes 471.
        assert_eq!(be32(handed, 12), 0);
        let mut delta = [0u8; 8];
        delta.copy_from_slice(&handed[16..24]);
        assert_eq!(i64::from_be_bytes(delta), 43);
        assert_eq!(be32(handed, 24), 94);
        assert_eq!(be32(handed, 28), 471);
        unsafe { pith_audio_free(out, out_len) };
        unsafe { pith_audio_index_free(idx) };
    }

    /// An empty index is legal: a valid signature matches nothing.
    #[test]
    fn ffi_match_on_an_empty_index_is_legal() {
        let pcm = i32_le_bytes(&voices_samples(7, 8192));
        let idx = unsafe { pith_audio_index_new() };
        let mut out: *mut u8 = core::ptr::null_mut();
        let mut out_len: usize = 0;
        let status =
            unsafe { pith_audio_match(idx, pcm.as_ptr(), pcm.len(), 1, &mut out, &mut out_len) };
        assert_eq!(status, PITH_OK);
        assert_eq!(out_len, 12);
        let handed = unsafe { core::slice::from_raw_parts(out, out_len) };
        assert_eq!(be32(handed, 8), 0);
        unsafe { pith_audio_free(out, out_len) };
        unsafe { pith_audio_index_free(idx) };
    }

    /// Null pointers are [`PITH_E_INVALID`]; malformed input is
    /// [`PITH_E_REJECTED`]; null frees are legal no-ops; a failed add
    /// leaves the handle alive.
    #[test]
    fn ffi_refusals() {
        let mut out: *mut u8 = core::ptr::null_mut();
        let mut out_len: usize = 0;
        let wav = fixture();

        // Null-pointer geometry.
        let null_data =
            unsafe { pith_audio_decode_wav(core::ptr::null(), 0, &mut out, &mut out_len) };
        assert_eq!(null_data, PITH_E_INVALID);
        let null_out = unsafe {
            pith_audio_decode_wav(wav.as_ptr(), wav.len(), core::ptr::null_mut(), &mut out_len)
        };
        assert_eq!(null_out, PITH_E_INVALID);
        let null_out_len = unsafe {
            pith_audio_decode_wav(wav.as_ptr(), wav.len(), &mut out, core::ptr::null_mut())
        };
        assert_eq!(null_out_len, PITH_E_INVALID);
        assert_eq!(
            unsafe { pith_audio_signature_wav(core::ptr::null(), 0, &mut out, &mut out_len) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_audio_signature_pcm(core::ptr::null(), 0, 1, &mut out, &mut out_len) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_audio_voices_pcm(1, 8, core::ptr::null_mut(), &mut out_len) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_audio_voices_pcm(1, 8, &mut out, core::ptr::null_mut()) },
            PITH_E_INVALID
        );

        // Core refusals surface as REJECTED.
        let garbage = [0u8; 16];
        assert_eq!(
            unsafe {
                pith_audio_decode_wav(garbage.as_ptr(), garbage.len(), &mut out, &mut out_len)
            },
            PITH_E_REJECTED
        );
        assert_eq!(
            unsafe {
                pith_audio_signature_wav(garbage.as_ptr(), garbage.len(), &mut out, &mut out_len)
            },
            PITH_E_REJECTED
        );

        // PCM geometry: partial samples and bad channel counts are
        // caller bugs (INVALID). A channel-count/sample-count mismatch
        // reaches the core and is REJECTED.
        let odd = [0u8; 6];
        assert_eq!(
            unsafe { pith_audio_signature_pcm(odd.as_ptr(), odd.len(), 1, &mut out, &mut out_len) },
            PITH_E_INVALID
        );
        let small = [0u8; 4]; // exactly one sample
        assert_eq!(
            unsafe {
                pith_audio_signature_pcm(small.as_ptr(), small.len(), 0, &mut out, &mut out_len)
            },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe {
                pith_audio_signature_pcm(
                    small.as_ptr(),
                    small.len(),
                    u32::from(u16::MAX) + 1,
                    &mut out,
                    &mut out_len,
                )
            },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe {
                pith_audio_signature_pcm(small.as_ptr(), small.len(), 2, &mut out, &mut out_len)
            },
            PITH_E_REJECTED
        );
        // Below one analysis window is legal: an empty signature.
        assert_eq!(
            unsafe {
                pith_audio_signature_pcm(small.as_ptr(), small.len(), 1, &mut out, &mut out_len)
            },
            PITH_OK
        );
        assert_eq!(out_len, 16); // header only, zero peaks
        let handed = unsafe { core::slice::from_raw_parts(out, out_len) };
        assert_eq!(be32(handed, 4), 0);
        unsafe { pith_audio_free(out, out_len) };

        // Index handles: null idx is INVALID, a rejected add leaves the
        // handle alive and usable.
        let idx = unsafe { pith_audio_index_new() };
        assert_eq!(
            unsafe { pith_audio_index_add(core::ptr::null_mut(), small.as_ptr(), 4, 1) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_audio_index_add(idx, core::ptr::null(), 4, 1) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_audio_index_add(idx, small.as_ptr(), 4, 0) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_audio_index_add(idx, small.as_ptr(), 6, 1) },
            PITH_E_INVALID
        );
        // One sample against 2 channels: a partial interleaved frame,
        // refused by the signature stage (the handle stays alive).
        assert_eq!(
            unsafe { pith_audio_index_add(idx, small.as_ptr(), 4, 2) },
            PITH_E_REJECTED
        );
        assert_eq!(
            unsafe {
                pith_audio_match(
                    core::ptr::null(),
                    small.as_ptr(),
                    4,
                    1,
                    &mut out,
                    &mut out_len,
                )
            },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe {
                pith_audio_match(
                    idx,
                    small.as_ptr(),
                    4,
                    1,
                    core::ptr::null_mut(),
                    &mut out_len,
                )
            },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_audio_match(idx, small.as_ptr(), 4, 2, &mut out, &mut out_len) },
            PITH_E_REJECTED
        );
        // Survived every failed add: a real corpus still lands.
        let pcm = mono_corpus();
        assert_eq!(
            unsafe { pith_audio_index_add(idx, pcm.as_ptr(), pcm.len(), 1) },
            PITH_OK
        );
        unsafe { pith_audio_index_free(idx) };

        // Null frees are accepted no-ops.
        unsafe { pith_audio_index_free(core::ptr::null_mut()) };
        unsafe { pith_audio_free(core::ptr::null_mut(), 0) };
    }

    /// The safe cores reject malformed input instead of panicking.
    #[test]
    fn safe_cores_reject_garbage() {
        assert_eq!(decode_and_stream(b"RIFF"), Err(PITH_E_REJECTED));
        // One sample against 2 channels: a partial interleaved frame.
        assert_eq!(signature_pcm_stream(&[0], 2), Err(PITH_E_REJECTED));
    }

    /// The argument-shape gate: sample-count geometry and channel
    /// bounds.
    #[test]
    fn pcm_args_gate() {
        assert_eq!(pcm_args(8, 2), Ok(2));
        assert_eq!(pcm_args(6, 1), Err(PITH_E_INVALID));
        assert_eq!(pcm_args(8, 0), Err(PITH_E_INVALID));
        assert_eq!(pcm_args(8, 65_536), Err(PITH_E_INVALID));
        assert_eq!(pcm_args(8, 65_535), Ok(65_535));
    }
}
