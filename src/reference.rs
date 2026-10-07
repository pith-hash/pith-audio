//! The canonical synthesis and serialization behind `reference.json`,
//! re-expressed as library code so the vector generator
//! (`tools/gen-reference`) and the C ABI surface ([`crate::ffi`]) share
//! one implementation.
//!
//! The synthesis half (SplitMix64 square voices) moved verbatim out of
//! `tools/gen-reference/builder.rs`: no RNG beyond the fixed SplitMix64
//! stream, no time, no transcendentals (the carriers are square waves,
//! whose ±1 values are exact), so every synthetic corpus is
//! byte-stable across runs and platforms. Because the synthesis is a
//! pure integer/exact-float recipe, it also runs *inside* the cdylib
//! ([`voices_samples`] through [`crate::ffi::pith_audio_voices_pcm`]):
//! the language SDKs never reimplement it, so the vectors carry zero
//! platform sensitivity.
//!
//! The serialization half defines the three wire formats the FFI
//! hands to the language SDKs — the byte contracts the
//! `reference.json` digests are computed over and every SDK test
//! replays:
//!
//! **Decode stream** ([`decode_stream`], the bytes
//! `pcm_i32_le_sha256` covers): `[0..4] sample_rate` as `u32`
//! big-endian, `[4..6] channels` as `u16` big-endian, `[6..8]
//! bits_per_sample` as `u16` big-endian, `[8..12] frames` as `u32`
//! big-endian, `[12..16] decoded_samples` as `u32` big-endian, then
//! `[16..]` the decoded samples as `i32` little-endian — so
//! `pcm_i32_le_sha256 == sha256(stream[16..])` and
//! `pcm_i32_le_fnv1a64 == fnv1a64(stream[16..])`.
//!
//! **Signature stream** ([`signature_stream`]): `[0..4] frames` as
//! `u32` big-endian, `[4..8] peak_count` as `u32` big-endian,
//! `[8..16] fingerprint` as `u64` big-endian, then one record per
//! peak: `t` as `u32` big-endian, `f` as `u16` big-endian. Note the
//! digest asymmetry: `peaks_sha256` in `reference.json` hashes the
//! *little*-endian re-serialization of the same records (`t` as `u32`
//! LE, `f` as `u16` LE), the exact layout `tools/gen-reference`
//! hashes — SDK tests re-serialize LE to recompute it.
//!
//! **Match stream** ([`match_stream`]): `[0..4] query_frames` as
//! `u32` big-endian, `[4..8] query_peak_count` as `u32`
//! big-endian, `[8..12] match_count` as `u32` big-endian, then one
//! record per match: `id` as `u32` big-endian, `delta_t` as `i64`
//! big-endian, `votes` as `u32` big-endian, `total_votes` as `u32`
//! big-endian.

use alloc::vec;
use alloc::vec::Vec;

use crate::peaks::Signature;
use crate::table::Match;
use crate::wav;

/// SplitMix64 — the exact stream the suite's tests and the committed
/// `tone.wav` fixture are built on (seed chosen per corpus).
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator at `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Half-periods (in samples) of the ten square-wave carriers, derived
/// from the slot-centre bins 43, 83, …, 403 as `round(4096 / bin)`:
/// `[95, 49, 33, 25, 20, 17, 14, 13, 11, 10]`. The same recipe the
/// committed `tests/fixtures/tone.wav` fixture was generated with.
pub const VOICE_HALF_PERIODS: [usize; 10] = [95, 49, 33, 25, 20, 17, 14, 13, 11, 10];

/// Burst-envelope length in analysis frames.
const BURST_FRAMES: usize = 10;

/// Synthesizes mono square-voice content at 44 100 Hz: ten carriers
/// with SplitMix64-scheduled bursts (attack, then ×0.82 decay), mixed
/// and scaled exactly like the suite's test recipes — clamp to
/// ±0.95, scale by `i32::MAX`, truncate toward zero. Deterministic on
/// every platform: only exact IEEE multiply/divide/compare, no
/// transcendentals.
#[must_use]
pub fn voices_samples(seed: u64, n_samples: usize) -> Vec<i32> {
    let hop = 2048usize;
    let n_frames = n_samples / hop;
    let mut out = vec![0i32; n_samples];
    if n_frames == 0 {
        // Shorter than one analysis frame: the pipeline sees no frames
        // either, so silence is the faithful synthesis.
        return out;
    }
    let mut rng = SplitMix64::new(seed);
    let mut env = vec![vec![0.0f64; n_frames]; VOICE_HALF_PERIODS.len()];
    for row in &mut env {
        for fr in 0..n_frames {
            let r = rng.next_u64();
            if r % 23 == 0 {
                let a = 0.5 + 0.45 * ((r >> 32) as f64 / u64::MAX as f64);
                for k in 0..BURST_FRAMES.min(n_frames - fr) {
                    row[fr + k] = a * 0.82f64.powi(k as i32);
                }
            }
        }
    }
    for (i, s) in out.iter_mut().enumerate() {
        let frame = (i / hop).min(n_frames - 1);
        let mut x = 0.0;
        for (v, hp) in VOICE_HALF_PERIODS.iter().enumerate() {
            let a = env[v][frame];
            if a > 0.0 {
                let sq = if i % (2 * hp) < *hp { 1.0 } else { -1.0 };
                x += a * sq;
            }
        }
        let clamped = x.clamp(-0.95, 0.95);
        *s = (clamped * 2147483647.0) as i32;
    }
    out
}

/// Synthesizes interleaved stereo content: left from `seed_l`, right
/// from `seed_r`, `n_samples` per channel.
#[must_use]
pub fn voices_samples_stereo(seed_l: u64, seed_r: u64, n_samples: usize) -> Vec<i32> {
    let left = voices_samples(seed_l, n_samples);
    let right = voices_samples(seed_r, n_samples);
    let mut out = Vec::with_capacity(n_samples * 2);
    for i in 0..n_samples {
        out.push(left[i]);
        out.push(right[i]);
    }
    out
}

/// The explicit mono average of a stereo interleaved stream — the
/// reference the mixdown vector compares the pipeline's own mix
/// against.
#[must_use]
pub fn mono_average(interleaved: &[i32], channels: u16) -> Vec<i32> {
    let ch = usize::from(channels);
    (0..interleaved.len() / ch)
        .map(|i| {
            let mut acc = 0i64;
            for c in 0..ch {
                acc += i64::from(interleaved[i * ch + c]);
            }
            (acc / i64::from(channels)) as i32
        })
        .collect()
}

/// Serializes one decoded WAV into the decode stream the
/// `pcm_i32_le_sha256` / `pcm_i32_le_fnv1a64` digests cover and the
/// FFI [`crate::ffi::pith_audio_decode_wav`] hands out (see the
/// module docs for the exact layout).
#[must_use]
pub fn decode_stream(w: &wav::Wav) -> Vec<u8> {
    let samples = w.samples();
    // A RIFF payload is u32-sized, so both counts fit `u32` for any
    // file the container can represent.
    let mut out = Vec::with_capacity(16 + samples.len() * 4);
    out.extend_from_slice(&w.sample_rate().to_be_bytes());
    out.extend_from_slice(&w.channels().to_be_bytes());
    out.extend_from_slice(&w.bits_per_sample().to_be_bytes());
    out.extend_from_slice(&(w.frames() as u32).to_be_bytes());
    out.extend_from_slice(&(samples.len() as u32).to_be_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Serializes one extracted signature into the signature stream the
/// FFI [`crate::ffi::pith_audio_signature_wav`] /
/// [`crate::ffi::pith_audio_signature_pcm`] hand out (see the module
/// docs for the exact layout and the LE digest asymmetry).
#[must_use]
pub fn signature_stream(s: &Signature) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + s.len() * 6);
    out.extend_from_slice(&s.frames().to_be_bytes());
    out.extend_from_slice(&(s.len() as u32).to_be_bytes());
    out.extend_from_slice(&s.fingerprint().to_be_bytes());
    for peak in s.peaks() {
        out.extend_from_slice(&peak.t.to_be_bytes());
        out.extend_from_slice(&peak.f.to_be_bytes());
    }
    out
}

/// Serializes one match run into the match stream the FFI
/// [`crate::ffi::pith_audio_match`] hands out (see the module docs
/// for the exact layout).
#[must_use]
pub fn match_stream(query: &Signature, matches: &[Match]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + matches.len() * 20);
    out.extend_from_slice(&query.frames().to_be_bytes());
    out.extend_from_slice(&(query.len() as u32).to_be_bytes());
    out.extend_from_slice(&(matches.len() as u32).to_be_bytes());
    for m in matches {
        out.extend_from_slice(&m.id.to_be_bytes());
        out.extend_from_slice(&m.delta_t.to_be_bytes());
        out.extend_from_slice(&m.votes.to_be_bytes());
        out.extend_from_slice(&m.total_votes.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        decode_stream, match_stream, mono_average, signature_stream, voices_samples,
        voices_samples_stereo,
    };

    /// The synthesis half keeps its moved-in properties: deterministic
    /// per seed, bounded, silent below one analysis frame.
    #[test]
    fn voices_are_deterministic_and_bounded() {
        // Seed 7 hits the 1-in-23 burst gate on draw 5, so 4 frames of
        // audio are guaranteed non-silent.
        let a = voices_samples(7, 8192);
        let b = voices_samples(7, 8192);
        assert_eq!(a, b);
        assert!(a.iter().any(|&s| s != 0), "burst schedule produced audio");
        // Shorter than one hop: silence, never a panic.
        assert!(voices_samples(7, 128).iter().all(|&s| s == 0));
    }

    /// The stereo interleave and the explicit average round-trip.
    #[test]
    fn stereo_interleaves_and_averages_back() {
        let st = voices_samples_stereo(0x11, 0x22, 8192);
        assert_eq!(st.len(), 16_384);
        let mono = mono_average(&st, 2);
        assert_eq!(mono.len(), 8192);
        let left = voices_samples(0x11, 8192);
        let right = voices_samples(0x22, 8192);
        for (i, m) in mono.iter().enumerate() {
            assert_eq!(
                i64::from(*m),
                (i64::from(left[i]) + i64::from(right[i])) / 2
            );
        }
    }

    /// The three stream builders emit the documented big-endian
    /// headers with sample-exact bodies and match records.
    #[test]
    fn streams_follow_the_documented_layouts() {
        // A committed vector input: 8 000 Hz mono 8-bit, 4 samples.
        let pcm8 = hex(b"524946462800000057415645666d74201000000001000100401f0000401f00000100080064617461040000000080ff01");
        let decoded = crate::wav::decode(&pcm8).expect("vector input decodes");
        let stream = decode_stream(&decoded);
        assert_eq!(stream.len(), 16 + 4 * 4);
        assert_eq!(&stream[..4], &8_000u32.to_be_bytes());
        assert_eq!(&stream[4..6], &1u16.to_be_bytes());
        assert_eq!(&stream[6..8], &8u16.to_be_bytes());
        assert_eq!(&stream[8..12], &4u32.to_be_bytes());
        assert_eq!(&stream[12..16], &4u32.to_be_bytes());
        assert_eq!(&stream[16..20], &(-128i32).to_le_bytes());
        assert_eq!(&stream[20..24], &0i32.to_le_bytes());

        // A voiced corpus gives a non-empty signature, so the match
        // stream carries a real record: match the corpus against
        // itself (delta 0, all votes).
        let pcm = voices_samples(7, 8192);
        let sig = crate::signature(&pcm, 1).expect("pcm accepted");
        assert!(!sig.is_empty());
        let sstream = signature_stream(&sig);
        assert_eq!(sstream.len(), 16 + sig.len() * 6);
        assert_eq!(&sstream[..4], &sig.frames().to_be_bytes());
        assert_eq!(&sstream[4..8], &(sig.len() as u32).to_be_bytes());
        assert_eq!(&sstream[8..16], &sig.fingerprint().to_be_bytes());
        let (t, f) = {
            let p = &sig.peaks()[0];
            (p.t, p.f)
        };
        assert_eq!(&sstream[16..20], &t.to_be_bytes());
        assert_eq!(&sstream[20..22], &f.to_be_bytes());

        let index = crate::build_index(&[crate::signature(&pcm, 1).expect("pcm accepted")]);
        let matches = crate::match_signature(&sig, &index);
        assert_eq!(matches.len(), 1);
        let mstream = match_stream(&sig, &matches);
        assert_eq!(mstream.len(), 12 + 20); // 12-byte header + one 20-byte record
        assert_eq!(&mstream[..4], &sig.frames().to_be_bytes());
        assert_eq!(&mstream[4..8], &(sig.len() as u32).to_be_bytes());
        assert_eq!(&mstream[8..12], &1u32.to_be_bytes());
        assert_eq!(&mstream[12..16], &0u32.to_be_bytes()); // id: self-match
        assert_eq!(&mstream[16..24], &0i64.to_be_bytes()); // modal delta 0
    }

    /// Decodes an even-length lowercase hex string (test inputs only).
    fn hex(s: &[u8]) -> alloc::vec::Vec<u8> {
        let s = core::str::from_utf8(s).expect("hex text");
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("hex byte"))
            .collect()
    }
}
