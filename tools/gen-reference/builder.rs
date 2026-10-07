//! Deterministic RIFF/WAVE construction and signal synthesis for the
//! reference vectors.
//!
//! The suite has no WAV encoder, so the vectors' synthetic inputs are
//! assembled here byte-by-byte exactly as the RIFF specification
//! describes — the same construction the `wav` module's integration
//! tests use — and the decoder is then measured on those bytes. Every
//! function is pure and every byte stream is stable across runs and
//! platforms: no RNG beyond the fixed SplitMix64 stream, no time, no
//! transcendentals (the carriers are square waves, whose ±1 values are
//! exact), so the committed `reference.json` stays byte-identical
//! everywhere.

use std::vec::Vec;

/// One complete RIFF chunk: id, u32-LE size, payload, and the pad byte
/// that RIFF requires after every odd-sized payload.
pub fn chunk(id: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + payload.len() + 1);
    v.extend_from_slice(id);
    v.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    v.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        v.push(0);
    }
    v
}

/// A 16-byte `fmt ` chunk (the WAVEFORMAT prefix the decoder reads).
pub fn fmt_chunk(tag: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
    let block_align = channels * (bits / 8);
    let byte_rate = rate * u32::from(block_align);
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&tag.to_le_bytes());
    v.extend_from_slice(&channels.to_le_bytes());
    v.extend_from_slice(&rate.to_le_bytes());
    v.extend_from_slice(&byte_rate.to_le_bytes());
    v.extend_from_slice(&block_align.to_le_bytes());
    v.extend_from_slice(&bits.to_le_bytes());
    v
}

/// A whole file: RIFF header (size filled from the real length), then
/// the given chunks in order.
pub fn wav_file(chunks: &[Vec<u8>]) -> Vec<u8> {
    let body_len: usize = 4 + chunks.iter().map(Vec::len).sum::<usize>();
    let mut v = Vec::with_capacity(8 + body_len);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(body_len as u32).to_le_bytes());
    v.extend_from_slice(b"WAVE");
    for c in chunks {
        v.extend_from_slice(c);
    }
    v
}

/// A minimal well-formed file: one `fmt ` chunk then one `data` chunk.
pub fn pcm_wav(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
    wav_file(&[
        chunk(b"fmt ", &fmt_chunk(tag, channels, rate, bits)),
        chunk(b"data", data),
    ])
}

/// Writes interleaved `i32` samples as 16-bit PCM data (top 16 bits),
/// the same scale step the `wav16` test helper uses.
pub fn pcm16_data(samples: &[i32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        v.extend_from_slice(&((*s >> 16) as i16).to_le_bytes());
    }
    v
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_pads_odd_payloads() {
        let c = chunk(b"LIST", b"INFOX");
        assert_eq!(c.len(), 8 + 5 + 1);
        assert_eq!(&c[8..13], b"INFOX");
        assert_eq!(c[13], 0);
        assert_eq!(chunk(b"data", &[1, 2]).len(), 10);
    }

    #[test]
    fn fmt_chunk_encodes_derived_fields() {
        let f = fmt_chunk(1, 2, 44_100, 16);
        assert_eq!(f.len(), 16);
        assert_eq!(u16::from_le_bytes([f[0], f[1]]), 1);
        assert_eq!(u16::from_le_bytes([f[2], f[3]]), 2);
        assert_eq!(u32::from_le_bytes([f[4], f[5], f[6], f[7]]), 44_100);
        // byte_rate = rate * block_align = 44100 * 4.
        assert_eq!(u32::from_le_bytes([f[8], f[9], f[10], f[11]]), 176_400);
        assert_eq!(u16::from_le_bytes([f[12], f[13]]), 4);
        assert_eq!(u16::from_le_bytes([f[14], f[15]]), 16);
    }

    #[test]
    fn wav_file_fills_the_riff_size() {
        let f = pcm_wav(1, 1, 8_000, 8, &[0x80]);
        // 12 header + 8+16 fmt + 8+1 data + 1 RIFF pad byte.
        assert_eq!(f.len(), 46);
        let size = u32::from_le_bytes([f[4], f[5], f[6], f[7]]);
        assert_eq!(size as usize, f.len() - 8);
    }

    #[test]
    fn pcm16_data_takes_the_top_bits() {
        assert_eq!(
            pcm16_data(&[0x1234_5678, -0x0002_0000]),
            [0x34, 0x12, 0xfe, 0xff]
        );
    }

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
}
