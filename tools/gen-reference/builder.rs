//! Deterministic RIFF/WAVE construction for the reference vectors.
//!
//! The suite has no WAV encoder, so the vectors' inputs are assembled
//! here byte-by-byte exactly as the RIFF specification describes — the
//! same construction the `wav` module's integration tests use — and
//! the decoder is then measured on those bytes. Every function is pure
//! and every byte stream is stable across runs and platforms: no time,
//! no platform-dependent bytes. The synthetic signal synthesis
//! (SplitMix64 square voices) lives in [`pith_audio::reference`], the
//! crate module the generator and the FFI share.

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
}
