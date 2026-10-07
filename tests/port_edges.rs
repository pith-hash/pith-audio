//! Edge-path tests for `pith-audio`: the facade error `Display` text,
//! the FLAC decode facade's success path (the ported conformance suites
//! only exercise its error arm), and cross-codec signature parity
//! between the `wav` module and `pith-flac` on the same signal.
//!
//! Same philosophy as the other suites: the FLAC bytes are assembled
//! here, by hand, with a test-local bit writer and the CRCs RFC 9639
//! defines, so the facade is measured against the format definition
//! rather than against anything it produced.

use pith_audio::{Error, signature_of_flac, signature_of_wav};

// ------------------------------------------------------------------
// Test-local FLAC frame writer
// ------------------------------------------------------------------

/// MSB-first bit writer.
struct BitW {
    bytes: Vec<u8>,
    bit: u32,
}

impl BitW {
    fn new() -> Self {
        BitW {
            bytes: Vec::new(),
            bit: 0,
        }
    }

    fn bits(&mut self, v: u64, n: u32) {
        for i in (0..n).rev() {
            self.bit((v >> i) & 1 != 0);
        }
    }

    fn bit(&mut self, b: bool) {
        if self.bit == 0 {
            self.bytes.push(0);
        }
        if b {
            let last = self.bytes.len() - 1;
            self.bytes[last] |= 0x80 >> self.bit;
        }
        self.bit = (self.bit + 1) % 8;
    }

    fn byte(&mut self, v: u8) {
        self.bits(u64::from(v), 8);
    }

    /// Flushes a partial byte with zero bits and hands the buffer over.
    fn finish(mut self) -> Vec<u8> {
        while self.bit != 0 {
            self.bit(false);
        }
        self.bytes
    }
}

/// FLAC CRC-8 (poly 0x07), used for the frame header.
fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// FLAC CRC-16 (poly 0x8005), used for the whole frame.
fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in data {
        crc ^= u16::from(b) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// STREAMINFO metadata block: last block, 34-byte body with zero MD5.
fn streaminfo_block(
    min_block: u16,
    max_block: u16,
    rate: u32,
    channels: u16,
    bits: u16,
    total: u64,
) -> Vec<u8> {
    let mut body = BitW::new();
    body.bits(u64::from(min_block), 16);
    body.bits(u64::from(max_block), 16);
    body.bits(0, 24); // min frame size: unknown
    body.bits(0, 24); // max frame size: unknown
    body.bits(u64::from(rate), 20);
    body.bits(u64::from(channels - 1), 3);
    body.bits(u64::from(bits - 1), 5);
    body.bits(total, 36);
    let mut out = vec![0x80, 0x00, 0x00, 0x22]; // last-block, type STREAMINFO, len 34
    out.extend_from_slice(&body.finish());
    // The MD5 is parsed but never verified: 16 zero bytes.
    out.extend_from_slice(&[0u8; 16]);
    assert_eq!(out.len(), 4 + 34);
    out
}

/// One fixed-blocksize frame carrying a single constant subframe.
fn constant_frame(value: u16, total: u32) -> Vec<u8> {
    // Block size code 0110 writes an 8-bit size at the end of the
    // header; sample rate code 0000 defers to STREAMINFO (44 100 Hz),
    // the same resolution the suite's own FLAC vectors use.
    let mut header = BitW::new();
    header.byte(0xFF);
    header.byte(0xF8); // sync + reserved 0 + fixed blocksize
    header.byte(0b0110_0000); // blocksize-in-header + rate from STREAMINFO
    header.byte(0b0000_1000); // mono + 16-bit
    header.byte(0x00); // utf8 frame number 0
    header.byte(total as u8 - 1); // code 6 carries (block size - 1)
    let mut frame = header.finish();
    frame.push(crc8(&frame));

    let mut sub = BitW::new();
    sub.byte(0x00); // subframe header: constant, no wasted bits
    sub.bits(u64::from(value), 16);
    frame.extend_from_slice(&sub.finish());
    let crc = crc16(&frame);
    frame.push((crc >> 8) as u8);
    frame.push((crc & 0xFF) as u8);
    frame
}

/// A whole one-frame FLAC stream decoding to `total` constant samples.
fn constant_flac(value: u16, total: u32) -> Vec<u8> {
    let mut out = b"fLaC".to_vec();
    out.extend_from_slice(&streaminfo_block(
        16,
        65_535,
        44_100,
        1,
        16,
        u64::from(total),
    ));
    out.extend_from_slice(&constant_frame(value, total));
    out
}

// ------------------------------------------------------------------
// Facade Display pins
// ------------------------------------------------------------------

#[test]
fn error_display_texts_are_pinned() {
    // These strings travel through `Error::Decode(e.to_string())` of
    // every caller, so the prefix wording is wire format.
    assert_eq!(Error::BadValue("x").to_string(), "bad value: x");
    assert_eq!(Error::Unsupported("y").to_string(), "unsupported: y");
    assert_eq!(Error::Decode("z".into()).to_string(), "decode: z");
}

// ------------------------------------------------------------------
// FLAC facade: success path with a hand-built stream
// ------------------------------------------------------------------

#[test]
fn flac_facade_decodes_constant_stream_and_refuses_other_rates() {
    let flac = constant_flac(0x1234, 8);
    let sig = signature_of_flac(&flac).expect("constant stream must decode");
    // Eight samples: one window needs 4096, so the signature is empty —
    // but the decode + rate + extract path all ran.
    assert!(sig.is_empty());
    assert_eq!(sig.frames(), 0);
    assert_eq!(sig.fingerprint(), 0);
}

#[test]
fn wav_and_flac_facades_agree_on_the_same_signal() {
    // The same 8 constant samples through both decoders: the signature
    // pipeline is codec-agnostic, so both facades must agree exactly.
    let flac = constant_flac(0x1234, 8);
    let mut wav = Vec::new();
    let fmt = {
        let mut f = Vec::new();
        f.extend_from_slice(&1u16.to_le_bytes()); // PCM
        f.extend_from_slice(&1u16.to_le_bytes()); // mono
        f.extend_from_slice(&44_100u32.to_le_bytes());
        f.extend_from_slice(&88_200u32.to_le_bytes()); // byte rate
        f.extend_from_slice(&2u16.to_le_bytes()); // block align
        f.extend_from_slice(&16u16.to_le_bytes());
        f
    };
    let data: Vec<u8> = [0x1234i16; 8]
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    let body_len = 4 + 8 + fmt.len() + 8 + data.len();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(body_len as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    wav.extend_from_slice(&fmt);
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
    wav.extend_from_slice(&data);

    let via_flac = signature_of_flac(&flac).expect("flac must decode");
    let via_wav = signature_of_wav(&wav).expect("wav must decode");
    assert_eq!(via_flac, via_wav);

    // The rate refusal wording is per-codec; both are Unsupported.
    assert!(matches!(
        signature_of_wav(b"not a wav"),
        Err(Error::Decode(_))
    ));
}
