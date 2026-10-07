//! Audio fingerprint facade: spectral peaks and the peak-to-id
//! reverse map.
//!
//! Part of the `pith` zero-dependency hashing suite: every crate depends
//! only on other `pith-*` crates plus `std`, so the whole suite resolves
//! without a single registry package. The decoders this facade fronts are
//! the crate-local `wav` module (RIFF/WAVE) and `pith-flac` (FLAC);
//! the FFT is `pith-math`'s.
//!
//! The pipeline: mono-mixed 44 100 Hz `i32` PCM → Hann-windowed 4096 real FFT
//! at a 2048 hop → 2049 bins max-pooled into 257 slots (bin stride 8)
//! → in-band (≈32–5000 Hz, bins 3–464 → slots 0–58) local maxima that
//! must also win their 9-frame temporal neighbourhood → the signature:
//! a sorted `(t, f)` set. Matching inverts the map — `f → (t, id)` —
//! and histograms `Δt = t_query − t_stored`; the modal offset within
//! ±1 frame is the answer.
//!
//! The crate is `no_std` apart from the `alloc` containers its API
//! returns and the error strings it carries; the `std` feature (on by
//! default) links `std` so the `cdylib` the language SDKs bind through
//! carries a panic handler.

#![cfg_attr(not(feature = "std"), no_std)]
// `unsafe` is denied everywhere except `ffi`, the C ABI surface the
// language SDKs bind through: raw pointers exist only at that boundary,
// and every exported function is a documented `unsafe extern "C"` fn.
#![deny(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

use alloc::string::{String, ToString};
use core::fmt;

mod peaks;
mod table;

pub mod ffi;
pub mod reference;
pub mod wav;

pub use peaks::{
    BIN_MAX, BIN_MIN, BIN_STRIDE, HOP, Peak, SAMPLE_RATE, SLOT_COUNT, SLOT_MAX, SLOT_MIN,
    Signature, TIME_RADIUS, WINDOW, signature,
};
pub use table::{DELTA_TOL, Index, Match, build_index, match_signature};

/// Crate-local result alias; `E` defaults to this crate's [`Error`].
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Every failure this crate can report. Deliberately narrower than
/// `pith_digest::Error` — the decoders' typed errors are flattened to
/// their `Display` text via [`Error::Decode`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A structurally valid but semantically impossible call: zero
    /// channels, a partial trailing frame.
    BadValue(&'static str),
    /// Input of a kind this pipeline is not tuned for — anything but
    /// 44 100 Hz through the decode facades.
    Unsupported(&'static str),
    /// The WAV/FLAC facade's decoder rejected the bytes. Carries the
    /// decoder's `Display` text; the underlying typed error stays in
    /// the decoder crate.
    Decode(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadValue(what) => {
                f.write_str("bad value: ")?;
                f.write_str(what)
            }
            Error::Unsupported(what) => {
                f.write_str("unsupported: ")?;
                f.write_str(what)
            }
            Error::Decode(what) => {
                f.write_str("decode: ")?;
                f.write_str(what)
            }
        }
    }
}

/// Decodes `bytes` as RIFF/WAVE via the [`wav`] module and extracts the
/// signature. Anything but 44 100 Hz is refused before a single FFT:
/// the constants are tuned to one rate and silently resampling is not
/// this crate's job.
///
/// # Errors
///
/// [`Error::Decode`] for a file [`wav::decode`] rejects, and
/// [`Error::BadValue`] / [`Error::Unsupported`] exactly as
/// [`signature`] documents plus the sample-rate refusal.
pub fn signature_of_wav(bytes: &[u8]) -> Result<Signature> {
    let decoded = wav::decode(bytes).map_err(|e| Error::Decode(e.to_string()))?;
    if decoded.sample_rate() != SAMPLE_RATE {
        return Err(Error::Unsupported("wav sample rate ≠ 44100 Hz"));
    }
    signature(decoded.samples(), decoded.channels())
}

/// Decodes `bytes` as FLAC via [`pith_flac`] with its default
/// [`pith_flac::Limits`] and extracts the signature. Same 44 100 Hz
/// contract as [`signature_of_wav`].
///
/// # Errors
///
/// [`Error::Decode`] for a stream `pith_flac` rejects, and
/// [`Error::BadValue`] / [`Error::Unsupported`] as [`signature`]
/// documents plus the sample-rate refusal.
pub fn signature_of_flac(bytes: &[u8]) -> Result<Signature> {
    let flac = pith_flac::decode(bytes, &pith_flac::Limits::default())
        .map_err(|e| Error::Decode(e.to_string()))?;
    if flac.sample_rate() != SAMPLE_RATE {
        return Err(Error::Unsupported("flac sample rate ≠ 44100 Hz"));
    }
    signature(flac.samples(), flac.channels())
}
