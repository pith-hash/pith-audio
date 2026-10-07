<p align="center">
  <img src="https://pith-audio.n24q02m.com/logo.svg" alt="pith-audio" width="120">
</p>

<h1 align="center">pith-audio</h1>

<p align="center">
  <strong>Audio fingerprinting: RIFF/WAVE and FLAC decode, Shazam-style spectral-peak signatures and delta-t histogram matching</strong>
</p>

<p align="center">
  <a href="https://github.com/pith-hash/pith-audio/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/pith-hash/pith-audio/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/pith-hash/pith-audio/actions/workflows/cd.yml"><img alt="CD" src="https://github.com/pith-hash/pith-audio/actions/workflows/cd.yml/badge.svg"></a>
  <a href="https://github.com/pith-hash/pith-audio/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/pith-hash/pith-audio?display_name=tag&sort=semver"></a>
  <a href="https://github.com/n24q02m/better-semantic-release"><img alt="semantic-release" src="https://img.shields.io/badge/semantic--release-e10079?logo=semantic-release&logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#the-pith-suite-contract">Suite contract</a>
</p>

<!-- BEGIN: AUTO-GENERATED-CROSS-PROMO -->
<!-- END: AUTO-GENERATED-CROSS-PROMO -->

## The pith suite contract

pith-audio is part of the **pith** suite (pith-hash). Every suite repository
follows the same rules; CI enforces them mechanically:

- **Naming**: a library is always `pith-<domain>` (`pith-image`, `pith-audio`,
  `pith-zip`, ...). The curator/repository of repositories is the bare
  `pith-hash`. Never invent a second naming scheme inside the suite.
- **Version pinning**: cross-library dependencies pin `~0.1` (e.g.
  `pith-image = { version = "~0.1", path = "../pith-image" }`). The whole suite
  moves together inside 0.1.x; breaking changes require a suite-wide version
  bump, never a silent minor drift.
- **Zero third-party dependencies**: every crate depends only on other
  `pith-*` crates plus `std`. `scripts/check-zero-deps.py` (run in CI) fails
  the build on any other crate, for normal, build and dev dependencies alike.
- **No unsafe**: every crate root carries `#![forbid(unsafe_code)]`.
- **Hex-exact vectors**: `reference.json` at the repo root is the
  cross-language source of truth. The `gen-reference` binary regenerates it;
  CI verifies the committed copy is current (`gen-reference verify`), and CD
  ships the regenerated file with every SDK artifact. Python, Node and Go SDKs
  MUST test against the same bytes.

## Repository layout

```
src/wav.rs         RIFF/WAVE decoder (the former standalone modhash-wav crate)
src/peaks.rs       spectral-peak extraction (FFT, slot pooling, landmarks)
src/table.rs       inverted peak index and delta-t histogram matching
tools/gen-reference  the vector generator binary (bin name: gen-reference)
tests/fixtures/tone.wav  the committed conformance fixture (byte-exact)
reference.json     hex-exact cross-SDK test vectors
```

## Install

Rust (the core library):

```bash
cargo add pith-audio
```

Python / Node / Go SDKs are published from the same cdylib on every release;
see the release assets or the package registries for the matching version.

## Quick start

Rust:

```rust
use pith_audio::{build_index, match_signature, signature_of_wav};

// Extract a landmark signature from RIFF/WAVE bytes
// (or `signature_of_flac` for FLAC; both refuse other rates).
let query = signature_of_wav(wav_bytes).unwrap();

// Index a corpus and match: the modal delta-t is the offset in
// 2048-sample frames (about 46 ms each) of the best-matching member.
let corpus: Vec<pith_audio::Signature> = other_wav_bytes
    .iter()
    .map(|b| signature_of_wav(b).unwrap())
    .collect();
let index = build_index(&corpus);
let matches = match_signature(&query, &index);
if let Some(m) = matches.first() {
    println!("member {} at delta_t {} frames ({} votes)", m.id, m.delta_t, m.votes);
}

// Raw RIFF/WAVE decoding, canonical interleaved i32 PCM:
let wav = pith_audio::wav::decode(riff_bytes).unwrap();
assert_eq!(wav.frames(), wav.samples().len() / usize::from(wav.channels()));
```

The crate is `no_std` apart from the `alloc` containers its API returns and
deliberately refuses what it does not implement: WAV format tags other than
PCM and IEEE float surface as errors, never a guess, and both facades refuse
anything but 44 100 Hz before a single FFT. The committed
[`reference.json`](reference.json) carries the hex-exact decode, peak-extraction
and match vectors (the `tone.wav` conformance fixture plus synthetic
constant/verbatim PCM, sign-extension, float-clamp, chunk-walk and
delta-t-offset corpora); regenerate with `cargo run --bin gen-reference -- gen`
and verify with `cargo run --bin gen-reference -- verify`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © pith-hash
