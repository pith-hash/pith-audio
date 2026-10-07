//! End-to-end FFI conformance through the rlib, from a *consumer*
//! crate's point of view — the same calls the language SDKs make
//! through ctypes/koffi/cgo. Complements the in-crate `ffi` unit
//! tests (raw-pointer edge cases, safe cores) with the
//! integration-level round trip over the committed fixture and the
//! synthetic offset-match corpus.

use pith_audio::reference::voices_samples;

/// i32-LE serializes interleaved samples (the PCM wire format).
fn pcm_le(samples: &[i32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        v.extend_from_slice(&s.to_le_bytes());
    }
    v
}

fn be32(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// A commit fixture decoded through the raw FFI: OK, the documented
/// 16-byte header, and a clean free.
#[test]
fn decode_and_signature_of_the_fixture() {
    let wav = std::fs::read(format!(
        "{}/tests/fixtures/tone.wav",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("fixture");

    let mut out: *mut u8 = core::ptr::null_mut();
    let mut out_len: usize = 0;
    let status = unsafe {
        pith_audio::ffi::pith_audio_decode_wav(wav.as_ptr(), wav.len(), &mut out, &mut out_len)
    };
    assert_eq!(status, pith_audio::ffi::PITH_OK);
    let stream = unsafe { core::slice::from_raw_parts(out, out_len) };
    assert_eq!(&stream[..4], &44_100u32.to_be_bytes());
    assert_eq!(&stream[4..6], &1u16.to_be_bytes());
    assert_eq!(&stream[6..8], &16u16.to_be_bytes());
    assert_eq!(&stream[8..12], &44_100u32.to_be_bytes());
    assert_eq!(stream.len(), 16 + 44_100 * 4);
    unsafe { pith_audio::ffi::pith_audio_free(out, out_len) };

    let mut out: *mut u8 = core::ptr::null_mut();
    let mut out_len: usize = 0;
    let status = unsafe {
        pith_audio::ffi::pith_audio_signature_wav(wav.as_ptr(), wav.len(), &mut out, &mut out_len)
    };
    assert_eq!(status, pith_audio::ffi::PITH_OK);
    let stream = unsafe { core::slice::from_raw_parts(out, out_len) };
    assert_eq!(be32(stream, 0), 20); // frames
    assert_eq!(be32(stream, 4), 42); // peak_count
    assert_eq!(out_len, 16 + 42 * 6);
    unsafe { pith_audio::ffi::pith_audio_free(out, out_len) };
}

/// The offset-match scenario through the index triad, PCM synthesized
/// inside the cdylib via `voices_pcm`.
#[test]
fn voices_index_and_match_end_to_end() {
    let mut out: *mut u8 = core::ptr::null_mut();
    let mut out_len: usize = 0;

    let mut query = Vec::new();
    for (seed, n) in [(0xBE_EF, 88_200), (0xD1_CE, 132_300)] {
        let status =
            unsafe { pith_audio::ffi::pith_audio_voices_pcm(seed, n, &mut out, &mut out_len) };
        assert_eq!(status, pith_audio::ffi::PITH_OK);
        query.extend_from_slice(unsafe { core::slice::from_raw_parts(out, out_len) });
        unsafe { pith_audio::ffi::pith_audio_free(out, out_len) };
    }

    let idx = unsafe { pith_audio::ffi::pith_audio_index_new() };
    for (seed, n) in [(0xD1_CE, 132_300), (0xF0_0D, 132_300), (0xE5_A5, 44_100)] {
        let status =
            unsafe { pith_audio::ffi::pith_audio_voices_pcm(seed, n, &mut out, &mut out_len) };
        assert_eq!(status, pith_audio::ffi::PITH_OK);
        let pcm = unsafe { core::slice::from_raw_parts(out, out_len) }.to_vec();
        unsafe { pith_audio::ffi::pith_audio_free(out, out_len) };
        let status =
            unsafe { pith_audio::ffi::pith_audio_index_add(idx, pcm.as_ptr(), pcm.len(), 1) };
        assert_eq!(status, pith_audio::ffi::PITH_OK);
    }

    let status = unsafe {
        pith_audio::ffi::pith_audio_match(
            idx,
            query.as_ptr(),
            query.len(),
            1,
            &mut out,
            &mut out_len,
        )
    };
    assert_eq!(status, pith_audio::ffi::PITH_OK);
    let stream = unsafe { core::slice::from_raw_parts(out, out_len) };
    assert_eq!(be32(stream, 0), 106); // query_frames
    assert_eq!(be32(stream, 4), 156); // query_peak_count
    assert_eq!(be32(stream, 8), 3); // match_count
    assert_eq!(be32(stream, 12), 0); // matches[0].id — the content member
    unsafe { pith_audio::ffi::pith_audio_free(out, out_len) };
    unsafe { pith_audio::ffi::pith_audio_index_free(idx) };
}

/// Refusals surface as status codes, never a crash: malformed WAV,
/// PCM geometry errors, null pointers, and null frees.
#[test]
fn refusals_and_null_safety() {
    let garbage = [0u8; 16];
    let mut out: *mut u8 = core::ptr::null_mut();
    let mut out_len: usize = 0;
    assert_eq!(
        unsafe {
            pith_audio::ffi::pith_audio_decode_wav(
                garbage.as_ptr(),
                garbage.len(),
                &mut out,
                &mut out_len,
            )
        },
        pith_audio::ffi::PITH_E_REJECTED
    );

    let pcm = pcm_le(&voices_samples(7, 8192));
    let idx = unsafe { pith_audio::ffi::pith_audio_index_new() };
    assert_eq!(
        unsafe { pith_audio::ffi::pith_audio_index_add(idx, pcm.as_ptr(), pcm.len(), 0) },
        pith_audio::ffi::PITH_E_INVALID
    );
    assert_eq!(
        unsafe {
            pith_audio::ffi::pith_audio_match(
                core::ptr::null(),
                pcm.as_ptr(),
                pcm.len(),
                1,
                &mut out,
                &mut out_len,
            )
        },
        pith_audio::ffi::PITH_E_INVALID
    );
    assert_eq!(
        unsafe {
            pith_audio::ffi::pith_audio_signature_pcm(
                pcm.as_ptr(),
                pcm.len() - 2,
                1,
                &mut out,
                &mut out_len,
            )
        },
        pith_audio::ffi::PITH_E_INVALID
    );

    unsafe { pith_audio::ffi::pith_audio_index_free(idx) };
    unsafe { pith_audio::ffi::pith_audio_index_free(core::ptr::null_mut()) };
    unsafe { pith_audio::ffi::pith_audio_free(core::ptr::null_mut(), 0) };
}
