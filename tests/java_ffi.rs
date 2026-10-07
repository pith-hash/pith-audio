//! Fake-JNI-environment coverage for the glue in `src/ffi_jni.rs`.
//!
//! `src/ffi_jni.rs` is compiled out of the unit-test build (the
//! `#[no_mangle]` exports would collide with the unit-test binary), so
//! this integration test drives every export through an `unsafe extern`
//! declaration against a synthetic environment: a zeroed function
//! table whose slots the glue calls carry test-local implementations
//! backed by a registry of fake Java arrays. The real-JVM proof is the
//! Java suite (`sdk/java`, `mvn test` against the built cdylib); this
//! file keeps the glue executed and visible to the coverage gate with
//! zero new dependencies (the suite's `check-zero-deps.py` gate forbids
//! registry crates, so the plain `std` mutexes stay unwrapped here).

#![allow(unsafe_code)]
// The JNI typedefs keep the jni.h spelling.
#![allow(non_camel_case_types)]

use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use pith_audio::reference::{match_stream, voices_samples};
use pith_audio::{Signature, build_index, match_signature, signature};

type JNIEnv = *const FakeTable;
type JArray = *mut c_void;
type JIntArray = *mut c_void;
type JClass = *mut c_void;
type jbyte = i8;
type jint = i32;
type jlong = i64;

/// Mirror of `src/ffi_jni.rs`'s function table — the same slot
/// positions (171, 176, 200, 208, 211; four reserved pointers in the
/// prefix).
#[repr(C)]
struct FakeTable {
    /// Slots 0..=170.
    _prefix: [*mut c_void; 171],
    /// Slot 171.
    get_array_length: unsafe extern "system" fn(env: *mut JNIEnv, array: JArray) -> jint,
    /// Slots 172..=175.
    _gap_before_new_byte_array: [*mut c_void; 4],
    /// Slot 176.
    new_byte_array: unsafe extern "system" fn(env: *mut JNIEnv, len: jint) -> JArray,
    /// Slots 177..=199.
    _gap_before_byte_region: [*mut c_void; 23],
    /// Slot 200.
    get_byte_array_region: unsafe extern "system" fn(
        env: *mut JNIEnv,
        array: JArray,
        start: jint,
        len: jint,
        buf: *mut jbyte,
    ),
    /// Slots 201..=207.
    _gap_before_set_byte_region: [*mut c_void; 7],
    /// Slot 208.
    set_byte_array_region: unsafe extern "system" fn(
        env: *mut JNIEnv,
        array: JArray,
        start: jint,
        len: jint,
        buf: *const jbyte,
    ),
    /// Slots 209..=210.
    _gap_before_set_int_region: [*mut c_void; 2],
    /// Slot 211.
    set_int_array_region: unsafe extern "system" fn(
        env: *mut JNIEnv,
        array: JIntArray,
        start: jint,
        len: jint,
        buf: *const jint,
    ),
}

// The exported symbols under test (linked from the crate's rlib).
unsafe extern "system" {
    fn Java_hash_pith_audio_PithAudio_decodeWavNative(
        env: *mut JNIEnv,
        class: JClass,
        data: JArray,
        status: JIntArray,
    ) -> JArray;

    fn Java_hash_pith_audio_PithAudio_signatureWavNative(
        env: *mut JNIEnv,
        class: JClass,
        data: JArray,
        status: JIntArray,
    ) -> JArray;

    fn Java_hash_pith_audio_PithAudio_signaturePcmNative(
        env: *mut JNIEnv,
        class: JClass,
        data: JArray,
        channels: jint,
        status: JIntArray,
    ) -> JArray;

    fn Java_hash_pith_audio_PithAudio_voicesPcmNative(
        env: *mut JNIEnv,
        class: JClass,
        seed: jlong,
        n_samples: jint,
        status: JIntArray,
    ) -> JArray;

    fn Java_hash_pith_audio_PithAudio_indexNewNative(env: *mut JNIEnv, class: JClass) -> jlong;

    fn Java_hash_pith_audio_PithAudio_indexAddNative(
        env: *mut JNIEnv,
        class: JClass,
        idx: jlong,
        data: JArray,
        channels: jint,
        status: JIntArray,
    );

    fn Java_hash_pith_audio_PithAudio_matchNative(
        env: *mut JNIEnv,
        class: JClass,
        idx: jlong,
        data: JArray,
        channels: jint,
        status: JIntArray,
    ) -> JArray;

    fn Java_hash_pith_audio_PithAudio_indexFreeNative(env: *mut JNIEnv, class: JClass, idx: jlong);
}

/// The state of one native call under test.
struct FakeCall {
    arrays: HashMap<usize, Vec<u8>>,
    next_handle: usize,
    out_bytes: Vec<u8>,
    out_ints: Vec<i32>,
    fail_new_byte_array: bool,
}

static CALL: LazyLock<Mutex<Option<FakeCall>>> = LazyLock::new(|| Mutex::new(None));
static SERIAL: Mutex<()> = Mutex::new(());

fn with_state<T>(f: impl FnOnce(&mut FakeCall) -> T) -> T {
    let mut guard = CALL.lock().unwrap();
    let state = guard.as_mut().expect("no fake call state installed");
    f(state)
}

/// Registers a fake `byte[]`; returns its opaque handle.
fn byte_array(bytes: Vec<u8>) -> JArray {
    with_state(|state| {
        state.next_handle += 8;
        let handle = state.next_handle;
        state.arrays.insert(handle, bytes);
        handle as JArray
    })
}

/// `GetArrayLength` (slot 171).
unsafe extern "system" fn fake_get_array_length(_env: *mut JNIEnv, array: JArray) -> jint {
    with_state(|state| {
        state
            .arrays
            .get(&(array as usize))
            .map_or(-1, |bytes| bytes.len() as jint)
    })
}

/// `NewByteArray` (slot 176).
unsafe extern "system" fn fake_new_byte_array(_env: *mut JNIEnv, len: jint) -> JArray {
    let fresh = with_state(|state| {
        if state.fail_new_byte_array {
            return None;
        }
        state.next_handle += 8;
        let handle = state.next_handle;
        state.arrays.insert(handle, vec![0; len as usize]);
        Some(handle as JArray)
    });
    fresh.unwrap_or_else(std::ptr::null_mut)
}

/// `GetByteArrayRegion` (slot 200).
unsafe extern "system" fn fake_get_byte_array_region(
    _env: *mut JNIEnv,
    array: JArray,
    start: jint,
    len: jint,
    buf: *mut jbyte,
) {
    let bytes = with_state(|state| {
        state.arrays.get(&(array as usize)).expect("live array")
            [start as usize..start as usize + len as usize]
            .to_vec()
    });
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.cast(), bytes.len()) };
}

/// `SetByteArrayRegion` (slot 208): copies into the target array —
/// faithful to the JVM — and records the written bytes.
unsafe extern "system" fn fake_set_byte_array_region(
    _env: *mut JNIEnv,
    array: JArray,
    start: jint,
    len: jint,
    buf: *const jbyte,
) {
    let written = unsafe { std::slice::from_raw_parts(buf.cast::<u8>(), len as usize) }.to_vec();
    with_state(|state| {
        let target = state.arrays.get_mut(&(array as usize)).expect("live array");
        target[start as usize..start as usize + len as usize].copy_from_slice(&written);
        state.out_bytes.extend_from_slice(&written);
    });
}

/// `SetIntArrayRegion` (slot 211): copies into the target array —
/// faithful to the JVM — and records the written values.
unsafe extern "system" fn fake_set_int_array_region(
    _env: *mut JNIEnv,
    array: JIntArray,
    _start: jint,
    len: jint,
    buf: *const jint,
) {
    unsafe { std::ptr::copy_nonoverlapping(buf, array as *mut jint, len as usize) };
    let mut guard = CALL.lock().unwrap();
    let state = guard.as_mut().expect("no fake call state installed");
    for i in 0..len as usize {
        state.out_ints.push(unsafe { *buf.add(i) });
    }
}

/// A zero-initialized `FakeTable`, leaked.
///
/// Raw `alloc_zeroed` bytes rather than `mem::zeroed`: the latter
/// runtime-refuses zeroed fn-pointer fields, while the former is just
/// memory — every slot the glue calls is assigned below before use.
fn zeroed_table() -> *mut FakeTable {
    let raw = unsafe { std::alloc::alloc_zeroed(std::alloc::Layout::new::<FakeTable>()) };
    assert!(!raw.is_null(), "alloc_zeroed failed");
    raw.cast::<FakeTable>()
}

/// Runs `f` against a synthetic environment; returns its result, the
/// bytes the glue wrote into Java arrays, and the int slots it wrote.
fn with_fake_env<T>(f: impl FnOnce(*mut JNIEnv) -> T) -> (T, Vec<u8>, Vec<i32>) {
    let _serial = SERIAL.lock().unwrap();
    *CALL.lock().unwrap() = Some(FakeCall {
        arrays: HashMap::new(),
        next_handle: 0,
        out_bytes: Vec::new(),
        out_ints: Vec::new(),
        fail_new_byte_array: false,
    });

    let table = zeroed_table();
    unsafe {
        (*table).get_array_length = fake_get_array_length;
        (*table).new_byte_array = fake_new_byte_array;
        (*table).get_byte_array_region = fake_get_byte_array_region;
        (*table).set_byte_array_region = fake_set_byte_array_region;
        (*table).set_int_array_region = fake_set_int_array_region;
    }
    let functions: *const FakeTable = table;
    let env: *mut JNIEnv = Box::into_raw(Box::new(functions));

    let result = f(env);
    let state = CALL.lock().unwrap().take().expect("fake call state");
    (result, state.out_bytes, state.out_ints)
}

/// A live one-element status array; read it back with [`read_status`].
fn status_slot() -> JIntArray {
    Box::into_raw(Box::new([i32::MIN; 1])) as JIntArray
}

/// Reads the slot's content and releases it.
fn read_status(slot: JIntArray) -> i32 {
    let value = unsafe { *(slot as *mut jint) };
    unsafe { drop(Box::from_raw(slot as *mut jint)) };
    value
}

/// The committed tone.wav fixture.
fn tone_wav() -> Vec<u8> {
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tone.wav"))
        .expect("fixture")
}

/// The decode stream the C export produces for `bytes`.
fn c_decode(bytes: &[u8]) -> Vec<u8> {
    let mut stream_ptr: *mut u8 = std::ptr::null_mut();
    let mut stream_len: usize = 0;
    let code = unsafe {
        pith_audio::ffi::pith_audio_decode_wav(
            bytes.as_ptr(),
            bytes.len(),
            &mut stream_ptr,
            &mut stream_len,
        )
    };
    assert_eq!(code, 0);
    let stream = unsafe { std::slice::from_raw_parts(stream_ptr, stream_len) }.to_vec();
    unsafe { pith_audio::ffi::pith_audio_free(stream_ptr, stream_len) };
    stream
}

#[test]
fn jni_decode_matches_the_c_export() {
    let input = tone_wav();
    let expected = c_decode(&input);
    let ((stream, status), produced, _recorded) = with_fake_env(|env| unsafe {
        let data = byte_array(input);
        let slot = status_slot();
        let stream =
            Java_hash_pith_audio_PithAudio_decodeWavNative(env, std::ptr::null_mut(), data, slot);
        (stream, read_status(slot))
    });
    assert_eq!(status, 0, "status");
    assert!(!stream.is_null(), "stream array");
    assert_eq!(produced, expected, "decode stream bytes");
}

#[test]
fn jni_voices_matches_the_c_export() {
    let expected = {
        let mut stream_ptr: *mut u8 = std::ptr::null_mut();
        let mut stream_len: usize = 0;
        let code = unsafe {
            pith_audio::ffi::pith_audio_voices_pcm(7, 8192, &mut stream_ptr, &mut stream_len)
        };
        assert_eq!(code, 0);
        let stream = unsafe { std::slice::from_raw_parts(stream_ptr, stream_len) }.to_vec();
        unsafe { pith_audio::ffi::pith_audio_free(stream_ptr, stream_len) };
        stream
    };
    let ((stream, status), produced, _recorded) = with_fake_env(|env| unsafe {
        let slot = status_slot();
        let stream = Java_hash_pith_audio_PithAudio_voicesPcmNative(
            env,
            std::ptr::null_mut(),
            7,
            8192,
            slot,
        );
        (stream, read_status(slot))
    });
    assert_eq!(status, 0, "status");
    assert!(!stream.is_null(), "stream array");
    assert_eq!(produced, expected, "voices bytes");
    assert_eq!(produced.len(), 8192 * 4);
}

#[test]
fn jni_signature_wav_non_wav_is_rejected() {
    let ((stream, status), _produced, _recorded) = with_fake_env(|env| unsafe {
        let data = byte_array(b"not a wav file at all........".to_vec());
        let slot = status_slot();
        let stream = Java_hash_pith_audio_PithAudio_signatureWavNative(
            env,
            std::ptr::null_mut(),
            data,
            slot,
        );
        (stream, read_status(slot))
    });
    assert_eq!(status, -2, "status");
    assert!(stream.is_null());
}

#[test]
fn jni_signature_pcm_partial_sample_is_invalid() {
    let ((stream, status), _produced, _recorded) = with_fake_env(|env| unsafe {
        let data = byte_array(vec![1, 0, 0]);
        let slot = status_slot();
        let stream = Java_hash_pith_audio_PithAudio_signaturePcmNative(
            env,
            std::ptr::null_mut(),
            data,
            1,
            slot,
        );
        (stream, read_status(slot))
    });
    assert_eq!(status, -1, "status");
    assert!(stream.is_null());
}

#[test]
fn jni_signature_pcm_zero_channels_is_invalid() {
    let ((stream, status), _produced, _recorded) = with_fake_env(|env| unsafe {
        let data = byte_array(vec![0, 0, 0, 0]);
        let slot = status_slot();
        let stream = Java_hash_pith_audio_PithAudio_signaturePcmNative(
            env,
            std::ptr::null_mut(),
            data,
            0,
            slot,
        );
        (stream, read_status(slot))
    });
    assert_eq!(status, -1, "status");
    assert!(stream.is_null());
}

#[test]
fn jni_decode_null_data_is_invalid() {
    let ((stream, status), _produced, _recorded) = with_fake_env(|env| unsafe {
        let slot = status_slot();
        let stream = Java_hash_pith_audio_PithAudio_decodeWavNative(
            env,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            slot,
        );
        (stream, read_status(slot))
    });
    assert_eq!(status, -1, "status");
    assert!(stream.is_null());
}

#[test]
fn jni_decode_environment_refusing_the_array_is_rejected() {
    let ((stream, status), _produced, _recorded) = with_fake_env(|env| unsafe {
        let data = byte_array(tone_wav());
        let slot = status_slot();
        with_state(|state| state.fail_new_byte_array = true);
        let stream =
            Java_hash_pith_audio_PithAudio_decodeWavNative(env, std::ptr::null_mut(), data, slot);
        (stream, read_status(slot))
    });
    assert_eq!(status, -2, "status");
    assert!(stream.is_null());
}

#[test]
fn jni_decode_null_status_short_circuits() {
    let (stream, _produced, _recorded) = with_fake_env(|env| unsafe {
        let data = byte_array(tone_wav());
        Java_hash_pith_audio_PithAudio_decodeWavNative(
            env,
            std::ptr::null_mut(),
            data,
            std::ptr::null_mut(),
        )
    });
    assert!(stream.is_null());
}

#[test]
fn jni_null_environment_short_circuits() {
    let ((stream, handle), _produced, _recorded) = with_fake_env(|_env| unsafe {
        let slot = status_slot();
        let data = byte_array(tone_wav());
        let stream = Java_hash_pith_audio_PithAudio_decodeWavNative(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            data,
            slot,
        );
        let handle = Java_hash_pith_audio_PithAudio_indexNewNative(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        (stream, handle)
    });
    assert!(stream.is_null());
    assert_eq!(handle, 0);
}

#[test]
fn jni_index_lifecycle_matches_the_safe_core() {
    // The indexed corpus and the query, mirroring the offset-match
    // vector's shape at a smaller scale.
    let member = voices_samples(0x0000_D1CE, 44_100);
    let other = voices_samples(0x0000_F00D, 44_100);
    let query_samples = {
        let prefix = voices_samples(0x0000_BEEF, 22_050);
        let mut all = prefix;
        all.extend_from_slice(&member);
        all
    };

    // The safe-core expectation for the same index + query.
    let member_sig: Signature = signature(&member, 1).expect("member");
    let other_sig: Signature = signature(&other, 1).expect("other");
    let query_sig: Signature = signature(&query_samples, 1).expect("query");
    let index = build_index(&[member_sig, other_sig]);
    let expected = match_stream(&query_sig, &match_signature(&query_sig, &index));

    let pcm_bytes = |samples: &[i32]| -> Vec<u8> {
        let mut out = Vec::with_capacity(samples.len() * 4);
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    };

    let ((stream, status), produced, _recorded) = with_fake_env(|env| unsafe {
        let idx = Java_hash_pith_audio_PithAudio_indexNewNative(env, std::ptr::null_mut());
        assert!(idx != 0, "index handle");
        for pcm in [&member, &other] {
            let slot = status_slot();
            let data = byte_array(pcm_bytes(pcm));
            Java_hash_pith_audio_PithAudio_indexAddNative(
                env,
                std::ptr::null_mut(),
                idx,
                data,
                1,
                slot,
            );
            assert_eq!(read_status(slot), 0, "add status");
        }
        let slot = status_slot();
        let data = byte_array(pcm_bytes(&query_samples));
        let stream = Java_hash_pith_audio_PithAudio_matchNative(
            env,
            std::ptr::null_mut(),
            idx,
            data,
            1,
            slot,
        );
        let status = read_status(slot);
        Java_hash_pith_audio_PithAudio_indexFreeNative(env, std::ptr::null_mut(), idx);
        (stream, status)
    });
    assert_eq!(status, 0, "match status");
    assert!(!stream.is_null(), "match stream array");
    assert_eq!(produced, expected, "match stream bytes");
}

#[test]
fn jni_index_null_handle_is_invalid() {
    let ((_stream, status), _produced, _recorded) = with_fake_env(|env| unsafe {
        let slot = status_slot();
        let data = byte_array(vec![0, 0, 0, 0]);
        let stream =
            Java_hash_pith_audio_PithAudio_matchNative(env, std::ptr::null_mut(), 0, data, 1, slot);
        (stream, read_status(slot))
    });
    assert_eq!(status, -1, "status");
}

#[test]
fn jni_index_add_null_handle_is_invalid() {
    let ((status,), _produced, _recorded) = with_fake_env(|env| unsafe {
        let slot = status_slot();
        let data = byte_array(vec![0, 0, 0, 0]);
        Java_hash_pith_audio_PithAudio_indexAddNative(env, std::ptr::null_mut(), 0, data, 1, slot);
        (read_status(slot),)
    });
    assert_eq!(status, -1, "status");
}

#[test]
fn jni_stale_array_handle_is_invalid() {
    let ((stream, status), _produced, _recorded) = with_fake_env(|env| unsafe {
        let data = byte_array(tone_wav());
        // Unregister the handle behind the JVM's back: the next
        // GetArrayLength reports -1, which the glue maps to INVALID.
        let stale = data as usize;
        with_state(|state| state.arrays.remove(&stale));
        let slot = status_slot();
        let stream =
            Java_hash_pith_audio_PithAudio_decodeWavNative(env, std::ptr::null_mut(), data, slot);
        (stream, read_status(slot))
    });
    assert_eq!(status, -1, "status");
    assert!(stream.is_null());
}

#[test]
fn jni_remaining_streams_null_status_short_circuit() {
    let ((wav, pcm, voices, matched), _produced, _recorded) = with_fake_env(|env| unsafe {
        let data = byte_array(tone_wav());
        let pcm_data = byte_array(vec![0, 0, 0, 0]);
        let wav = Java_hash_pith_audio_PithAudio_signatureWavNative(
            env,
            std::ptr::null_mut(),
            data,
            std::ptr::null_mut(),
        );
        let pcm = Java_hash_pith_audio_PithAudio_signaturePcmNative(
            env,
            std::ptr::null_mut(),
            pcm_data,
            1,
            std::ptr::null_mut(),
        );
        let voices = Java_hash_pith_audio_PithAudio_voicesPcmNative(
            env,
            std::ptr::null_mut(),
            7,
            128,
            std::ptr::null_mut(),
        );
        let matched = Java_hash_pith_audio_PithAudio_matchNative(
            env,
            std::ptr::null_mut(),
            0,
            pcm_data,
            1,
            std::ptr::null_mut(),
        );
        (wav, pcm, voices, matched)
    });
    assert!(wav.is_null());
    assert!(pcm.is_null());
    assert!(voices.is_null());
    assert!(matched.is_null());
}

#[test]
fn jni_index_add_null_status_short_circuits() {
    let ((), _produced, recorded) = with_fake_env(|env| unsafe {
        let data = byte_array(vec![0, 0, 0, 0]);
        Java_hash_pith_audio_PithAudio_indexAddNative(
            env,
            std::ptr::null_mut(),
            0,
            data,
            1,
            std::ptr::null_mut(),
        );
    });
    assert!(recorded.is_empty(), "no status written");
}

#[test]
fn jni_index_free_null_environment_is_a_noop() {
    with_fake_env(|_env| unsafe {
        Java_hash_pith_audio_PithAudio_indexFreeNative(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        );
    });
}
