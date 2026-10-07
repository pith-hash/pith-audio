//! The JNI surface of `pith-audio`: the `Java_hash_pith_audio_PithAudio_*`
//! exports the Java SDK (`sdk/java`) binds its `native` methods through.
//!
//! The C ABI of [`crate::ffi`] is untouched: JNI requires exports named
//! `Java_<package>_<Class>_<method>`, so the Java-facing shims live here
//! and forward every call to the existing `pith_*` C exports — same
//! status codes, same refusals, no second implementation of the
//! pipeline. The module is compiled out of the unit-test build
//! (`#[cfg(all(not(test), feature = "std"))]` at the registration site
//! in `lib.rs`); the integration tests in `tests/java_ffi.rs` exercise
//! every export against a synthetic environment so the coverage gate
//! still sees the glue.
//!
//! JNI conventions of this module (the Java-side contract):
//!
//! * every export takes the JNI environment first and the receiving
//!   class second (the methods are static), then the Java arguments;
//! * the status code crosses back through a trailing one-element
//!   `int[]` — the same `PITH_OK` / `PITH_E_INVALID` /
//!   `PITH_E_REJECTED` values the C ABI returns;
//! * stream results (decode facts, signature facts, match tables,
//!   synthesized PCM) cross back as a fresh `byte[]` — null unless the
//!   status is `PITH_OK` — and the C ABI's buffer is released with
//!   [`pith_audio_free`] before the export returns; Java never sees a
//!   raw pointer;
//! * the index handle crosses back as a `jlong` ([`pith_audio_index_new`]);
//!   a `0` handle is the null pointer: [`pith_audio_index_add`] and
//!   [`pith_audio_match`] map it to `PITH_E_INVALID`, and
//!   [`pith_audio_index_free`] ignores it (the C ABI's own rule);
//! * a null input array maps to `PITH_E_INVALID`, mirroring the C
//!   ABI's null-pointer rule; a null environment or status array
//!   short-circuits to a zero return without touching memory.
//!
//! The suite is zero-third-party (CI's `check-zero-deps.py` fails any
//! registry crate), so the JNI function table is hand-declared below:
//! every slot is pointer-sized and the positions are the fixed
//! `JNINativeInterface_` member order of `jni.h`. The slot indices were
//! parsed mechanically from the JDK 21 header and are validated
//! end-to-end against a live JVM every time the Java suite runs.

#![allow(unsafe_code)]
// The JNI typedefs keep the jni.h spelling (jint, jbyte, ...).
#![allow(non_camel_case_types)]

use core::ffi::c_void;

use crate::ffi::{
    PITH_E_INVALID, PITH_E_REJECTED, PITH_OK, pith_audio_decode_wav, pith_audio_free,
    pith_audio_index_add, pith_audio_index_free, pith_audio_index_new, pith_audio_match,
    pith_audio_signature_pcm, pith_audio_signature_wav, pith_audio_voices_pcm,
};

/// A JNI environment handle — C-mode `JNIEnv*`, a pointer to the
/// function table.
type JNIEnv = *const JniTable;

/// Any Java array reference; the glue only checks nullness before
/// handing arrays through the table.
type JArray = *mut c_void;

/// A Java `int[]` reference.
type JIntArray = *mut c_void;

/// A Java class object reference (static methods receive the class).
type JClass = *mut c_void;

/// `jbyte` per `jni.h`.
type jbyte = i8;
/// `jint`/`jsize` per `jni.h`.
type jint = i32;
/// `jlong` per `jni.h`.
type jlong = i64;

/// The JNI function-table slots this module calls.
///
/// Underscore-prefixed gap fields hold the slots between the used ones
/// (slot = field position; the four reserved pointers are part of the
/// prefix). Slot indices parsed from the JDK 21 `include/jni.h`:
/// `GetArrayLength` = 171, `NewByteArray` = 176,
/// `GetByteArrayRegion` = 200, `SetByteArrayRegion` = 208,
/// `SetIntArrayRegion` = 211.
#[repr(C)]
struct JniTable {
    /// Slots 0..=170: the four reserved pointers through
    /// `ReleaseStringUTFChars`.
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

/// The function table behind an environment handle.
///
/// # Safety
///
/// `env` must be a live JNI environment pointer.
unsafe fn table<'a>(env: *mut JNIEnv) -> &'a JniTable {
    // `env` points at the function-table pointer (C-mode `JNIEnv*`):
    // deref twice to reach the table itself.
    unsafe { &**env }
}

/// Copies a Java `byte[]` through the environment into an owned
/// buffer.
///
/// # Safety
///
/// `env` must be a live JNI environment and `array` a live `byte[]`
/// reference for the duration of the call; a null array is
/// [`PITH_E_INVALID`], mirroring the C ABI's null-pointer rule.
unsafe fn java_bytes(env: *mut JNIEnv, array: JArray) -> Result<Vec<u8>, i32> {
    if array.is_null() {
        return Err(PITH_E_INVALID);
    }
    let functions = unsafe { table(env) };
    let len = unsafe { (functions.get_array_length)(env, array) };
    if len < 0 {
        return Err(PITH_E_INVALID);
    }
    let mut bytes = vec![0u8; len as usize];
    unsafe { (functions.get_byte_array_region)(env, array, 0, len, bytes.as_mut_ptr().cast()) };
    Ok(bytes)
}

/// Builds a fresh Java `byte[]` holding `bytes`, or `None` if the
/// environment refuses the allocation.
///
/// # Safety
///
/// `env` must be a live JNI environment.
unsafe fn new_java_bytes(env: *mut JNIEnv, bytes: &[u8]) -> Option<JArray> {
    let functions = unsafe { table(env) };
    let array = unsafe { (functions.new_byte_array)(env, bytes.len() as jint) };
    if array.is_null() {
        return None;
    }
    unsafe {
        (functions.set_byte_array_region)(env, array, 0, bytes.len() as jint, bytes.as_ptr().cast())
    };
    Some(array)
}

/// Writes `value` into the one-element `int[]` status slot.
///
/// # Safety
///
/// `status` must be a live `int[]` of length ≥ 1 (checked by the
/// caller).
unsafe fn set_status(env: *mut JNIEnv, status: JIntArray, value: jint) {
    let functions = unsafe { table(env) };
    unsafe { (functions.set_int_array_region)(env, status, 0, 1, &value) };
}

/// Shared body of the four stream-returning operations: calls
/// `produce` (the C export with the out-parameter pair filled in),
/// copies the handed-out buffer into a fresh Java `byte[]`, releases
/// the buffer, and reports the status.
///
/// # Safety
///
/// `env` must be a live JNI environment and `status` a live `int[]`;
/// `produce` must follow the C ABI's out-buffer ownership contract.
unsafe fn stream_out(
    env: *mut JNIEnv,
    status: JIntArray,
    produce: impl FnOnce(*mut *mut u8, *mut usize) -> i32,
) -> JArray {
    let mut stream_ptr: *mut u8 = core::ptr::null_mut();
    let mut stream_len: usize = 0;
    let code = produce(&mut stream_ptr, &mut stream_len);
    if code != PITH_OK {
        unsafe { set_status(env, status, code) };
        return core::ptr::null_mut();
    }
    let stream = unsafe { core::slice::from_raw_parts(stream_ptr, stream_len) };
    let array = match unsafe { new_java_bytes(env, stream) } {
        Some(array) => array,
        None => {
            unsafe { pith_audio_free(stream_ptr, stream_len) };
            // The only failure left is the environment refusing the
            // array allocation; there is no dedicated code for it, so
            // it surfaces as a rejection, never as a panic.
            unsafe { set_status(env, status, PITH_E_REJECTED) };
            return core::ptr::null_mut();
        }
    };
    unsafe { pith_audio_free(stream_ptr, stream_len) };
    unsafe { set_status(env, status, PITH_OK) };
    array
}

/// The Java binding of [`pith_audio_decode_wav`]: the decode stream of
/// a RIFF/WAVE file.
///
/// `data` is the file bytes; on success the export returns the stream
/// the `reference.json` decode vectors are defined over —
/// `[0..4)` sample rate u32 BE, `[4..6)` channels u16 BE, `[6..8)`
/// bits-per-sample u16 BE, `[8..12)` frames u32 BE, `[12..16)` sample
/// count u32 BE, then the decoded i32 samples little-endian.
///
/// # Safety
///
/// `env` must be a live JNI environment and `data`/`status` live Java
/// array references for the duration of the call.
//
// Private: the JVM links the export by symbol name; a public Rust
// signature over the private table type would trip
// `private_interfaces`.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_audio_PithAudio_decodeWavNative(
    env: *mut JNIEnv,
    _class: JClass,
    data: JArray,
    status: JIntArray,
) -> JArray {
    if env.is_null() || status.is_null() {
        return core::ptr::null_mut();
    }
    let bytes = match unsafe { java_bytes(env, data) } {
        Ok(bytes) => bytes,
        Err(status_code) => {
            unsafe { set_status(env, status, status_code) };
            return core::ptr::null_mut();
        }
    };
    unsafe {
        stream_out(env, status, |out, out_len| {
            pith_audio_decode_wav(bytes.as_ptr(), bytes.len(), out, out_len)
        })
    }
}

/// The Java binding of [`pith_audio_signature_wav`]: the signature
/// stream of a 44 100 Hz RIFF/WAVE file (other rates are refused).
///
/// # Safety
///
/// `env` must be a live JNI environment and `data`/`status` live Java
/// array references for the duration of the call.
//
// Private: the JVM links the export by symbol name; a public Rust
// signature over the private table type would trip
// `private_interfaces`.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_audio_PithAudio_signatureWavNative(
    env: *mut JNIEnv,
    _class: JClass,
    data: JArray,
    status: JIntArray,
) -> JArray {
    if env.is_null() || status.is_null() {
        return core::ptr::null_mut();
    }
    let bytes = match unsafe { java_bytes(env, data) } {
        Ok(bytes) => bytes,
        Err(status_code) => {
            unsafe { set_status(env, status, status_code) };
            return core::ptr::null_mut();
        }
    };
    unsafe {
        stream_out(env, status, |out, out_len| {
            pith_audio_signature_wav(bytes.as_ptr(), bytes.len(), out, out_len)
        })
    }
}

/// The Java binding of [`pith_audio_signature_pcm`]: the signature
/// stream of interleaved little-endian i32 PCM at 44 100 Hz.
///
/// `channels` must divide the byte length into whole i32 frames and be
/// non-zero (`PITH_E_INVALID` otherwise).
///
/// # Safety
///
/// `env` must be a live JNI environment and `data`/`status` live Java
/// array references for the duration of the call.
//
// Private: the JVM links the export by symbol name; a public Rust
// signature over the private table type would trip
// `private_interfaces`.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_audio_PithAudio_signaturePcmNative(
    env: *mut JNIEnv,
    _class: JClass,
    data: JArray,
    channels: jint,
    status: JIntArray,
) -> JArray {
    if env.is_null() || status.is_null() {
        return core::ptr::null_mut();
    }
    let bytes = match unsafe { java_bytes(env, data) } {
        Ok(bytes) => bytes,
        Err(status_code) => {
            unsafe { set_status(env, status, status_code) };
            return core::ptr::null_mut();
        }
    };
    unsafe {
        stream_out(env, status, |out, out_len| {
            pith_audio_signature_pcm(bytes.as_ptr(), bytes.len(), channels as u32, out, out_len)
        })
    }
}

/// The Java binding of [`pith_audio_voices_pcm`]: the SplitMix64
/// square-voice corpus as `n_samples` × i32 LE — the same bytes the
/// synthetic `reference.json` vectors are built from. Synthesis runs
/// inside the cdylib, so no SDK re-implements the float pipeline.
///
/// # Safety
///
/// `env` must be a live JNI environment and `status` a live `int[]`.
//
// Private: the JVM links the export by symbol name; a public Rust
// signature over the private table type would trip
// `private_interfaces`.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_audio_PithAudio_voicesPcmNative(
    env: *mut JNIEnv,
    _class: JClass,
    seed: jlong,
    n_samples: jint,
    status: JIntArray,
) -> JArray {
    if env.is_null() || status.is_null() {
        return core::ptr::null_mut();
    }
    unsafe {
        stream_out(env, status, |out, out_len| {
            pith_audio_voices_pcm(seed as u64, n_samples as usize, out, out_len)
        })
    }
}

/// The Java binding of [`pith_audio_index_new`]: creates an empty
/// signature index. The handle is owned by the caller and must be
/// released with [`pith_audio_index_free`] (indexFreeNative).
///
/// # Safety
///
/// `env` must be a live JNI environment.
//
// Private: the JVM links the export by symbol name; a public Rust
// signature over the private table type would trip
// `private_interfaces`.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_audio_PithAudio_indexNewNative(
    env: *mut JNIEnv,
    _class: JClass,
) -> jlong {
    if env.is_null() {
        return 0;
    }
    (unsafe { pith_audio_index_new() }) as jlong
}

/// The Java binding of [`pith_audio_index_add`]: appends one signature
/// — extracted from interleaved little-endian i32 PCM — to the index.
/// A failed add leaves the handle alive and unchanged.
///
/// # Safety
///
/// `env` must be a live JNI environment and `data`/`status` live Java
/// array references for the duration of the call; `idx` must be a
/// handle from indexNewNative not yet freed (`0` is
/// [`PITH_E_INVALID`]).
//
// Private: the JVM links the export by symbol name; a public Rust
// signature over the private table type would trip
// `private_interfaces`.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_audio_PithAudio_indexAddNative(
    env: *mut JNIEnv,
    _class: JClass,
    idx: jlong,
    data: JArray,
    channels: jint,
    status: JIntArray,
) {
    if env.is_null() || status.is_null() {
        return;
    }
    let bytes = match unsafe { java_bytes(env, data) } {
        Ok(bytes) => bytes,
        Err(status_code) => {
            unsafe { set_status(env, status, status_code) };
            return;
        }
    };
    let code = unsafe {
        pith_audio_index_add(idx as *mut _, bytes.as_ptr(), bytes.len(), channels as u32)
    };
    unsafe { set_status(env, status, code) };
}

/// The Java binding of [`pith_audio_match`]: the match stream of the
/// query PCM against the index — query facts, match count, then one
/// record per matched id (id u32, delta_t i64, votes u32,
/// total_votes u32, all big-endian).
///
/// # Safety
///
/// `env` must be a live JNI environment and `data`/`status` live Java
/// array references for the duration of the call; `idx` must be a
/// handle from indexNewNative not yet freed (`0` is
/// [`PITH_E_INVALID`]).
//
// Private: the JVM links the export by symbol name; a public Rust
// signature over the private table type would trip
// `private_interfaces`.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_audio_PithAudio_matchNative(
    env: *mut JNIEnv,
    _class: JClass,
    idx: jlong,
    data: JArray,
    channels: jint,
    status: JIntArray,
) -> JArray {
    if env.is_null() || status.is_null() {
        return core::ptr::null_mut();
    }
    let bytes = match unsafe { java_bytes(env, data) } {
        Ok(bytes) => bytes,
        Err(status_code) => {
            unsafe { set_status(env, status, status_code) };
            return core::ptr::null_mut();
        }
    };
    unsafe {
        stream_out(env, status, |out, out_len| {
            pith_audio_match(
                idx as *const _,
                bytes.as_ptr(),
                bytes.len(),
                channels as u32,
                out,
                out_len,
            )
        })
    }
}

/// The Java binding of [`pith_audio_index_free`]: releases an index
/// handle. A `0` handle is ignored (the C ABI's own rule), so callers
/// can free unconditionally on the error path.
///
/// # Safety
///
/// `env` must be a live JNI environment; `idx` must not have been
/// freed before.
//
// Private: the JVM links the export by symbol name; a public Rust
// signature over the private table type would trip
// `private_interfaces`.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_audio_PithAudio_indexFreeNative(
    env: *mut JNIEnv,
    _class: JClass,
    idx: jlong,
) {
    if env.is_null() {
        return;
    }
    unsafe { pith_audio_index_free(idx as *mut _) };
}
