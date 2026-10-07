// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build !windows && cgo

package pithaudio

/*
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef int32_t (*pith_bytes_out_fn)(const uint8_t *, size_t, uint8_t **, size_t *);
typedef int32_t (*pith_sig_pcm_fn)(const uint8_t *, size_t, uint32_t, uint8_t **, size_t *);
typedef int32_t (*pith_voices_fn)(uint64_t, size_t, uint8_t **, size_t *);
typedef void *(*pith_index_new_fn)();
typedef int32_t (*pith_index_add_fn)(void *, const uint8_t *, size_t, uint32_t);
typedef int32_t (*pith_match_fn)(const void *, const uint8_t *, size_t, uint32_t, uint8_t **, size_t *);
typedef void (*pith_index_free_fn)(void *);
typedef void (*pith_free_fn)(uint8_t *, size_t);

static int32_t pith_call_bytes_out(void *fn, const uint8_t *data, size_t len,
                                   uint8_t **out, size_t *out_len) {
    return ((pith_bytes_out_fn)fn)(data, len, out, out_len);
}

static int32_t pith_call_sig_pcm(void *fn, const uint8_t *data, size_t len, uint32_t ch,
                                 uint8_t **out, size_t *out_len) {
    return ((pith_sig_pcm_fn)fn)(data, len, ch, out, out_len);
}

static int32_t pith_call_voices(void *fn, uint64_t seed, size_t n,
                                uint8_t **out, size_t *out_len) {
    return ((pith_voices_fn)fn)(seed, n, out, out_len);
}

static void *pith_call_index_new(void *fn) {
    return ((pith_index_new_fn)fn)();
}

static int32_t pith_call_index_add(void *fn, void *idx, const uint8_t *data, size_t len, uint32_t ch) {
    return ((pith_index_add_fn)fn)(idx, data, len, ch);
}

static int32_t pith_call_match(void *fn, const void *idx, const uint8_t *data, size_t len, uint32_t ch,
                               uint8_t **out, size_t *out_len) {
    return ((pith_match_fn)fn)(idx, data, len, ch, out, out_len);
}

static void pith_call_index_free(void *fn, void *idx) {
    ((pith_index_free_fn)fn)(idx);
}

static void pith_call_free(void *fn, uint8_t *ptr, size_t len) {
    ((pith_free_fn)fn)(ptr, len);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// cIndexHandle is the opaque index handle the cdylib hands out: a C
// pointer, stored as-is and only ever passed back through the static
// call helpers.
type cIndexHandle struct {
	p unsafe.Pointer
}

func (h cIndexHandle) valid() bool { return h.p != nil }

func (h *cIndexHandle) invalidate() { h.p = nil }

// ffiSymbols resolves every exported symbol of one open cdylib handle.
type cSymbols struct {
	decodeWav    unsafe.Pointer
	signatureWav unsafe.Pointer
	signaturePcm unsafe.Pointer
	voicesPcm    unsafe.Pointer
	indexNew     unsafe.Pointer
	indexAdd     unsafe.Pointer
	match        unsafe.Pointer
	indexFree    unsafe.Pointer
	free         unsafe.Pointer
}

// ffiSymbols resolves the nine exported symbols by name.
func ffiSymbols(handle unsafe.Pointer, libPath string) (cSymbols, error) {
	var s cSymbols
	for _, e := range []struct {
		name string
		dst  *unsafe.Pointer
	}{
		{"pith_audio_decode_wav", &s.decodeWav},
		{"pith_audio_signature_wav", &s.signatureWav},
		{"pith_audio_signature_pcm", &s.signaturePcm},
		{"pith_audio_voices_pcm", &s.voicesPcm},
		{"pith_audio_index_new", &s.indexNew},
		{"pith_audio_index_add", &s.indexAdd},
		{"pith_audio_match", &s.match},
		{"pith_audio_index_free", &s.indexFree},
		{"pith_audio_free", &s.free},
	} {
		cName := C.CString(e.name)
		sym := C.dlsym(handle, cName)
		C.free(unsafe.Pointer(cName))
		if sym == nil {
			return cSymbols{}, fmt.Errorf("pithaudio: symbol %s missing from %s", e.name, libPath)
		}
		*e.dst = sym
	}
	return s, nil
}

// openCdylib dlopens libPath with error text surfaced verbatim.
func openCdylib(libPath string) (unsafe.Pointer, error) {
	cPath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cPath))
	handle := C.dlopen(cPath, C.RTLD_NOW|C.RTLD_LOCAL)
	if handle == nil {
		msg := "unknown dlopen failure"
		if e := C.dlerror(); e != nil {
			msg = C.GoString(e)
		}
		return nil, fmt.Errorf("pithaudio: dlopen(%s): %s", libPath, msg)
	}
	return handle, nil
}

// ffiDecodeWav opens the cdylib, resolves pith_audio_decode_wav and
// calls it. The handle is released before returning; repeated calls
// reuse the loader's own refcount.
func ffiDecodeWav(libPath string, data *byte, n int, out **byte, outLen *uintptr) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)
	syms, err := ffiSymbols(handle, libPath)
	if err != nil {
		return 0, err
	}
	var cOut *C.uint8_t
	var cLen C.size_t
	rc := C.pith_call_bytes_out(syms.decodeWav, (*C.uint8_t)(unsafe.Pointer(data)), C.size_t(n), &cOut, &cLen)
	*out = (*byte)(unsafe.Pointer(cOut))
	*outLen = uintptr(cLen)
	return int32(rc), nil
}

// ffiSignatureWav mirrors ffiDecodeWav for pith_audio_signature_wav.
func ffiSignatureWav(libPath string, data *byte, n int, out **byte, outLen *uintptr) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)
	syms, err := ffiSymbols(handle, libPath)
	if err != nil {
		return 0, err
	}
	var cOut *C.uint8_t
	var cLen C.size_t
	rc := C.pith_call_bytes_out(syms.signatureWav, (*C.uint8_t)(unsafe.Pointer(data)), C.size_t(n), &cOut, &cLen)
	*out = (*byte)(unsafe.Pointer(cOut))
	*outLen = uintptr(cLen)
	return int32(rc), nil
}

// ffiSignaturePcm mirrors ffiDecodeWav for pith_audio_signature_pcm.
func ffiSignaturePcm(libPath string, data *byte, n int, channels uint32, out **byte, outLen *uintptr) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)
	syms, err := ffiSymbols(handle, libPath)
	if err != nil {
		return 0, err
	}
	var cOut *C.uint8_t
	var cLen C.size_t
	rc := C.pith_call_sig_pcm(syms.signaturePcm, (*C.uint8_t)(unsafe.Pointer(data)), C.size_t(n), C.uint32_t(channels), &cOut, &cLen)
	*out = (*byte)(unsafe.Pointer(cOut))
	*outLen = uintptr(cLen)
	return int32(rc), nil
}

// ffiVoicesPcm mirrors ffiDecodeWav for pith_audio_voices_pcm.
func ffiVoicesPcm(libPath string, seed uint64, nSamples int, out **byte, outLen *uintptr) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)
	syms, err := ffiSymbols(handle, libPath)
	if err != nil {
		return 0, err
	}
	var cOut *C.uint8_t
	var cLen C.size_t
	rc := C.pith_call_voices(syms.voicesPcm, C.uint64_t(seed), C.size_t(nSamples), &cOut, &cLen)
	*out = (*byte)(unsafe.Pointer(cOut))
	*outLen = uintptr(cLen)
	return int32(rc), nil
}

// ffiIndexNew resolves pith_audio_index_new and returns the handle.
func ffiIndexNew(libPath string) (cIndexHandle, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return cIndexHandle{}, err
	}
	defer C.dlclose(handle)
	syms, err := ffiSymbols(handle, libPath)
	if err != nil {
		return cIndexHandle{}, err
	}
	return cIndexHandle{p: C.pith_call_index_new(syms.indexNew)}, nil
}

// ffiIndexAdd resolves pith_audio_index_add and calls it.
func ffiIndexAdd(libPath string, h cIndexHandle, data *byte, n int, channels uint32) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)
	syms, err := ffiSymbols(handle, libPath)
	if err != nil {
		return 0, err
	}
	rc := C.pith_call_index_add(syms.indexAdd, h.p, (*C.uint8_t)(unsafe.Pointer(data)), C.size_t(n), C.uint32_t(channels))
	return int32(rc), nil
}

// ffiIndexMatch resolves pith_audio_match and calls it.
func ffiIndexMatch(libPath string, h cIndexHandle, data *byte, n int, channels uint32, out **byte, outLen *uintptr) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)
	syms, err := ffiSymbols(handle, libPath)
	if err != nil {
		return 0, err
	}
	var cOut *C.uint8_t
	var cLen C.size_t
	rc := C.pith_call_match(syms.match, h.p, (*C.uint8_t)(unsafe.Pointer(data)), C.size_t(n), C.uint32_t(channels), &cOut, &cLen)
	*out = (*byte)(unsafe.Pointer(cOut))
	*outLen = uintptr(cLen)
	return int32(rc), nil
}

// ffiIndexFree resolves pith_audio_index_free and releases the handle.
func ffiIndexFree(libPath string, h cIndexHandle) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return // the library vanished mid-flight; nothing to free
	}
	defer C.dlclose(handle)
	syms, err := ffiSymbols(handle, libPath)
	if err != nil {
		return
	}
	C.pith_call_index_free(syms.indexFree, h.p)
}

// ffiFree resolves pith_audio_free and releases a buffer handed out by
// the bytes-out ops. Null is accepted (the cdylib ignores it).
func ffiFree(libPath string, ptr *byte, n uintptr) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return
	}
	defer C.dlclose(handle)
	syms, err := ffiSymbols(handle, libPath)
	if err != nil {
		return
	}
	C.pith_call_free(syms.free, (*C.uint8_t)(unsafe.Pointer(ptr)), C.size_t(n))
}
