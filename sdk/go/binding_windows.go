// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build windows

package pithaudio

import (
	"fmt"
	"syscall"
	"unsafe"
)

// cIndexHandle is the opaque index handle the cdylib hands out. It is
// kept as a uintptr on Windows and only ever passed back to SyscallN —
// never converted to unsafe.Pointer (go vet's unsafeptr rule).
type cIndexHandle uintptr

func (h cIndexHandle) valid() bool { return h != 0 }

func (h *cIndexHandle) invalidate() { *h = 0 }

// openProc loads libPath and resolves name. The library is released
// before returning: on Windows FreeLibrary unmaps the cdylib, so the
// proc must be used (and its buffer copied out) inside the caller.
func openProc(libPath, name string) (proc uintptr, release func(), err error) {
	lib, err := syscall.LoadLibrary(libPath)
	if err != nil {
		return 0, nil, fmt.Errorf("pithaudio: LoadLibrary(%s): %w", libPath, err)
	}
	release = func() { syscall.FreeLibrary(lib) }
	proc, err = syscall.GetProcAddress(lib, name)
	if err != nil {
		release()
		return 0, nil, fmt.Errorf("pithaudio: symbol %s missing from %s: %w", name, libPath, err)
	}
	return proc, release, nil
}

// callBytesOut runs one bytes-out op (data, len[, extra...], out**,
// out_len*) and surfaces the proc release on every path.
func callBytesOut(libPath, name string, data *byte, n int, extra []uintptr, out **byte, outLen *uintptr) (int32, error) {
	proc, release, err := openProc(libPath, name)
	if err != nil {
		return 0, err
	}
	defer release()

	args := make([]uintptr, 0, len(extra)+4)
	args = append(args, uintptr(unsafe.Pointer(data)), uintptr(n))
	args = append(args, extra...)
	args = append(args, uintptr(unsafe.Pointer(out)), uintptr(unsafe.Pointer(outLen)))
	rc, _, _ := syscall.SyscallN(proc, args...)
	return int32(rc), nil
}

// ffiDecodeWav loads the cdylib with LoadLibrary (absolute path, no
// PATH involvement), resolves pith_audio_decode_wav and calls it. The
// returned buffer stays alive in the cdylib until ffiFree.
func ffiDecodeWav(libPath string, data *byte, n int, out **byte, outLen *uintptr) (int32, error) {
	return callBytesOut(libPath, "pith_audio_decode_wav", data, n, nil, out, outLen)
}

// ffiSignatureWav resolves pith_audio_signature_wav and calls it.
func ffiSignatureWav(libPath string, data *byte, n int, out **byte, outLen *uintptr) (int32, error) {
	return callBytesOut(libPath, "pith_audio_signature_wav", data, n, nil, out, outLen)
}

// ffiSignaturePcm resolves pith_audio_signature_pcm and calls it.
func ffiSignaturePcm(libPath string, data *byte, n int, channels uint32, out **byte, outLen *uintptr) (int32, error) {
	return callBytesOut(libPath, "pith_audio_signature_pcm", data, n, []uintptr{uintptr(channels)}, out, outLen)
}

// ffiVoicesPcm resolves pith_audio_voices_pcm and calls it: (seed,
// n_samples, out**, out_len*).
func ffiVoicesPcm(libPath string, seed uint64, nSamples int, out **byte, outLen *uintptr) (int32, error) {
	proc, release, err := openProc(libPath, "pith_audio_voices_pcm")
	if err != nil {
		return 0, err
	}
	defer release()
	rc, _, _ := syscall.SyscallN(proc,
		uintptr(seed),
		uintptr(nSamples),
		uintptr(unsafe.Pointer(out)),
		uintptr(unsafe.Pointer(outLen)),
	)
	return int32(rc), nil
}

// ffiIndexNew resolves pith_audio_index_new and returns the handle.
func ffiIndexNew(libPath string) (cIndexHandle, error) {
	proc, release, err := openProc(libPath, "pith_audio_index_new")
	if err != nil {
		return 0, err
	}
	defer release()
	rc, _, _ := syscall.SyscallN(proc)
	return cIndexHandle(rc), nil
}

// ffiIndexAdd resolves pith_audio_index_add and calls it.
func ffiIndexAdd(libPath string, h cIndexHandle, data *byte, n int, channels uint32) (int32, error) {
	proc, release, err := openProc(libPath, "pith_audio_index_add")
	if err != nil {
		return 0, err
	}
	defer release()
	rc, _, _ := syscall.SyscallN(proc,
		uintptr(h),
		uintptr(unsafe.Pointer(data)),
		uintptr(n),
		uintptr(channels),
	)
	return int32(rc), nil
}

// ffiIndexMatch resolves pith_audio_match and calls it.
func ffiIndexMatch(libPath string, h cIndexHandle, data *byte, n int, channels uint32, out **byte, outLen *uintptr) (int32, error) {
	proc, release, err := openProc(libPath, "pith_audio_match")
	if err != nil {
		return 0, err
	}
	defer release()
	rc, _, _ := syscall.SyscallN(proc,
		uintptr(h),
		uintptr(unsafe.Pointer(data)),
		uintptr(n),
		uintptr(channels),
		uintptr(unsafe.Pointer(out)),
		uintptr(unsafe.Pointer(outLen)),
	)
	return int32(rc), nil
}

// ffiIndexFree resolves pith_audio_index_free and releases the handle.
func ffiIndexFree(libPath string, h cIndexHandle) {
	proc, release, err := openProc(libPath, "pith_audio_index_free")
	if err != nil {
		return // the library vanished mid-flight; nothing to free
	}
	defer release()
	syscall.SyscallN(proc, uintptr(h))
}

// ffiFree resolves pith_audio_free and releases a buffer handed out by
// the bytes-out ops. Null is accepted (the cdylib ignores it).
func ffiFree(libPath string, ptr *byte, n uintptr) {
	proc, release, err := openProc(libPath, "pith_audio_free")
	if err != nil {
		return // the library vanished mid-flight; nothing to free
	}
	defer release()
	syscall.SyscallN(proc, uintptr(unsafe.Pointer(ptr)), n)
}
