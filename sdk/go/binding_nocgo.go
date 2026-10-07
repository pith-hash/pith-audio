// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build !windows && !cgo

package pithaudio

import "fmt"

// cIndexHandle mirrors the windows-leg handle shape; unused without
// cgo (there is no way to load the cdylib).
type cIndexHandle uintptr

func (h cIndexHandle) valid() bool { return h != 0 }

func (h *cIndexHandle) invalidate() { *h = 0 }

// ffiDecodeWav is unavailable without cgo on unix: there is no
// pure-Go dlopen in the standard library. Build with CGO_ENABLED=1
// (the CD pipeline always does).
func ffiDecodeWav(string, *byte, int, **byte, *uintptr) (int32, error) {
	return 0, fmt.Errorf("pithaudio: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiSignatureWav mirrors the unavailable decode.
func ffiSignatureWav(string, *byte, int, **byte, *uintptr) (int32, error) {
	return 0, fmt.Errorf("pithaudio: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiSignaturePcm mirrors the unavailable decode.
func ffiSignaturePcm(string, *byte, int, uint32, **byte, *uintptr) (int32, error) {
	return 0, fmt.Errorf("pithaudio: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiVoicesPcm mirrors the unavailable decode.
func ffiVoicesPcm(string, uint64, int, **byte, *uintptr) (int32, error) {
	return 0, fmt.Errorf("pithaudio: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiIndexNew mirrors the unavailable decode.
func ffiIndexNew(string) (cIndexHandle, error) {
	return 0, fmt.Errorf("pithaudio: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiIndexAdd mirrors the unavailable decode.
func ffiIndexAdd(string, cIndexHandle, *byte, int, uint32) (int32, error) {
	return 0, fmt.Errorf("pithaudio: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiIndexMatch mirrors the unavailable decode.
func ffiIndexMatch(string, cIndexHandle, *byte, int, uint32, **byte, *uintptr) (int32, error) {
	return 0, fmt.Errorf("pithaudio: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiIndexFree mirrors the unavailable decode.
func ffiIndexFree(string, cIndexHandle) {}

// ffiFree mirrors the unavailable decode.
func ffiFree(string, *byte, uintptr) {}
