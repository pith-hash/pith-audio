// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

// Package pithaudio provides Go bindings for the pith-audio Rust
// cdylib: RIFF/WAVE decode, spectral-peak signatures, the SplitMix64
// square-voice synthesis and delta-t histogram matching.
//
// The single Rust core (built by `cargo build --release`) is loaded at
// runtime; the package carries zero module dependencies. On unix the
// cdylib is opened with dlopen through cgo, on Windows with
// LoadLibrary through the standard syscall package — both resolve the
// library through the same discovery chain, so `go build ./... &&
// go test ./...` works unchanged on every OS the CD matrix builds.
//
// Discovery order (the suite's cdylib convention):
//
//  1. PITH_CDYLIB — an explicit cdylib file path;
//  2. PITH_CDYLIB_DIR — a directory scanned for the cdylib names (the
//     CD pipeline points this at target/release);
//  3. <repo root>/target/release — the repository working-tree layout,
//     anchored at this package's source directory, so a source
//     checkout runs against a local cargo build unconfigured.
//
// All byte streams are big-endian headers plus little-endian i32
// bodies (see the Parse helpers for the exact layouts):
//
//   - DecodeWav produces the decode stream the pcm_i32_le_* digests
//     are defined over;
//   - SignatureWav / SignaturePcm produce the signature stream;
//   - VoicesPcm synthesizes the SplitMix64 square-voice corpus inside
//     the cdylib — the same pure recipe the synthetic reference.json
//     vectors are built from, so no SDK-side float reimplementation
//     exists and the vectors carry zero platform sensitivity;
//   - NewIndex / Index.Add / Index.Match build an incremental
//     signature index (ids are insertion order) and produce the match
//     stream.
package pithaudio

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"
)

// Status codes returned by the cdylib's C ABI.
const (
	// StatusOK: success.
	StatusOK int32 = 0
	// StatusInvalid: a caller argument is invalid (a null pointer, a
	// length that is not a whole number of i32 samples, or a
	// zero/oversized channel count).
	StatusInvalid int32 = -1
	// StatusRejected: the core pipeline refused the input (malformed
	// WAV, or PCM the signature stage rejects).
	StatusRejected int32 = -2
)

// cdylibNames are the file names cargo may drop into the build
// directory, per platform (windows / linux / macOS).
var cdylibNames = []string{"pith_audio.dll", "libpith_audio.so", "libpith_audio.dylib"}

// FfiError reports a non-zero status code from the cdylib.
type FfiError struct {
	// Op is the FFI operation name.
	Op string
	// Status is the raw status code the FFI returned.
	Status int32
}

func (e *FfiError) Error() string {
	kind := "unknown failure"
	switch e.Status {
	case StatusInvalid:
		kind = "invalid argument"
	case StatusRejected:
		kind = "input rejected"
	}
	return fmt.Sprintf("%s failed: %s (status %d)", e.Op, kind, e.Status)
}

// FindCdylib locates the cdylib through the suite's discovery chain.
func FindCdylib() (string, error) {
	if p := os.Getenv("PITH_CDYLIB"); p != "" {
		if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
			return filepath.Abs(p)
		}
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("pithaudio: cannot locate the package source directory")
	}
	pkgDir := filepath.Dir(thisFile)
	repoRoot := filepath.Dir(filepath.Dir(pkgDir)) // sdk/go -> sdk -> repo root

	var dirs []string
	if env := os.Getenv("PITH_CDYLIB_DIR"); env != "" {
		dirs = append(dirs, env)
		if !filepath.IsAbs(env) {
			dirs = append(dirs, filepath.Join(repoRoot, env))
		}
	}
	dirs = append(dirs, filepath.Join(repoRoot, "target", "release"))
	for _, dir := range dirs {
		for _, name := range cdylibNames {
			p := filepath.Join(dir, name)
			if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
				return p, nil
			}
		}
	}
	return "", fmt.Errorf(
		"pithaudio: no cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR and <repo>/target/release); run `cargo build --release` first",
	)
}

// locate resolves the cdylib path once per process.
var locate = sync.OnceValues(FindCdylib)

// Peak is one spectral landmark: analysis frame t, frequency slot f.
type Peak struct {
	// T is the analysis frame (HOP = 2048 samples each).
	T uint32
	// F is the frequency slot (0..58).
	F uint16
}

// DecodeStream is one decoded WAV, re-expressed from the decode
// stream: the recorded container facts plus the samples.
type DecodeStream struct {
	// SampleRate is the rate the file declared.
	SampleRate uint32
	// Channels is the channel count the file declared.
	Channels uint16
	// BitsPerSample is the sample width the file declared
	// (8/16/24/32/64).
	BitsPerSample uint16
	// Frames is the whole-frame count of the payload.
	Frames uint32
	// DecodedSamples is the interleaved sample count (Frames*Channels).
	DecodedSamples uint32
	// Samples are the decoded i32 values in file order — exactly the
	// bytes the pcm_i32_le_* digests cover.
	Samples []int32
}

// SignatureStream is one extracted signature, re-expressed from the
// signature stream.
type SignatureStream struct {
	// Frames is the analysis-frame count the signature covers.
	Frames uint32
	// PeakCount is the number of spectral peaks.
	PeakCount uint32
	// Fingerprint is the presence-mask fingerprint.
	Fingerprint uint64
	// Peaks is the sorted (t, f) peak list.
	Peaks []Peak
}

// MatchEntry is one corpus member's hit.
type MatchEntry struct {
	// ID is the matched signature's position in the index (insertion
	// order).
	ID uint32
	// DeltaT is the modal query_t − stored_t in analysis frames.
	DeltaT int64
	// Votes is the vote mass within the tolerance window around DeltaT.
	Votes uint32
	// TotalVotes is the total raw votes the id received across all
	// offsets.
	TotalVotes uint32
}

// MatchStream is one match run, re-expressed from the match stream.
type MatchStream struct {
	// QueryFrames is the analysis-frame count of the query signature.
	QueryFrames uint32
	// QueryPeakCount is the peak count of the query signature.
	QueryPeakCount uint32
	// Matches is one entry per matched id, best first.
	Matches []MatchEntry
}

// ParseDecodeStream re-expresses the decode stream. Layout: sample_rate
// u32 BE, channels u16 BE, bits_per_sample u16 BE, frames u32 BE,
// decoded_samples u32 BE, then the samples i32 LE — so
// sha256(stream[16:]) is the pcm_i32_le_sha256 digest.
func ParseDecodeStream(raw []byte) (DecodeStream, error) {
	if len(raw) < 16 {
		return DecodeStream{}, fmt.Errorf("pithaudio: decode stream is shorter than the 16-byte header")
	}
	decoded := binary.BigEndian.Uint32(raw[12:16])
	if len(raw) != 16+int(decoded)*4 {
		return DecodeStream{}, fmt.Errorf("pithaudio: decode stream body does not match decoded_samples")
	}
	samples := make([]int32, decoded)
	for i := range samples {
		samples[i] = int32(binary.LittleEndian.Uint32(raw[16+i*4:]))
	}
	return DecodeStream{
		SampleRate:     binary.BigEndian.Uint32(raw[0:4]),
		Channels:       binary.BigEndian.Uint16(raw[4:6]),
		BitsPerSample:  binary.BigEndian.Uint16(raw[6:8]),
		Frames:         binary.BigEndian.Uint32(raw[8:12]),
		DecodedSamples: decoded,
		Samples:        samples,
	}, nil
}

// ParseSignatureStream re-expresses the signature stream. Layout:
// frames u32 BE, peak_count u32 BE, fingerprint u64 BE, then per peak
// t u32 BE, f u16 BE. (peaks_sha256 hashes the LITTLE-endian
// re-serialization of the peaks — t u32 LE, f u16 LE.)
func ParseSignatureStream(raw []byte) (SignatureStream, error) {
	if len(raw) < 16 {
		return SignatureStream{}, fmt.Errorf("pithaudio: signature stream is shorter than its 16-byte header")
	}
	peakCount := binary.BigEndian.Uint32(raw[4:8])
	if len(raw) != 16+int(peakCount)*6 {
		return SignatureStream{}, fmt.Errorf("pithaudio: signature stream body does not match peak_count")
	}
	peaks := make([]Peak, peakCount)
	for i := range peaks {
		peaks[i] = Peak{
			T: binary.BigEndian.Uint32(raw[16+i*6:]),
			F: binary.BigEndian.Uint16(raw[16+i*6+4:]),
		}
	}
	return SignatureStream{
		Frames:      binary.BigEndian.Uint32(raw[0:4]),
		PeakCount:   peakCount,
		Fingerprint: binary.BigEndian.Uint64(raw[8:16]),
		Peaks:       peaks,
	}, nil
}

// ParseMatchStream re-expresses the match stream. Layout: query_frames
// u32 BE, query_peak_count u32 BE, match_count u32 BE, then per match
// id u32 BE, delta_t i64 BE, votes u32 BE, total_votes u32 BE.
func ParseMatchStream(raw []byte) (MatchStream, error) {
	if len(raw) < 12 {
		return MatchStream{}, fmt.Errorf("pithaudio: match stream is shorter than the 12-byte header")
	}
	matchCount := binary.BigEndian.Uint32(raw[8:12])
	if len(raw) != 12+int(matchCount)*20 {
		return MatchStream{}, fmt.Errorf("pithaudio: match stream body does not match match_count")
	}
	matches := make([]MatchEntry, matchCount)
	for i := range matches {
		at := 12 + i*20
		matches[i] = MatchEntry{
			ID:         binary.BigEndian.Uint32(raw[at:]),
			DeltaT:     int64(binary.BigEndian.Uint64(raw[at+4:])),
			Votes:      binary.BigEndian.Uint32(raw[at+12:]),
			TotalVotes: binary.BigEndian.Uint32(raw[at+16:]),
		}
	}
	return MatchStream{
		QueryFrames:    binary.BigEndian.Uint32(raw[0:4]),
		QueryPeakCount: binary.BigEndian.Uint32(raw[4:8]),
		Matches:        matches,
	}, nil
}

// copyOut moves a handed-out cdylib buffer into a Go copy; the caller
// releases the original with pith_audio_free before returning (null is
// accepted and ignored by the cdylib, matching the C contract).
func copyOut(status int32, out *byte, outLen uintptr, op string) ([]byte, error) {
	if status != StatusOK {
		return nil, &FfiError{Op: op, Status: status}
	}
	buf := make([]byte, outLen)
	if outLen > 0 {
		copy(buf, unsafe.Slice(out, outLen))
	}
	if libPath, err := locate(); err == nil {
		ffiFree(libPath, out, outLen)
	}
	return buf, nil
}

// dataPtr returns a pointer to data's first byte, nil-safe for empty
// slices (the FFI distinguishes null from a zero-length buffer).
func dataPtr(data []byte) *byte {
	if len(data) == 0 {
		return nil
	}
	return &data[0]
}

// DecodeWav decodes a complete RIFF/WAVE stream into the decode stream
// the pcm_i32_le_sha256 / pcm_i32_le_fnv1a64 digests are defined over.
func DecodeWav(data []byte) ([]byte, error) {
	libPath, err := locate()
	if err != nil {
		return nil, err
	}
	var out *byte
	var outLen uintptr
	status, err := ffiDecodeWav(libPath, dataPtr(data), len(data), &out, &outLen)
	if err != nil {
		return nil, err
	}
	return copyOut(status, out, outLen, "pith_audio_decode_wav")
}

// SignatureWav decodes a RIFF/WAVE stream and extracts its signature
// (the signature_of_wav facade: anything but 44100 Hz is refused).
func SignatureWav(data []byte) ([]byte, error) {
	libPath, err := locate()
	if err != nil {
		return nil, err
	}
	var out *byte
	var outLen uintptr
	status, err := ffiSignatureWav(libPath, dataPtr(data), len(data), &out, &outLen)
	if err != nil {
		return nil, err
	}
	return copyOut(status, out, outLen, "pith_audio_signature_wav")
}

// SignaturePcm extracts a signature from interleaved little-endian
// i32 PCM at 44100 Hz into the signature stream.
func SignaturePcm(data []byte, channels uint32) ([]byte, error) {
	if channels == 0 || channels > 0xffff {
		return nil, &FfiError{Op: "pith_audio_signature_pcm", Status: StatusInvalid}
	}
	libPath, err := locate()
	if err != nil {
		return nil, err
	}
	var out *byte
	var outLen uintptr
	status, err := ffiSignaturePcm(libPath, dataPtr(data), len(data), channels, &out, &outLen)
	if err != nil {
		return nil, err
	}
	return copyOut(status, out, outLen, "pith_audio_signature_pcm")
}

// VoicesPcm synthesizes the SplitMix64 square-voice corpus inside the
// cdylib: nSamples interleaved i32 LE samples, exactly the bytes the
// synthetic reference.json vectors are built from. The recipe is pure
// integer/exact-float math, so no SDK ever reimplements it and no
// platform sensitivity enters the vectors.
func VoicesPcm(seed uint64, nSamples int) ([]byte, error) {
	libPath, err := locate()
	if err != nil {
		return nil, err
	}
	var out *byte
	var outLen uintptr
	status, err := ffiVoicesPcm(libPath, seed, nSamples, &out, &outLen)
	if err != nil {
		return nil, err
	}
	return copyOut(status, out, outLen, "pith_audio_voices_pcm")
}

// Index is an incremental signature index: PCM signatures accumulate
// in insertion order, and the id a match reports is that order. A
// failed Add leaves the index alive and unchanged.
type Index struct {
	h cIndexHandle
}

// NewIndex creates an empty signature index. Close releases it.
func NewIndex() (*Index, error) {
	libPath, err := locate()
	if err != nil {
		return nil, err
	}
	h, err := ffiIndexNew(libPath)
	if err != nil {
		return nil, err
	}
	return &Index{h: h}, nil
}

// Add appends the signature of interleaved little-endian i32 PCM to
// the index.
func (ix *Index) Add(data []byte, channels uint32) error {
	if ix == nil || !ix.h.valid() {
		return &FfiError{Op: "pith_audio_index_add", Status: StatusInvalid}
	}
	if channels == 0 || channels > 0xffff {
		return &FfiError{Op: "pith_audio_index_add", Status: StatusInvalid}
	}
	libPath, err := locate()
	if err != nil {
		return err
	}
	status, err := ffiIndexAdd(libPath, ix.h, dataPtr(data), len(data), channels)
	if err != nil {
		return err
	}
	if status != StatusOK {
		return &FfiError{Op: "pith_audio_index_add", Status: status}
	}
	return nil
}

// Match matches the signature of interleaved little-endian i32 PCM
// against the index and returns the match stream. An empty index is
// legal and yields zero matches.
func (ix *Index) Match(data []byte, channels uint32) ([]byte, error) {
	if ix == nil || !ix.h.valid() {
		return nil, &FfiError{Op: "pith_audio_match", Status: StatusInvalid}
	}
	if channels == 0 || channels > 0xffff {
		return nil, &FfiError{Op: "pith_audio_match", Status: StatusInvalid}
	}
	libPath, err := locate()
	if err != nil {
		return nil, err
	}
	var out *byte
	var outLen uintptr
	status, err := ffiIndexMatch(libPath, ix.h, dataPtr(data), len(data), channels, &out, &outLen)
	if err != nil {
		return nil, err
	}
	return copyOut(status, out, outLen, "pith_audio_match")
}

// Close releases the index handle; idempotent.
func (ix *Index) Close() {
	if ix == nil || !ix.h.valid() {
		return
	}
	if libPath, err := locate(); err == nil {
		ffiIndexFree(libPath, ix.h)
	}
	ix.h.invalidate()
}

// MonoAverage is the explicit mono average of an interleaved stream —
// mirrors reference::mono_average: the per-frame sum accumulates in
// i64 and divides by the channel count truncating toward zero (Rust
// integer division semantics).
func MonoAverage(interleaved []int32, channels int) []int32 {
	out := make([]int32, len(interleaved)/channels)
	for i := range out {
		var acc int64
		for c := 0; c < channels; c++ {
			acc += int64(interleaved[i*channels+c])
		}
		out[i] = int32(acc / int64(channels))
	}
	return out
}

// Interleave interleaves two i32 sample slices into one stereo slice.
func Interleave(left, right []int32) []int32 {
	out := make([]int32, len(left)+len(right))
	for i := range left {
		out[2*i] = left[i]
		out[2*i+1] = right[i]
	}
	return out
}
