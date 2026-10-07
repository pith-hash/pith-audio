// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

package pithaudio

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// repoRoot resolves the repository root relative to this package
// (sdk/go -> sdk -> repo root), the anchor for reference.json and the
// committed fixture.
func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if st, err := os.Stat(filepath.Join(root, "reference.json")); err != nil || st.IsDir() {
		t.Fatalf("reference.json not found at %s", root)
	}
	return root
}

// reference parses the committed reference.json into named vectors.
func reference(t *testing.T) (map[string]map[string]any, map[string]map[string]any) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var parsed struct {
		Vectors      []map[string]any `json:"vectors"`
		ErrorVectors []map[string]any `json:"error_vectors"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatal(err)
	}
	vectors := make(map[string]map[string]any, len(parsed.Vectors))
	for _, v := range parsed.Vectors {
		vectors[v["name"].(string)] = v
	}
	errors := make(map[string]map[string]any, len(parsed.ErrorVectors))
	for _, v := range parsed.ErrorVectors {
		errors[v["name"].(string)] = v
	}
	return vectors, errors
}

// num reads a JSON number field as int64.
func num(v map[string]any, key string) int64 {
	return int64(v[key].(float64))
}

// seedParam reads a "0x…" seed string field.
func seedParam(s string) uint64 {
	n, err := strconv.ParseUint(s, 0, 64)
	if err != nil {
		panic(err)
	}
	return n
}

// fnv1a64 is the standard FNV-1a 64 — the digest pcm_i32_le_fnv1a64
// pins.
func fnv1a64(data []byte) uint64 {
	h := uint64(0xcbf29ce484222325)
	for _, b := range data {
		h ^= uint64(b)
		h *= 0x100000001b3
	}
	return h
}

// peaksLE serializes peaks the way peaks_sha256 hashes them: t u32 LE,
// f u16 LE (the FFI stream carries them big-endian; the digest is over
// the LE form).
func peaksLE(peaks []Peak) []byte {
	out := make([]byte, 0, len(peaks)*6)
	for _, p := range peaks {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], p.T)
		out = append(out, b[:]...)
		var f [2]byte
		binary.LittleEndian.PutUint16(f[:], p.F)
		out = append(out, f[:]...)
	}
	return out
}

// assertSignatureHexExact compares one signature stream against its
// committed vector.
func assertSignatureHexExact(t *testing.T, name string, raw []byte) {
	t.Helper()
	vectors, _ := reference(t)
	vector := vectors[name]
	facts, err := ParseSignatureStream(raw)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if facts.Frames != uint32(num(vector, "frames")) {
		t.Errorf("%s: frames %d, want %d", name, facts.Frames, num(vector, "frames"))
	}
	if facts.PeakCount != uint32(num(vector, "peak_count")) {
		t.Errorf("%s: peak_count %d, want %d", name, facts.PeakCount, num(vector, "peak_count"))
	}
	if len(facts.Peaks) != len(vector["peaks"].([]any)) {
		t.Fatalf("%s: peaks %d, want %d", name, len(facts.Peaks), len(vector["peaks"].([]any)))
	}
	for i, p := range facts.Peaks {
		entry := vector["peaks"].([]any)[i].([]any)
		if p.T != uint32(entry[0].(float64)) || p.F != uint16(entry[1].(float64)) {
			t.Errorf("%s: peak %d = [%d %d], want [%v %v]", name, i, p.T, p.F, entry[0], entry[1])
		}
	}
	wantFP, _ := strconv.ParseUint(vector["fingerprint"].(string), 16, 64)
	if facts.Fingerprint != wantFP {
		t.Errorf("%s: fingerprint %016x, want %016x", name, facts.Fingerprint, wantFP)
	}
	digest := sha256.Sum256(peaksLE(facts.Peaks))
	if got := hex.EncodeToString(digest[:]); got != vector["peaks_sha256"].(string) {
		t.Errorf("%s: peaks_sha256 %s, want %s", name, got, vector["peaks_sha256"].(string))
	}
}

// TestDecodeVectorsHexExact replays every decode vector: the recorded
// container facts plus both digests over the decode-stream body.
func TestDecodeVectorsHexExact(t *testing.T) {
	vectors, _ := reference(t)
	for _, name := range []string{
		"fixture-tone-wav",
		"pcm8-offset-centred",
		"pcm24-sign-extension",
		"float32-clamp-and-nan",
		"float64-halves-and-thirds",
		"riff-odd-chunk-pad",
		"partial-frame-dropped",
	} {
		t.Run(name, func(t *testing.T) {
			vector := vectors[name]
			var data []byte
			if vector["input_kind"] == "fixture-file" {
				b, err := os.ReadFile(filepath.Join(repoRoot(t), "tests", "fixtures", "tone.wav"))
				if err != nil {
					t.Fatal(err)
				}
				data = b
			} else {
				b, err := hex.DecodeString(vector["input_hex"].(string))
				if err != nil {
					t.Fatal(err)
				}
				data = b
			}
			raw, err := DecodeWav(data)
			if err != nil {
				t.Fatalf("DecodeWav(%s): %v", name, err)
			}
			facts, err := ParseDecodeStream(raw)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if facts.SampleRate != uint32(num(vector, "sample_rate")) {
				t.Errorf("%s: sample_rate %d, want %d", name, facts.SampleRate, num(vector, "sample_rate"))
			}
			if facts.Channels != uint16(num(vector, "channels")) {
				t.Errorf("%s: channels %d, want %d", name, facts.Channels, num(vector, "channels"))
			}
			if facts.BitsPerSample != uint16(num(vector, "bits_per_sample")) {
				t.Errorf("%s: bits_per_sample %d, want %d", name, facts.BitsPerSample, num(vector, "bits_per_sample"))
			}
			if facts.Frames != uint32(num(vector, "frames")) {
				t.Errorf("%s: frames %d, want %d", name, facts.Frames, num(vector, "frames"))
			}
			if facts.DecodedSamples != uint32(num(vector, "decoded_samples")) {
				t.Errorf("%s: decoded_samples %d, want %d", name, facts.DecodedSamples, num(vector, "decoded_samples"))
			}
			if samples, ok := vector["samples_i32"].([]any); ok {
				if len(facts.Samples) != len(samples) {
					t.Fatalf("%s: samples %d, want %d", name, len(facts.Samples), len(samples))
				}
				for i, s := range facts.Samples {
					if s != int32(samples[i].(float64)) {
						t.Errorf("%s: sample %d = %d, want %v", name, i, s, samples[i])
					}
				}
			}
			body := raw[16:]
			digest := sha256.Sum256(body)
			if got := hex.EncodeToString(digest[:]); got != vector["pcm_i32_le_sha256"].(string) {
				t.Errorf("%s: pcm_i32_le_sha256 %s, want %s", name, got, vector["pcm_i32_le_sha256"].(string))
			}
			if got := fmt.Sprintf("%016x", fnv1a64(body)); got != vector["pcm_i32_le_fnv1a64"].(string) {
				t.Errorf("%s: pcm_i32_le_fnv1a64 %s, want %s", name, got, vector["pcm_i32_le_fnv1a64"].(string))
			}
		})
	}
}

// TestSignatureVectorsHexExact replays the fixture signature vector
// and the three synthetic-voices signature vectors (corpora synthesized
// inside the cdylib via VoicesPcm; the stereo pair interleave + the
// explicit mono average are the only SDK-side mixdown steps).
func TestSignatureVectorsHexExact(t *testing.T) {
	t.Run("fixture-tone-signature", func(t *testing.T) {
		wav, err := os.ReadFile(filepath.Join(repoRoot(t), "tests", "fixtures", "tone.wav"))
		if err != nil {
			t.Fatal(err)
		}
		raw, err := SignatureWav(wav)
		if err != nil {
			t.Fatal(err)
		}
		assertSignatureHexExact(t, "fixture-tone-signature", raw)
	})

	t.Run("voices-signature-mono", func(t *testing.T) {
		pcm, err := VoicesPcm(0x00c0ffee, 66150)
		if err != nil {
			t.Fatal(err)
		}
		raw, err := SignaturePcm(pcm, 1)
		if err != nil {
			t.Fatal(err)
		}
		assertSignatureHexExact(t, "voices-signature-mono", raw)
	})

	stereo := func() []int32 {
		left, err := VoicesPcm(0x00000011, 88200)
		if err != nil {
			t.Fatal(err)
		}
		right, err := VoicesPcm(0x00000022, 88200)
		if err != nil {
			t.Fatal(err)
		}
		return Interleave(leI32(t, left), leI32(t, right))
	}()

	t.Run("voices-stereo-mixdown", func(t *testing.T) {
		raw, err := SignaturePcm(leBytes(t, stereo), 2)
		if err != nil {
			t.Fatal(err)
		}
		assertSignatureHexExact(t, "voices-stereo-mixdown", raw)
	})

	t.Run("voices-stereo-mono-average", func(t *testing.T) {
		mono := MonoAverage(stereo, 2)
		raw, err := SignaturePcm(leBytes(t, mono), 1)
		if err != nil {
			t.Fatal(err)
		}
		assertSignatureHexExact(t, "voices-stereo-mono-average", raw)
	})
}

// TestOffsetMatchHexExact replays the committed offset-match vector
// through the index triad.
func TestOffsetMatchHexExact(t *testing.T) {
	vectors, _ := reference(t)
	vector := vectors["voices-offset-match"]

	prefix, err := VoicesPcm(0x0000beef, 88200)
	if err != nil {
		t.Fatal(err)
	}
	content, err := VoicesPcm(0x0000d1ce, 132300)
	if err != nil {
		t.Fatal(err)
	}
	query := append(append([]byte{}, prefix...), content...)

	index, err := NewIndex()
	if err != nil {
		t.Fatal(err)
	}
	defer index.Close()
	for _, member := range []struct {
		seed uint64
		n    int
	}{
		{0x0000d1ce, 132300},
		{0x0000f00d, 132300},
		{0x0000e5a5, 44100},
	} {
		pcm, err := VoicesPcm(member.seed, member.n)
		if err != nil {
			t.Fatal(err)
		}
		if err := index.Add(pcm, 1); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := index.Match(query, 1)
	if err != nil {
		t.Fatal(err)
	}
	table, err := ParseMatchStream(raw)
	if err != nil {
		t.Fatal(err)
	}
	if table.QueryFrames != uint32(num(vector, "query_frames")) {
		t.Errorf("query_frames %d, want %d", table.QueryFrames, num(vector, "query_frames"))
	}
	if table.QueryPeakCount != uint32(num(vector, "query_peak_count")) {
		t.Errorf("query_peak_count %d, want %d", table.QueryPeakCount, num(vector, "query_peak_count"))
	}
	want := vector["matches"].([]any)
	if len(table.Matches) != len(want) {
		t.Fatalf("matches %d, want %d", len(table.Matches), len(want))
	}
	for i, m := range table.Matches {
		entry := want[i].(map[string]any)
		if m.ID != uint32(entry["id"].(float64)) || m.DeltaT != int64(entry["delta_t"].(float64)) ||
			m.Votes != uint32(entry["votes"].(float64)) || m.TotalVotes != uint32(entry["total_votes"].(float64)) {
			t.Errorf("match %d = %+v, want %v", i, m, entry)
		}
	}
	if table.Matches[0].ID != 0 {
		t.Errorf("matches[0].id = %d, want 0 (the content member must win)", table.Matches[0].ID)
	}
}

// TestPinnedFixtureSignature pins one signature the Rust unit tests
// re-derive, so the binding fails loudly even if reference.json were
// regenerated wrongly.
func TestPinnedFixtureSignature(t *testing.T) {
	wav, err := os.ReadFile(filepath.Join(repoRoot(t), "tests", "fixtures", "tone.wav"))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := SignatureWav(wav)
	if err != nil {
		t.Fatal(err)
	}
	facts, err := ParseSignatureStream(raw)
	if err != nil {
		t.Fatal(err)
	}
	if facts.Frames != 20 || facts.PeakCount != 42 {
		t.Fatalf("frames/peaks = %d/%d, want 20/42", facts.Frames, facts.PeakCount)
	}
	const wantFP = uint64(0x0400000002001425)
	if facts.Fingerprint != wantFP {
		t.Errorf("fingerprint %016x, want %016x", facts.Fingerprint, wantFP)
	}
	digest := sha256.Sum256(peaksLE(facts.Peaks))
	const wantDigest = "8c6ac0719bb7caed61e66f978da85ab5b66639aa279457d8f88731be2dea1ec8"
	if got := hex.EncodeToString(digest[:]); got != wantDigest {
		t.Errorf("peaks_sha256 %s, want %s", got, wantDigest)
	}
}

// TestRefusalsAndHandleLifetime checks the refusal paths (malformed
// WAV, PCM geometry, null handle) and that a failed Add leaves the
// index alive.
func TestRefusalsAndHandleLifetime(t *testing.T) {
	_, errors := reference(t)
	for _, name := range []string{"short-riff-header", "not-a-wav", "adpcm-format-tag", "wrong-sample-rate-facade"} {
		t.Run(name, func(t *testing.T) {
			vector := errors[name]
			data, err := hex.DecodeString(vector["input_hex"].(string))
			if err != nil {
				t.Fatal(err)
			}
			var status int32
			if vector["api"] == "wav.decode" {
				_, err = DecodeWav(data)
			} else {
				_, err = SignatureWav(data)
			}
			var ffiErr *FfiError
			if err == nil {
				t.Fatalf("%s: expected refusal", name)
			} else if asFfi, ok := err.(*FfiError); ok {
				ffiErr = asFfi
				status = ffiErr.Status
			} else {
				t.Fatalf("%s: unexpected error %v", name, err)
			}
			_ = status
			if ffiErr.Status != StatusRejected {
				t.Errorf("%s: status %d, want %d", name, ffiErr.Status, StatusRejected)
			}
		})
	}

	t.Run("pcm-geometry", func(t *testing.T) {
		if _, err := SignaturePcm([]byte{1, 0, 0}, 1); err == nil {
			t.Fatal("partial sample: expected refusal")
		} else if ffiErr, ok := err.(*FfiError); !ok || ffiErr.Status != StatusInvalid {
			t.Fatalf("partial sample: %v", err)
		}
		if _, err := SignaturePcm([]byte{0, 0, 0, 0}, 0); err == nil {
			t.Fatal("zero channels: expected refusal")
		} else if ffiErr, ok := err.(*FfiError); !ok || ffiErr.Status != StatusInvalid {
			t.Fatalf("zero channels: %v", err)
		}
	})

	t.Run("index-lifetime", func(t *testing.T) {
		index, err := NewIndex()
		if err != nil {
			t.Fatal(err)
		}
		pcm, err := VoicesPcm(0x00c0ffee, 66150)
		if err != nil {
			t.Fatal(err)
		}
		if err := index.Add(pcm, 0); err == nil {
			t.Fatal("zero channels: expected refusal")
		}
		// The handle survived: a real corpus lands and the match runs.
		if err := index.Add(pcm, 1); err != nil {
			t.Fatal(err)
		}
		raw, err := index.Match(pcm, 1)
		if err != nil {
			t.Fatal(err)
		}
		table, err := ParseMatchStream(raw)
		if err != nil {
			t.Fatal(err)
		}
		if len(table.Matches) == 0 || table.Matches[0].ID != 0 {
			t.Fatalf("matches = %+v, want member 0 first", table.Matches)
		}
		index.Close()
		index.Close() // idempotent
	})
}

// leI32 decodes an i32-LE PCM buffer into samples.
func leI32(t *testing.T, buf []byte) []int32 {
	t.Helper()
	out := make([]int32, len(buf)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return out
}

// leBytes encodes samples as i32-LE PCM.
func leBytes(t *testing.T, s []int32) []byte {
	t.Helper()
	out := make([]byte, len(s)*4)
	for i, v := range s {
		binary.LittleEndian.PutUint32(out[i*4:], uint32(v))
	}
	return out
}
