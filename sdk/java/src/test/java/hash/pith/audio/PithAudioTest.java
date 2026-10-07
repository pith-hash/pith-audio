// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
package hash.pith.audio;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Hex-exact conformance: the committed reference vectors, replayed
 * through the Java JNI surface of the pith-audio cdylib. The synthetic
 * corpora are rebuilt via {@code voicesPcm} (synthesis runs inside the
 * cdylib — no SDK-side float reimplementation).
 */
class PithAudioTest {
    private static final Path REPO_ROOT = findRepoRoot();
    private static final Map<String, JsonNode> VECTORS = new LinkedHashMap<>();
    private static final List<JsonNode> ERROR_VECTORS = new ArrayList<>();

    static {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(REPO_ROOT.resolve("reference.json").toFile());
            for (JsonNode v : root.get("vectors")) {
                VECTORS.put(v.get("name").asText(), v);
            }
            for (JsonNode v : root.get("error_vectors")) {
                ERROR_VECTORS.add(v);
            }
        } catch (Exception e) {
            throw new IllegalStateException("reference.json", e);
        }
    }

    /** Every decode vector: facts, samples, and body digests. */
    @org.junit.jupiter.api.Test
    void every_decode_vector_is_reproduced_hex_exact() throws Exception {
        String[] names = {"fixture-tone-wav", "pcm8-offset-centred", "pcm24-sign-extension",
            "float32-clamp-and-nan", "float64-halves-and-thirds", "riff-odd-chunk-pad",
            "partial-frame-dropped"};
        for (String name : names) {
            JsonNode v = VECTORS.get(name);
            byte[] stream = PithAudio.decodeWav(vectorBytes(v));
            DecodeFacts facts = DecodeFacts.parse(stream);
            assertEquals(v.get("sample_rate").asInt(), facts.sampleRate, name + " sample_rate");
            assertEquals(v.get("channels").asInt(), facts.channels, name + " channels");
            assertEquals(v.get("bits_per_sample").asInt(), facts.bitsPerSample, name + " bits");
            assertEquals(v.get("frames").asInt(), facts.frames, name + " frames");
            assertEquals(v.get("decoded_samples").asInt(), facts.sampleCount, name + " samples");
            if (v.has("samples_i32")) {
                JsonNode expected = v.get("samples_i32");
                assertEquals(expected.size(), facts.samples.length, name + " sample count");
                for (int i = 0; i < facts.samples.length; i++) {
                    assertEquals(expected.get(i).asInt(), facts.samples[i], name + " sample[" + i + "]");
                }
            }
            assertEquals(v.get("pcm_i32_le_sha256").asText(), sha256(body(stream)), name + " sha256");
            assertEquals(v.get("pcm_i32_le_fnv1a64").asText(), hex64(fnv1a64(body(stream))), name + " fnv1a64");
        }
        assertEquals(7, names.length, "decode vectors checked");
    }

    /** Every signature vector, including the stereo pair via both mixdown paths. */
    @org.junit.jupiter.api.Test
    void every_signature_vector_is_reproduced_hex_exact() throws Exception {
        String[] names = {"fixture-tone-signature", "voices-signature-mono",
            "voices-stereo-mixdown", "voices-stereo-mono-average"};
        for (String name : names) {
            JsonNode v = VECTORS.get(name);
            byte[] stream;
            if ("fixture-tone-signature".equals(name)) {
                stream = PithAudio.signatureWav(vectorBytes(v));
            } else if ("voices-signature-mono".equals(name)) {
                stream = PithAudio.signaturePcm(PithAudio.voicesPcm(0x00C0FFEEL, 66_150), 1);
            } else {
                int[] stereo = interleave(
                    i32le(PithAudio.voicesPcm(0x00000011L, 88_200)),
                    i32le(PithAudio.voicesPcm(0x00000022L, 88_200)));
                if ("voices-stereo-mixdown".equals(name)) {
                    stream = PithAudio.signaturePcm(i32leBytes(stereo), 2);
                } else {
                    int[] mono = monoAverage(stereo, 2);
                    stream = PithAudio.signaturePcm(i32leBytes(mono), 1);
                }
            }
            assertSignature(name, v, stream);
        }
        assertEquals(4, names.length, "signature vectors checked");
    }

    /** The offset-match vector: the full match table through AudioIndex. */
    @org.junit.jupiter.api.Test
    void offset_match_vector_is_reproduced_hex_exact() {
        JsonNode v = VECTORS.get("voices-offset-match");
        byte[] query = concat(
            PithAudio.voicesPcm(0x0000BEEFL, 88_200),
            PithAudio.voicesPcm(0x0000D1CEL, 132_300));
        try (PithAudio.AudioIndex index = new PithAudio.AudioIndex()) {
            index.add(PithAudio.voicesPcm(0x0000D1CEL, 132_300), 1);
            index.add(PithAudio.voicesPcm(0x0000F00DL, 132_300), 1);
            index.add(PithAudio.voicesPcm(0x0000E5A5L, 44_100), 1);
            MatchTable table = MatchTable.parse(index.match(query, 1));

            assertEquals(v.get("query_frames").asInt(), table.queryFrames, "query_frames");
            assertEquals(v.get("query_peak_count").asInt(), table.queryPeakCount, "query_peak_count");
            JsonNode expected = v.get("matches");
            assertEquals(expected.size(), table.matches.size(), "match count");
            for (int i = 0; i < table.matches.size(); i++) {
                JsonNode m = expected.get(i);
                MatchTable.Row row = table.matches.get(i);
                assertEquals(m.get("id").asInt(), row.id, "match[" + i + "] id");
                assertEquals(m.get("delta_t").asLong(), row.deltaT, "match[" + i + "] delta_t");
                assertEquals(m.get("votes").asInt(), row.votes, "match[" + i + "] votes");
                assertEquals(m.get("total_votes").asInt(), row.totalVotes, "match[" + i + "] total_votes");
            }
            assertEquals(0, table.matches.get(0).id, "the content member must win");
        }
    }

    /** One rust-derived literal pin, byte-independent of reference.json. */
    @org.junit.jupiter.api.Test
    void tone_signature_matches_a_rust_pinned_value() throws Exception {
        byte[] wav = Files.readAllBytes(REPO_ROOT.resolve(Paths.get("tests", "fixtures", "tone.wav")));
        SignatureFacts facts = SignatureFacts.parse(PithAudio.signatureWav(wav));
        assertEquals(20, facts.frames);
        assertEquals(42, facts.peakCount);
        assertEquals("0400000002001425", hex64(facts.fingerprint));
        assertEquals("8c6ac0719bb7caed61e66f978da85ab5b66639aa279457d8f88731be2dea1ec8",
            sha256(peaksLeBytes(facts.peaks)));
    }

    /** Every decode/signature refusal is FfiError(-2), never a crash. */
    @org.junit.jupiter.api.Test
    void error_vectors_are_refused_not_crashing() {
        assertEquals(11, ERROR_VECTORS.size(), "error vectors present");
        for (JsonNode v : ERROR_VECTORS) {
            byte[] data = hex(v.get("input_hex").asText());
            PithAudio.FfiError err = assertThrows(PithAudio.FfiError.class, () -> {
                if ("wav.decode".equals(v.get("api").asText())) {
                    PithAudio.decodeWav(data);
                } else {
                    PithAudio.signatureWav(data);
                }
            }, v.get("name").asText());
            assertEquals(-2, err.status, v.get("name").asText());
        }
    }

    /** Malformed PCM arguments are invalid (-1), never crashes. */
    @org.junit.jupiter.api.Test
    void bad_pcm_arguments_are_invalid() {
        PithAudio.FfiError partial = assertThrows(PithAudio.FfiError.class,
            () -> PithAudio.signaturePcm(new byte[] {1, 0, 0}, 1));
        assertEquals(-1, partial.status);
        PithAudio.FfiError zeroChannels = assertThrows(PithAudio.FfiError.class,
            () -> PithAudio.signaturePcm(new byte[] {0, 0, 0, 0}, 0));
        assertEquals(-1, zeroChannels.status);
    }

    /** The cdylib resolves without environment hints. */
    @org.junit.jupiter.api.Test
    void cdylib_is_discoverable() {
        assertTrue(Paths.get(PithAudio.cdylibPath()).isAbsolute());
    }

    private static void assertSignature(String name, JsonNode v, byte[] stream) throws Exception {
        SignatureFacts facts = SignatureFacts.parse(stream);
        assertEquals(v.get("frames").asInt(), facts.frames, name + " frames");
        assertEquals(v.get("peak_count").asInt(), facts.peakCount, name + " peak_count");
        JsonNode peaks = v.get("peaks");
        assertEquals(peaks.size(), facts.peaks.length, name + " peaks size");
        for (int i = 0; i < facts.peaks.length; i++) {
            assertEquals(peaks.get(i).get(0).asInt(), facts.peaks[i][0], name + " peak[" + i + "].t");
            assertEquals(peaks.get(i).get(1).asInt(), facts.peaks[i][1], name + " peak[" + i + "].f");
        }
        assertEquals(v.get("fingerprint").asText(), hex64(facts.fingerprint), name + " fingerprint");
        assertEquals(v.get("peaks_sha256").asText(), sha256(peaksLeBytes(facts.peaks)), name + " peaks_sha256");
    }

    private static byte[] vectorBytes(JsonNode v) throws Exception {
        if ("fixture-file".equals(v.get("input_kind").asText())) {
            return Files.readAllBytes(REPO_ROOT.resolve(Paths.get("tests", "fixtures", "tone.wav")));
        }
        return hex(v.get("input_hex").asText());
    }

    private static int[] interleave(int[] left, int[] right) {
        int[] out = new int[left.length + right.length];
        for (int i = 0, j = 0; i < left.length; i++) {
            out[j++] = left[i];
            out[j++] = right[i];
        }
        return out;
    }

    private static int[] monoAverage(int[] interleaved, int channels) {
        int frames = interleaved.length / channels;
        int[] out = new int[frames];
        for (int i = 0; i < frames; i++) {
            long acc = 0;
            for (int c = 0; c < channels; c++) {
                acc += interleaved[i * channels + c];
            }
            out[i] = (int) (acc / channels);
        }
        return out;
    }

    private static int[] i32le(byte[] stream) {
        int[] out = new int[stream.length / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = i32le(stream, i * 4);
        }
        return out;
    }

    private static byte[] i32leBytes(int[] samples) {
        byte[] out = new byte[samples.length * 4];
        for (int i = 0; i < samples.length; i++) {
            int s = samples[i];
            out[i * 4] = (byte) s;
            out[i * 4 + 1] = (byte) (s >>> 8);
            out[i * 4 + 2] = (byte) (s >>> 16);
            out[i * 4 + 3] = (byte) (s >>> 24);
        }
        return out;
    }

    private static byte[] body(byte[] stream) {
        byte[] out = new byte[stream.length - 16];
        System.arraycopy(stream, 16, out, 0, out.length);
        return out;
    }

    private static byte[] peaksLeBytes(int[][] peaks) {
        byte[] out = new byte[peaks.length * 6];
        for (int i = 0; i < peaks.length; i++) {
            int t = peaks[i][0];
            int f = peaks[i][1];
            out[i * 6] = (byte) t;
            out[i * 6 + 1] = (byte) (t >>> 8);
            out[i * 6 + 2] = (byte) (t >>> 16);
            out[i * 6 + 3] = (byte) (t >>> 24);
            out[i * 6 + 4] = (byte) f;
            out[i * 6 + 5] = (byte) (f >>> 8);
        }
        return out;
    }

    private static byte[] concat(byte[] a, byte[] b) {
        byte[] out = new byte[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static long fnv1a64(byte[] data) {
        long hash = 0xcbf29ce484222325L;
        for (byte b : data) {
            hash = (hash ^ (b & 0xffL)) * 0x100000001b3L;
        }
        return hash;
    }

    private static String sha256(byte[] data) throws Exception {
        return hex(MessageDigest.getInstance("SHA-256").digest(data));
    }

    private static String hex64(long value) {
        return String.format("%016x", value);
    }

    private static String hex(byte[] data) {
        StringBuilder out = new StringBuilder(data.length * 2);
        for (byte b : data) {
            out.append(String.format("%02x", b));
        }
        return out.toString();
    }

    private static byte[] hex(String s) {
        byte[] out = new byte[s.length() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Integer.parseInt(s.substring(i * 2, i * 2 + 2), 16);
        }
        return out;
    }

    private static int i32le(byte[] s, int off) {
        return (s[off] & 0xff) | ((s[off + 1] & 0xff) << 8) | ((s[off + 2] & 0xff) << 16) | (s[off + 3] << 24);
    }

    private static int u16be(byte[] s, int off) {
        return ((s[off] & 0xff) << 8) | (s[off + 1] & 0xff);
    }

    private static int u32be(byte[] s, int off) {
        return (s[off] << 24) | ((s[off + 1] & 0xff) << 16) | ((s[off + 2] & 0xff) << 8) | (s[off + 3] & 0xff);
    }

    private static long u64be(byte[] s, int off) {
        long out = 0;
        for (int i = 0; i < 8; i++) {
            out = (out << 8) | (s[off + i] & 0xffL);
        }
        return out;
    }

    private static long i64be(byte[] s, int off) {
        long out = 0;
        for (int i = 0; i < 8; i++) {
            out = (out << 8) | (s[off + i] & 0xffL);
        }
        return out;
    }

    private static Path findRepoRoot() {
        Path dir = Paths.get("").toAbsolutePath();
        for (int up = 0; up <= 6 && dir != null; up++) {
            if (Files.isRegularFile(dir.resolve("reference.json"))) {
                return dir;
            }
            dir = dir.getParent();
        }
        throw new IllegalStateException("repo root with reference.json not found");
    }

    /** The decode stream parsed into its parts. */
    static final class DecodeFacts {
        final int sampleRate;
        final int channels;
        final int bitsPerSample;
        final int frames;
        final int sampleCount;
        final int[] samples;

        private DecodeFacts(int sampleRate, int channels, int bitsPerSample, int frames,
                            int sampleCount, int[] samples) {
            this.sampleRate = sampleRate;
            this.channels = channels;
            this.bitsPerSample = bitsPerSample;
            this.frames = frames;
            this.sampleCount = sampleCount;
            this.samples = samples;
        }

        static DecodeFacts parse(byte[] s) {
            int sampleRate = u32be(s, 0);
            int channels = u16be(s, 4);
            int bits = u16be(s, 6);
            int frames = u32be(s, 8);
            int count = u32be(s, 12);
            int[] samples = new int[count];
            for (int i = 0; i < count; i++) {
                samples[i] = i32le(s, 16 + i * 4);
            }
            return new DecodeFacts(sampleRate, channels, bits, frames, count, samples);
        }
    }

    /** The signature stream parsed into its parts. */
    static final class SignatureFacts {
        final int frames;
        final int peakCount;
        final long fingerprint;
        final int[][] peaks;

        private SignatureFacts(int frames, int peakCount, long fingerprint, int[][] peaks) {
            this.frames = frames;
            this.peakCount = peakCount;
            this.fingerprint = fingerprint;
            this.peaks = peaks;
        }

        static SignatureFacts parse(byte[] s) {
            int frames = u32be(s, 0);
            int peakCount = u32be(s, 4);
            long fingerprint = u64be(s, 8);
            int[][] peaks = new int[peakCount][2];
            for (int i = 0; i < peakCount; i++) {
                peaks[i][0] = u32be(s, 16 + i * 6);
                peaks[i][1] = u16be(s, 16 + i * 6 + 4);
            }
            return new SignatureFacts(frames, peakCount, fingerprint, peaks);
        }
    }

    /** The match stream parsed into its parts. */
    static final class MatchTable {
        final int queryFrames;
        final int queryPeakCount;
        final List<Row> matches = new ArrayList<>();

        private MatchTable(int queryFrames, int queryPeakCount) {
            this.queryFrames = queryFrames;
            this.queryPeakCount = queryPeakCount;
        }

        static MatchTable parse(byte[] s) {
            MatchTable table = new MatchTable(u32be(s, 0), u32be(s, 4));
            int count = u32be(s, 8);
            for (int i = 0; i < count; i++) {
                int base = 12 + i * 20;
                table.matches.add(new Row(u32be(s, base), i64be(s, base + 4), u32be(s, base + 12), u32be(s, base + 16)));
            }
            return table;
        }

        static final class Row {
            final int id;
            final long deltaT;
            final int votes;
            final int totalVotes;

            Row(int id, long deltaT, int votes, int totalVotes) {
                this.id = id;
                this.deltaT = deltaT;
                this.votes = votes;
                this.totalVotes = totalVotes;
            }
        }
    }
}
