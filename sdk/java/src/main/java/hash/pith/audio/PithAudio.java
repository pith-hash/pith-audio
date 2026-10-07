// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
package hash.pith.audio;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Java JNI bindings for the {@code pith-audio} cdylib: RIFF/WAVE and
 * PCM fingerprinting, SplitMix64 voice synthesis, and offset-match
 * indexing — the same C ABI ({@code pith_audio_*}) the Python, Node,
 * and Go SDKs bind.
 *
 * <p>Every operation answers with the exact byte stream the
 * {@code reference.json} vectors are defined over:</p>
 *
 * <ul>
 *   <li><b>decode stream</b> — {@code [0..4)} sample rate u32 BE,
 *       {@code [4..6)} channels u16 BE, {@code [6..8)}
 *       bits-per-sample u16 BE, {@code [8..12)} frames u32 BE,
 *       {@code [12..16)} sample count u32 BE, then the decoded i32
 *       samples little-endian;</li>
 *   <li><b>signature stream</b> — {@code [0..4)} frames u32 BE,
 *       {@code [4..8)} peak count u32 BE, {@code [8..16)}
 *       fingerprint u64 BE, then one record per peak (t u32 BE,
 *       f bin u16 BE);</li>
 *   <li><b>match stream</b> — {@code [0..4)} query frames u32 BE,
 *       {@code [4..8)} query peak count u32 BE, {@code [8..12)} match
 *       count u32 BE, then one record per match (id u32 BE,
 *       delta_t i64 BE, votes u32 BE, total_votes u32 BE).</li>
 * </ul>
 *
 * <p>The synthetic voices corpus is synthesized inside the cdylib
 * ({@link #voicesPcm}); no SDK re-implements the float pipeline. PCM
 * inputs are interleaved little-endian i32 at 44 100 Hz.</p>
 *
 * <p>The cdylib is resolved once at class-load time, mirroring the
 * discovery chain of the other SDKs: {@code PITH_CDYLIB} — the
 * explicit file, or {@code PITH_CDYLIB_DIR} — a directory holding one
 * of the platform library names, or {@code target/release} at or above
 * the working directory.</p>
 */
public final class PithAudio {
    /** Status: success. */
    public static final int PITH_OK = 0;
    /** Status: the C ABI judged the input invalid. */
    public static final int PITH_E_INVALID = -1;
    /** Status: the core pipeline refused the input. */
    public static final int PITH_E_REJECTED = -2;

    private static final String[] CDYLIB_NAMES = {"pith_audio.dll", "libpith_audio.so", "libpith_audio.dylib"};
    private static final String CDYLIB_PATH = findCdylib();

    static {
        System.load(CDYLIB_PATH);
    }

    private PithAudio() { }

    /**
     * The absolute path of the loaded cdylib (tests and diagnostics).
     */
    public static String cdylibPath() {
        return CDYLIB_PATH;
    }

    /**
     * Decodes a RIFF/WAVE stream into the decode stream.
     *
     * @throws FfiError on a decode refusal (truncated file, bad
     *                   magic, unsupported format)
     */
    public static byte[] decodeWav(byte[] data) {
        int[] status = new int[1];
        byte[] stream = decodeWavNative(data, status);
        if (status[0] != PITH_OK) {
            throw new FfiError("pith_audio_decode_wav", status[0]);
        }
        return stream;
    }

    /**
     * Extracts the signature of a 44 100 Hz RIFF/WAVE file (other
     * rates are refused) into the signature stream.
     */
    public static byte[] signatureWav(byte[] data) {
        int[] status = new int[1];
        byte[] stream = signatureWavNative(data, status);
        if (status[0] != PITH_OK) {
            throw new FfiError("pith_audio_signature_wav", status[0]);
        }
        return stream;
    }

    /**
     * Extracts the signature of interleaved little-endian i32 PCM at
     * 44 100 Hz into the signature stream.
     *
     * @param channels must be non-zero and divide the byte length into
     *                 whole i32 frames
     */
    public static byte[] signaturePcm(byte[] data, int channels) {
        int[] status = new int[1];
        byte[] stream = signaturePcmNative(data, channels, status);
        if (status[0] != PITH_OK) {
            throw new FfiError("pith_audio_signature_pcm", status[0]);
        }
        return stream;
    }

    /**
     * Synthesizes the SplitMix64 square-voice corpus: {@code nSamples}
     * i32 samples, little-endian.
     */
    public static byte[] voicesPcm(long seed, int nSamples) {
        int[] status = new int[1];
        byte[] stream = voicesPcmNative(seed, nSamples, status);
        if (status[0] != PITH_OK) {
            throw new FfiError("pith_audio_voices_pcm", status[0]);
        }
        return stream;
    }

    /**
     * A signature index: signatures are appended with {@link #add},
     * queries run through {@link #match}, and the id a match reports
     * is the insertion order.
     */
    public static final class AudioIndex implements AutoCloseable {
        private long handle;

        /** Creates an empty index. */
        public AudioIndex() {
            this.handle = indexNewNative();
        }

        /**
         * Appends the signature of one PCM buffer (interleaved
         * little-endian i32). A failed add leaves the index unchanged.
         */
        public void add(byte[] pcm, int channels) {
            int[] status = new int[1];
            indexAddNative(handle, pcm, channels, status);
            if (status[0] != PITH_OK) {
                throw new FfiError("pith_audio_index_add", status[0]);
            }
        }

        /** Matches the query PCM against the index (match stream). */
        public byte[] match(byte[] pcm, int channels) {
            int[] status = new int[1];
            byte[] stream = matchNative(handle, pcm, channels, status);
            if (status[0] != PITH_OK) {
                throw new FfiError("pith_audio_match", status[0]);
            }
            return stream;
        }

        /** Releases the index; safe to call repeatedly. */
        @Override
        public void close() {
            if (handle != 0) {
                indexFreeNative(handle);
                handle = 0;
            }
        }
    }

    private static native byte[] decodeWavNative(byte[] data, int[] status);

    private static native byte[] signatureWavNative(byte[] data, int[] status);

    private static native byte[] signaturePcmNative(byte[] data, int channels, int[] status);

    private static native byte[] voicesPcmNative(long seed, int nSamples, int[] status);

    private static native long indexNewNative();

    private static native void indexAddNative(long idx, byte[] data, int channels, int[] status);

    private static native byte[] matchNative(long idx, byte[] data, int channels, int[] status);

    private static native void indexFreeNative(long idx);

    /**
     * A native refusal or failure, carrying the C ABI status code.
     */
    public static final class FfiError extends RuntimeException {
        private static final long serialVersionUID = 1L;
        /** The refusing operation (its C ABI name). */
        public final String op;
        /** The C ABI status code ({@code -1} invalid, {@code -2} rejected). */
        public final int status;

        FfiError(String op, int status) {
            super(op + " failed: status " + status);
            this.op = op;
            this.status = status;
        }
    }

    private static String findCdylib() {
        String explicitFile = System.getenv("PITH_CDYLIB");
        if (explicitFile != null && !explicitFile.isEmpty() && Files.isRegularFile(Paths.get(explicitFile))) {
            return Paths.get(explicitFile).toAbsolutePath().toString();
        }
        String explicitDir = System.getenv("PITH_CDYLIB_DIR");
        if (explicitDir != null && !explicitDir.isEmpty()) {
            for (String name : CDYLIB_NAMES) {
                Path candidate = Paths.get(explicitDir).toAbsolutePath().resolve(name);
                if (Files.isRegularFile(candidate)) {
                    return candidate.toString();
                }
            }
        }
        Path cwd = Paths.get("").toAbsolutePath();
        for (int up = 0; up <= 6; up++) {
            Path base = cwd;
            for (int i = 0; i < up; i++) {
                base = base.getParent();
                if (base == null) {
                    break;
                }
            }
            if (base == null) {
                break;
            }
            for (String name : CDYLIB_NAMES) {
                Path candidate = base.resolve(Paths.get("target", "release")).resolve(name);
                if (Files.isRegularFile(candidate)) {
                    return candidate.toString();
                }
            }
        }
        throw new LinkageError(
            "cannot locate the pith-audio cdylib; set PITH_CDYLIB or PITH_CDYLIB_DIR"
                + " (probed PITH_CDYLIB, PITH_CDYLIB_DIR, and target/release at "
                + cwd + " and its ancestors)");
    }
}
