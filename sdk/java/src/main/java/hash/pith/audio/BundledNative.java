package hash.pith.audio;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Locale;

/**
 * Locates the native library bundled in this JAR under
 * {@code hash/pith/native/<platform>/} and extracts it to a private temp
 * directory so {@link System#load} can map it. The platform ids are the ones the
 * npm prebuilds use ({@code linux-x64}, {@code darwin-arm64}, ...).
 */
final class BundledNative {
    private static final String RESOURCE_ROOT = "/hash/pith/native/";

    private static volatile String platformNote = "";

    private BundledNative() {
    }

    /** Node-style platform id, e.g. {@code linux-x64}. */
    static String platformId(String osName, String osArch) {
        String os = osName.toLowerCase(Locale.ROOT);
        String arch = osArch.toLowerCase(Locale.ROOT);
        String normOs = os.contains("win") ? "win32"
                : (os.contains("mac") || os.contains("darwin")) ? "darwin"
                : os.contains("linux") ? "linux" : null;
        String normArch = (arch.equals("amd64") || arch.equals("x86_64")) ? "x64"
                : (arch.equals("aarch64") || arch.equals("arm64")) ? "arm64" : null;
        if (normOs == null || normArch == null) {
            throw new IllegalArgumentException(
                    "unsupported platform " + os.replace(' ', '_') + "-" + arch
                            + " (bundled natives: linux-x64, linux-arm64, darwin-arm64, darwin-x64, win32-x64)");
        }
        return normOs + "-" + normArch;
    }

    /**
     * Copies the bundled native for {@code platformId} into a fresh owner-only temp
     * directory and returns the absolute file path, or {@code null} when the JAR
     * carries no such resource. A fresh directory per call keeps two JVMs from
     * racing on one file and keeps other users from pre-creating the target.
     */
    static String extractBundled(String platformId, String fileName) {
        String resource = RESOURCE_ROOT + platformId + "/" + fileName;
        try (InputStream in = BundledNative.class.getResourceAsStream(resource)) {
            if (in == null) {
                return null;
            }
            Path dir = Files.createTempDirectory("pith-native-");
            try {
                Files.setPosixFilePermissions(dir, PosixFilePermissions.fromString("rwx------"));
            } catch (UnsupportedOperationException ignored) {
                // Windows: the per-user temp directory is already private.
            }
            Path file = dir.resolve(fileName);
            Files.copy(in, file, StandardCopyOption.REPLACE_EXISTING);
            file.toFile().deleteOnExit();
            dir.toFile().deleteOnExit();
            return file.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new LinkageError("could not extract bundled native " + resource + ": " + e, e);
        }
    }

    /**
     * Extracts the bundled native whose name fits the running OS out of
     * {@code names} (the platform file names in probe order). Returns {@code null}
     * when nothing is bundled; an unsupported platform is remembered for
     * {@link #platformNote()} instead of failing, so the checkout lookup still runs.
     */
    static String extract(String[] names) {
        String osName = System.getProperty("os.name", "");
        String os = osName.toLowerCase(Locale.ROOT);
        String suffix = os.contains("win") ? ".dll"
                : (os.contains("mac") || os.contains("darwin")) ? ".dylib" : ".so";
        String id;
        try {
            id = platformId(osName, System.getProperty("os.arch", ""));
        } catch (IllegalArgumentException e) {
            platformNote = "; " + e.getMessage();
            return null;
        }
        for (String name : names) {
            if (name.endsWith(suffix)) {
                String bundled = extractBundled(id, name);
                if (bundled != null) {
                    return bundled;
                }
            }
        }
        return null;
    }

    /** Suffix for the "cannot locate" error: names the platform when it has no bundled native. */
    static String platformNote() {
        return platformNote;
    }
}
