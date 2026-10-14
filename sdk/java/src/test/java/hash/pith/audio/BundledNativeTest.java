package hash.pith.audio;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.PosixFilePermissions;
import org.junit.jupiter.api.Test;

class BundledNativeTest {
    @Test
    void platformIdMapsOsAndArch() {
        assertEquals("linux-x64", BundledNative.platformId("Linux", "amd64"));
        assertEquals("linux-arm64", BundledNative.platformId("Linux", "aarch64"));
        assertEquals("darwin-arm64", BundledNative.platformId("Mac OS X", "aarch64"));
        assertEquals("darwin-x64", BundledNative.platformId("Mac OS X", "x86_64"));
        assertEquals("win32-x64", BundledNative.platformId("Windows 11", "amd64"));
    }

    @Test
    void unsupportedPlatformIsNamedInTheError() {
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class, () -> BundledNative.platformId("FreeBSD", "sparc"));
        assertTrue(e.getMessage().contains("freebsd-sparc"), e.getMessage());
    }

    @Test
    void extractsBundledResourceToAFreshPrivateDirectory() throws Exception {
        String first = BundledNative.extractBundled("test-platform", "fake.bin");
        String second = BundledNative.extractBundled("test-platform", "fake.bin");
        assertEquals("fake-native", Files.readString(Paths.get(first)).trim());
        assertNotEquals(first, second, "each call must use its own directory");
        Path dir = Paths.get(first).getParent();
        if (Files.getFileStore(dir).supportsFileAttributeView("posix")) {
            assertEquals("rwx------", PosixFilePermissions.toString(Files.getPosixFilePermissions(dir)));
        }
    }

    @Test
    void missingResourceReturnsNull() {
        assertNull(BundledNative.extractBundled("test-platform", "absent.bin"));
    }

    @Test
    void nothingBundledForTheRunningPlatformYieldsNull() {
        // The unit-test classpath carries no real native: extract() must fall through.
        assertNull(BundledNative.extract(new String[] {"pith_none.dll", "libpith_none.so", "libpith_none.dylib"}));
    }
}
